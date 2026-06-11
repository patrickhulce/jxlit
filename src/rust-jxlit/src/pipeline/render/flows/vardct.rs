//! VarDCT frame-decode flow.
//!
//! Thin orchestrator that wires the staged VarDCT path together: read the
//! frame-global entropy structures, build the coefficient buffer and LF image,
//! enumerate tiles, run per-tile ANS decode (inline), then the per-tile
//! transforms, and finally finish the modular sub-image. Behavior is kept
//! byte-for-byte equivalent to the vendored `render_vardct`; only the
//! organization differs. The inline pass-group loop plus the per-tile transform
//! (`render::tile::render_tile`) are the future GPU offload region.

use std::sync::RwLock;

use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::pipeline::gpu::{
    Device, DeviceCoefficients, DeviceImage, GpuEnvironment, build_coefficient_buffer,
};
use crate::types::DecodeOptions;
use crate::vendor::jxl_frame::data::{PassGroupParams, PassGroupParamsVardct};
use crate::vendor::jxl_render::{Error, IndexedFrame, Reference, Region, RenderCache, Result};

use crate::pipeline::structs::frame::{FrameCtx, FrameDeclaration};
use crate::pipeline::{decode, parse, render};

#[allow(clippy::too_many_arguments)]
pub fn run_vardct_flow<S: Sample>(
    device: Device,
    frame: &IndexedFrame,
    lf_frame: Option<&Reference<S>>,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &JxlThreadPool,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<DeviceImage> {
    let _vardct_flow = crate::phase_guard!("vardct_flow");
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();

    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

    // Frame-global LF metadata (cached across progressive passes).
    let low_frequency_global = match &cache.lf_global {
        Some(x) if !x.gmodular.is_partial() => x,
        _ => {
            let _read_lf_global = crate::phase_guard!("read_lf_global");
            let low_frequency_global = parse::frames::read_low_frequency_global(frame)?;
            cache.lf_global = Some(low_frequency_global);
            cache.lf_global.as_ref().unwrap()
        }
    };
    if lf_frame.is_some() && low_frequency_global.gmodular.is_partial() {
        return Err(Error::IncompleteFrame);
    }

    let mut gmodular = low_frequency_global.gmodular.try_clone()?;
    let low_frequency_global_vardct = low_frequency_global.vardct.as_ref().unwrap();

    let width = frame_header.color_sample_width() as usize;
    let height = frame_header.color_sample_height() as usize;
    let (width_rounded, height_rounded) = {
        let mut bw = width.div_ceil(8);
        let mut bh = height.div_ceil(8);
        let h_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 2);
        let v_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 3);
        if h_upsample {
            bw = bw.div_ceil(2) * 2;
        }
        if v_upsample {
            bh = bh.div_ceil(2) * 2;
        }
        (bw * 8, bh * 8)
    };

    let regions = render::region::build_vardct_regions(
        frame_header,
        region,
        width_rounded,
        height_rounded,
        &gmodular,
    );
    let modular_lf_region = regions.modular_lf_region;

    let frame_declaration = FrameDeclaration {
        frame_header,
        image_header,
        group_dim: frame_header.group_dim(),
        subsampled,
        aligned_region: regions.aligned_region,
        modular_region: regions.modular_region,
        tiles: parse::tiles::build_tiles(frame_header),
    };

    let mut modular_image = gmodular.modular.image_mut();
    let groups = modular_image
        .as_mut()
        .map(|x| x.prepare_groups(frame.pass_shifts()))
        .transpose()?;
    let (lf_group_image, pass_group_image) = groups.map(|x| (x.lf_groups, x.pass_groups)).unzip();
    let lf_group_image = lf_group_image.unwrap_or_else(Vec::new);
    let pass_group_image = pass_group_image.unwrap_or_else(|| {
        let passes = frame_header.passes.num_passes as usize;
        let mut ret = Vec::with_capacity(passes);
        ret.resize_with(passes, Vec::new);
        ret
    });

    let high_frequency_global_slot = &mut cache.hf_global;
    let low_frequency_groups_mut = &mut cache.lf_groups;

    let result = RwLock::new(Result::Ok(()));
    let low_frequency_image = pool.scope(|scope| -> Result<_> {
        if high_frequency_global_slot.is_none() {
            scope.spawn(|_| {
                let ret = (|| -> Result<_> {
                    *high_frequency_global_slot =
                        parse::frames::read_high_frequency_global(frame, low_frequency_global)?;
                    Ok(())
                })();
                if let Err(e) = ret {
                    *result.write().unwrap() = Err(e);
                }
            });
        }

        let low_frequency_image = {
            let _build_lf_image = crate::phase_guard!("build_lf_image");
            render::frame::build_low_frequency_image(
                device,
                frame,
                low_frequency_global,
                low_frequency_groups_mut,
                lf_group_image,
                modular_lf_region,
                lf_frame,
                low_frequency_global_vardct,
                subsampled,
                pool,
                options,
                env,
            )?
        };

        Ok(low_frequency_image)
    })?;
    result.into_inner().unwrap()?;

    let mut color_buffer = {
        let _build_coefficient_buffer = crate::phase_guard!("build_coefficient_buffer");
        build_coefficient_buffer(
            device,
            frame_header,
            frame_declaration.modular_region,
            tracker,
            options,
            env,
        )?
    };

    let high_frequency_global = cache.hf_global.as_ref();
    let low_frequency_groups = &cache.lf_groups;

    let mut tiles = parse::tiles::build_tile_contexts(
        &frame_declaration.tiles,
        &mut color_buffer,
        frame_header,
        low_frequency_groups,
    );

    (|| -> Result<()> {
        let _tile_entropy = crate::phase_guard!("tile_entropy");
        let Some(high_frequency_global) = high_frequency_global else {
            return Ok(());
        };

        let result = RwLock::new(Result::Ok(()));

        for (pass_idx, pass_image) in pass_group_image.into_iter().enumerate() {
            let pass_idx = pass_idx as u32;

            pool.scope(|scope| {
                let global_ma_config = gmodular.ma_config.as_ref();

                let mut image_it = pass_image.into_iter().enumerate();
                for tile in &mut tiles {
                    let group_idx = tile.declaration.group_index;
                    let lf_group = tile.low_frequency_group;
                    if lf_group.hf_meta.is_none() {
                        continue;
                    }

                    let bitstream = match frame.pass_group_bitstream(pass_idx, group_idx) {
                        Some(Ok(bitstream)) => bitstream,
                        Some(Err(e)) => {
                            *result.write().unwrap() = Err(e.into());
                            continue;
                        }
                        None => continue,
                    };
                    let allow_partial = bitstream.partial;
                    let mut bitstream = bitstream.bitstream;

                    let modular = image_it
                        .find(|(image_idx, _)| *image_idx == group_idx as usize)
                        .map(|(_, modular)| modular);

                    let result = &result;
                    match &mut tile.xyb_coefficients {
                        DeviceCoefficients::Cpu([x, y, b]) => {
                            let mut xyb_coefficients =
                                [x, y, b].map(|grid| grid.borrow_mut().into_i32());
                            scope.spawn(move |_| {
                                let vardct = Some(PassGroupParamsVardct {
                                    lf_vardct: low_frequency_global_vardct,
                                    hf_global: high_frequency_global,
                                    hf_coeff_output: &mut xyb_coefficients,
                                });

                                let r = decode::entropy::read_pass_group(
                                    Device::Cpu,
                                    &mut bitstream,
                                    PassGroupParams {
                                        frame_header,
                                        lf_group,
                                        pass_idx,
                                        group_idx,
                                        global_ma_config,
                                        modular,
                                        vardct,
                                        allow_partial,
                                        tracker,
                                        pool,
                                    },
                                    group_idx,
                                    pass_idx,
                                    options,
                                    env,
                                );
                                if !allow_partial && r.is_err() {
                                    *result.write().unwrap() = r.map_err(From::from);
                                }
                            });
                        }
                        DeviceCoefficients::Gpu(_) => {
                            scope.spawn(move |_| {
                                let r = decode::entropy::read_pass_group(
                                    Device::Gpu,
                                    &mut bitstream,
                                    PassGroupParams {
                                        frame_header,
                                        lf_group,
                                        pass_idx,
                                        group_idx,
                                        global_ma_config,
                                        modular,
                                        vardct: None,
                                        allow_partial,
                                        tracker,
                                        pool,
                                    },
                                    group_idx,
                                    pass_idx,
                                    options,
                                    env,
                                );
                                if !allow_partial && r.is_err() {
                                    *result.write().unwrap() = r.map_err(From::from);
                                }
                            });
                        }
                    }
                }
            });
        }

        result.into_inner().unwrap()
    })()?;

    let frame_ctx = FrameCtx {
        device,
        options: *options,
        env,
        frame_header: frame_declaration.frame_header,
        image_header: frame_declaration.image_header,
        low_frequency_global,
        low_frequency_global_vardct,
        high_frequency_global,
        low_frequency_groups,
        low_frequency_image,
        aligned_region: frame_declaration.aligned_region,
        group_dim: frame_declaration.group_dim,
        subsampled: frame_declaration.subsampled,
    };
    {
        let _tile_transform = crate::phase_guard!("tile_transform");
        pool.for_each_vec(tiles, |tile| {
            render::tile::render_tile(&frame_ctx, tile);
        });
    }

    if let Some(modular_image) = modular_image {
        let _finish_modular = crate::phase_guard!("finish_modular");
        modular_image.prepare_subimage().unwrap().finish(pool);
        color_buffer.extend_from_gmodular(gmodular);
    }

    Ok(color_buffer)
}
