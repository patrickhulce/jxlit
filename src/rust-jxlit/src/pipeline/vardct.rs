//! VarDCT frame-decode path.
//!
//! This is a fork of the vendored `jxl_render::vardct::render_vardct`, kept
//! byte-for-byte equivalent in behavior but reorganized so the three logical
//! phases are routed through the staged modules:
//!
//! * entropy decode  -> [`crate::pipeline::render::entropy`]
//! * dequantization  -> [`crate::pipeline::render::dequant`]
//! * inverse DCT     -> [`crate::pipeline::render::idct`]
//!
//! The surrounding scaffolding (region math, modular groups, coefficient buffer
//! construction, thread scoping) is unchanged from the upstream renderer.

use jxl_grid::AlignedGrid;
use jxl_modular::{ChannelShift, Sample};
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::data::{PassGroupParams, PassGroupParamsVardct};
use crate::vendor::jxl_render::{
    Error, ImageBuffer, ImageWithRegion, IndexedFrame, Reference, Region, RenderCache, Result,
    modular,
};

use crate::pipeline::render::{dequant, entropy, idct};

pub(crate) fn decode<S: Sample>(
    frame: &IndexedFrame,
    lf_frame: Option<&Reference<S>>,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &JxlThreadPool,
) -> Result<ImageWithRegion> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();

    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

    let lf_global = match &cache.lf_global {
        Some(x) if !x.gmodular.is_partial() => x,
        _ => {
            let lf_global = entropy::parse_lf_global(frame)?;
            cache.lf_global = Some(lf_global);
            cache.lf_global.as_ref().unwrap()
        }
    };
    if lf_frame.is_some() && lf_global.gmodular.is_partial() {
        return Err(Error::IncompleteFrame);
    }

    let mut gmodular = lf_global.gmodular.try_clone()?;
    let lf_global_vardct = lf_global.vardct.as_ref().unwrap();

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

    let aligned_region = region.container_aligned(frame_header.group_dim());
    let aligned_lf_region = {
        // group_dim is multiple of 8
        let aligned_region_div8 = Region {
            left: aligned_region.left / 8,
            top: aligned_region.top / 8,
            width: aligned_region.width / 8,
            height: aligned_region.height / 8,
        };
        if frame_header.flags.skip_adaptive_lf_smoothing() {
            aligned_region_div8
        } else {
            aligned_region_div8.pad(1)
        }
        .container_aligned(frame_header.group_dim())
    };

    let aligned_region = aligned_region.intersection(Region::with_size(
        width_rounded as u32,
        height_rounded as u32,
    ));
    let aligned_lf_region = aligned_lf_region.intersection(Region::with_size(
        width_rounded as u32 / 8,
        height_rounded as u32 / 8,
    ));
    let modular_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_region, false);
    let modular_lf_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_lf_region, true)
            .intersection(Region::with_size(
                width_rounded as u32 / 8,
                height_rounded as u32 / 8,
            ));

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

    let hf_global = &mut cache.hf_global;
    let lf_groups = &mut cache.lf_groups;
    let group_dim = frame_header.group_dim();

    let result = std::sync::RwLock::new(Result::Ok(()));
    let (mut fb, lf_xyb) = pool.scope(|scope| -> Result<_> {
        if hf_global.is_none() {
            scope.spawn(|_| {
                let ret = (|| -> Result<_> {
                    *hf_global = entropy::parse_hf_global(frame, lf_global)?;
                    Ok(())
                })();
                if let Err(e) = ret {
                    *result.write().unwrap() = Err(e);
                }
            });
        }

        let lf_xyb = entropy::load_lf_groups(
            frame,
            lf_global,
            lf_groups,
            lf_group_image,
            modular_lf_region,
            pool,
        )?;

        let lf_xyb = if let Some(x) = lf_frame {
            let lf_frame = std::sync::Arc::clone(&x.image).run_with_image()?;

            lf_frame.blend(None, pool)?.try_clone()?
        } else {
            let mut lf_xyb = lf_xyb.unwrap();

            dequant::prepare_lf(
                &mut lf_xyb,
                &lf_global.lf_dequant,
                &lf_global_vardct.quantizer,
                &lf_global_vardct.lf_chan_corr,
                subsampled,
                frame_header.flags.skip_adaptive_lf_smoothing(),
            )?;

            lf_xyb
        };

        let fb = {
            let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
                ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
            });
            let Region { width, height, .. } = modular_region;

            let mut fb = ImageWithRegion::new(3, tracker);
            for shift in shifts_cbycr {
                let (w8, h8) = shift.shift_size((width.div_ceil(8), height.div_ceil(8)));
                let width = w8 * 8;
                let height = h8 * 8;
                let buffer =
                    AlignedGrid::with_alloc_tracker(width as usize, height as usize, tracker)?;
                fb.append_channel_shifted(ImageBuffer::F32(buffer), modular_region, shift);
            }
            fb
        };

        Ok((fb, lf_xyb))
    })?;
    result.into_inner().unwrap()?;

    let hf_global = cache.hf_global.as_ref();
    let lf_groups = &mut cache.lf_groups;

    let mut it = fb
        .color_groups_with_group_id(frame_header)
        .into_iter()
        .filter_map(|(group_idx, grid_xyb)| {
            let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
            let lf_group = lf_groups.get(&lf_group_idx)?;

            Some((group_idx, grid_xyb, lf_group))
        })
        .collect::<Vec<_>>();

    // Decode PassGroup (entropy).
    (|| -> Result<()> {
        let Some(hf_global) = hf_global else {
            return Ok(());
        };

        let result = std::sync::RwLock::new(Result::Ok(()));

        for (pass_idx, pass_image) in pass_group_image.into_iter().enumerate() {
            let pass_idx = pass_idx as u32;

            pool.scope(|scope| {
                let global_ma_config = gmodular.ma_config.as_ref();

                let mut image_it = pass_image.into_iter().enumerate();
                for &mut (group_idx, ref mut grid_xyb, lf_group) in &mut it {
                    if lf_group.hf_meta.is_none() {
                        continue;
                    }

                    let mut grid_xyb = {
                        let [x, y, b] = grid_xyb;
                        [x, y, b].map(|grid| grid.borrow_mut().into_i32())
                    };

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
                    scope.spawn(move |_| {
                        let vardct = Some(PassGroupParamsVardct {
                            lf_vardct: lf_global_vardct,
                            hf_global,
                            hf_coeff_output: &mut grid_xyb,
                        });

                        let r = entropy::decode_pass_group(
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
                        );
                        if !allow_partial && r.is_err() {
                            *result.write().unwrap() = r.map_err(From::from);
                        }
                    });
                }
            });
        }

        result.into_inner().unwrap()
    })()?;

    // Dequant and transform (dequant -> inverse DCT).
    {
        let groups_per_row = frame_header.groups_per_row();

        pool.for_each_vec(it, |job| {
            let (group_idx, mut grid_xyb, lf_group) = job;
            let grid_xyb = &mut grid_xyb;
            let group_x = group_idx % groups_per_row;
            let group_y = group_idx / groups_per_row;

            let transform_hf = {
                let left = group_x * group_dim;
                let top = group_y * group_dim;

                let group_region = Region {
                    left: left as i32,
                    top: top as i32,
                    width: group_dim,
                    height: group_dim,
                };
                !group_region.intersection(aligned_region).is_empty()
            };

            if lf_group.hf_meta.is_none() || hf_global.is_none() || !transform_hf {
                idct::transform_group(&lf_xyb, grid_xyb, group_idx, frame_header, lf_groups);
                return;
            }

            let hf_global = hf_global.unwrap();

            dequant::dequant_group(
                grid_xyb,
                group_idx,
                image_header,
                frame_header,
                lf_global,
                lf_groups,
                hf_global,
            );

            if !subsampled {
                let hf_meta = lf_group.hf_meta.as_ref().unwrap();
                let lf_chan_corr = &lf_global_vardct.lf_chan_corr;
                let cfl_base_x = ((group_x % 8) * group_dim / 64) as usize;
                let cfl_base_y = ((group_y % 8) * group_dim / 64) as usize;
                let gw = grid_xyb[0].width().div_ceil(64);
                let gh = grid_xyb[0].height().div_ceil(64);
                let x_from_y = hf_meta
                    .x_from_y
                    .as_subgrid()
                    .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                let b_from_y = hf_meta
                    .b_from_y
                    .as_subgrid()
                    .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                dequant::chroma_from_luma_hf(grid_xyb, &x_from_y, &b_from_y, lf_chan_corr);
            }

            idct::transform_group(&lf_xyb, grid_xyb, group_idx, frame_header, lf_groups);
        });
    }

    if let Some(modular_image) = modular_image {
        modular_image.prepare_subimage().unwrap().finish(pool);
        fb.extend_from_gmodular(gmodular);
    }

    Ok(fb)
}
