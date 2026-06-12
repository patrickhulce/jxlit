//! Frame-scope orchestration.
//!
//! `render_frame` is the per-keyframe entry: it decodes the frame (deciding the
//! VarDCT vs Modular flow), runs the shared post-decode (loop filters, features,
//! upsampling, color-for-record), blends, applies the color transform, and
//! interleaves into the final [`DecodedImage`]. This file also holds the
//! frame-scope buffer/LF builders and the blend wrapper used by the flows.

use std::collections::HashMap;
use std::sync::Arc;

use jxl_grid::{AlignedGrid, AllocTracker};
use jxl_image::{BitDepth, ExtraChannelType};
use jxl_modular::{ChannelShift, Sample, image::TransformedModularSubimage};
use jxl_threadpool::JxlThreadPool;

use crate::pipeline::gpu::{
    Device, DeviceImage, GpuEnvironment, GpuImageWithRegion, availability, download_pixels,
    from_cpu, from_cpu_arc, into_cpu_arc, kernels, upload_pixels,
};
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{LfGlobal, LfGlobalVarDct, LfGroup};
use crate::vendor::jxl_frame::header::Encoding;
use crate::vendor::jxl_render::{
    Error, ImageBuffer, ImageWithRegion, IndexedFrame, Reference, ReferenceFrames, Region,
    RenderCache, RenderContext, Result, features,
};

use crate::pipeline::process::export::{
    analyze_planar_memcpy, export_planar_memcpy, export_planar_sample,
};
use crate::pipeline::process::interleave::{SpotColor, run_interleave};
use crate::pipeline::structs::container::ContainerCtx;
use crate::pipeline::{decode, parse, process, render};
use crate::types::{DecodeMetadata, DecodeOptions, DecodedPixels, Destination, PixelLayout};
use crate::{DecodeError, DecodedImage};

/// A rendered keyframe in planar form, plus the metadata needed to interleave.
/// Held here (rather than in `structs/`) since it is the direct output of the
/// render stage; `process::interleave` consumes it.
pub struct RenderedFrame {
    pub image: Arc<DeviceImage>,
    pub target_frame_region: Region,
    pub orientation: u32,
    pub color_bit_depth: BitDepth,
    pub extra_channels: Vec<(ExtraChannelType, BitDepth)>,
    pub is_cmyk: bool,
    pub render_spot_color: bool,
}

/// Renders the given keyframe through the staged decode flow and interleaves it
/// into an HWC `f32` [`DecodedImage`].
pub fn render_frame(
    container: &ContainerCtx,
    keyframe_index: usize,
    options: &DecodeOptions,
) -> std::result::Result<DecodedImage, DecodeError> {
    let _render = crate::phase_guard!("render");
    let ctx = &container.render_context;
    let image_header = container.declaration.image_header.as_ref();

    let idx = ctx
        .keyframe_frame_index(keyframe_index)
        .ok_or_else(|| DecodeError::new("keyframe not loaded"))?;

    let env = GpuEnvironment::current();
    let image = render_keyframe_image(ctx, idx, options, env)
        .map_err(|e| DecodeError::new(e.to_string()))?;

    let frame = &ctx.frames()[idx];
    let frame_header = frame.header();

    let image_region = ctx.image_region().apply_orientation(image_header);
    let target_frame_region = image_region.translate(-frame_header.x0, -frame_header.y0);

    let metadata = &image_header.metadata;
    let extra_channels = metadata
        .ec_info
        .iter()
        .map(|ec_info| (ec_info.ty, ec_info.bit_depth))
        .collect();

    let rendered = RenderedFrame {
        image,
        target_frame_region,
        orientation: metadata.orientation,
        color_bit_depth: metadata.bit_depth,
        extra_channels,
        is_cmyk: ctx.requested_color_encoding().is_cmyk(),
        render_spot_color: !metadata.grayscale(),
    };

    build_decoded_image(&rendered, options, env)
}

/// Fuses spot-color extra channels into RGB grids in place when possible.
fn fuse_spot_colors(
    rendered: &RenderedFrame,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> std::result::Result<(Arc<DeviceImage>, bool), DecodeError> {
    if rendered.image.as_ref().device().is_gpu()
        && availability::fuse_spot_colors_available(
            rendered.image.as_ref(),
            rendered.color_bit_depth,
            &rendered.extra_channels,
            options,
            env,
        )
    {
        return kernels::fuse_spot_colors_on_gpu(
            Arc::clone(&rendered.image),
            rendered.color_bit_depth,
            &rendered.extra_channels,
        )
        .map_err(|e| DecodeError::new(e.to_string()));
    }

    match rendered.image.as_ref() {
        DeviceImage::Cpu(_) if rendered.render_spot_color => {
            // fall through to CPU path below
        }
        _ => return Ok((Arc::clone(&rendered.image), false)),
    }

    let color_channels = rendered.image.color_channels();
    if color_channels != 3 {
        return Ok((Arc::clone(&rendered.image), false));
    }

    let has_spots = rendered
        .extra_channels
        .iter()
        .any(|(ec, _)| matches!(ec, ExtraChannelType::SpotColour { .. }));
    if !has_spots {
        return Ok((Arc::clone(&rendered.image), false));
    }

    let mut device_image = Arc::try_unwrap(Arc::clone(&rendered.image)).unwrap_or_else(|arc| {
        arc.try_clone()
            .map_err(|e| DecodeError::new(e.to_string()))
            .expect("clone rendered image for spot fusion")
    });

    let DeviceImage::Cpu(ref mut cpu_image) = device_image else {
        unreachable!();
    };

    cpu_image
        .convert_modular_color(rendered.color_bit_depth)
        .map_err(|e| DecodeError::new(e.to_string()))?;

    for (ec_idx, (ec_ty, ec_bit_depth)) in rendered.extra_channels.iter().enumerate() {
        let ExtraChannelType::SpotColour { .. } = ec_ty else {
            continue;
        };
        let spot_buf_idx = color_channels + ec_idx;

        {
            let spot_buf = &mut cpu_image.buffer_mut()[spot_buf_idx];
            spot_buf
                .convert_to_float_modular(*ec_bit_depth)
                .map_err(|e| DecodeError::new(e.to_string()))?;
        }

        let (prefix, suffix) = cpu_image.buffer_mut().split_at_mut(spot_buf_idx);
        let (color_bufs, _) = prefix.split_at_mut(color_channels);
        let spot_buf = &suffix[0];

        let spot_grid = spot_buf
            .as_float()
            .expect("spot channel must be F32 after conversion");
        let (c0, rest) = color_bufs.split_at_mut(1);
        let (c1, c2) = rest.split_at_mut(1);
        let color_grids: [&mut AlignedGrid<f32>; 3] = [
            c0[0].as_float_mut().expect("color channel 0 F32"),
            c1[0].as_float_mut().expect("color channel 1 F32"),
            c2[0].as_float_mut().expect("color channel 2 F32"),
        ];

        features::render_spot_color(color_grids, spot_grid, ec_ty)
            .map_err(|e| DecodeError::new(e.to_string()))?;
    }

    Ok((Arc::new(device_image), true))
}

struct ExportChannelSelection<'a> {
    channel_indices: Vec<usize>,
    bit_depth: Vec<BitDepth>,
    start_offset_xy: Vec<(i32, i32)>,
    grid_shifts: Vec<ChannelShift>,
    spot_colors: Vec<SpotColor<'a>>,
    has_float_sample: bool,
}

fn gather_export_channels<'a>(
    rendered: &RenderedFrame,
    device_image: &DeviceImage,
    cpu_image: Option<&'a ImageWithRegion>,
    left: i32,
    top: i32,
    spots_fused: bool,
) -> ExportChannelSelection<'a> {
    let color_channels = device_image.color_channels();
    let regions_and_shifts = device_image.regions_and_shifts();

    let mut channel_indices: Vec<usize> = (0..color_channels).collect();
    let mut bit_depth = vec![rendered.color_bit_depth; color_channels];
    let mut start_offset_xy = Vec::with_capacity(color_channels);
    let mut grid_shifts = Vec::with_capacity(color_channels);
    for (region, shift) in &regions_and_shifts[..color_channels] {
        start_offset_xy.push((left - region.left, top - region.top));
        grid_shifts.push(*shift);
    }

    if rendered.is_cmyk {
        for (ec_idx, (ec, (region, shift))) in rendered
            .extra_channels
            .iter()
            .zip(&regions_and_shifts[color_channels..])
            .enumerate()
        {
            if matches!(ec.0, ExtraChannelType::Black) {
                channel_indices.push(color_channels + ec_idx);
                bit_depth.push(ec.1);
                start_offset_xy.push((left - region.left, top - region.top));
                grid_shifts.push(*shift);
                break;
            }
        }
    }

    for (ec_idx, (ec, (region, shift))) in rendered
        .extra_channels
        .iter()
        .zip(&regions_and_shifts[color_channels..])
        .enumerate()
    {
        if matches!(ec.0, ExtraChannelType::Alpha { .. }) {
            channel_indices.push(color_channels + ec_idx);
            bit_depth.push(ec.1);
            start_offset_xy.push((left - region.left, top - region.top));
            grid_shifts.push(*shift);
            break;
        }
    }

    let mut spot_colors = Vec::new();
    if !spots_fused
        && rendered.render_spot_color
        && color_channels == 3
        && let Some(cpu_image) = cpu_image
    {
        let fb = cpu_image.buffer();
        for (ec_idx, (ec, (region, _))) in rendered
            .extra_channels
            .iter()
            .zip(&regions_and_shifts[color_channels..])
            .enumerate()
        {
            if let ExtraChannelType::SpotColour {
                red,
                green,
                blue,
                solidity,
            } = ec.0
            {
                spot_colors.push(SpotColor {
                    grid: &fb[color_channels + ec_idx],
                    start_offset_xy: (left - region.left, top - region.top),
                    bit_depth: ec.1,
                    rgb: (red, green, blue),
                    solidity,
                });
            }
        }
    }

    let has_float_sample = bit_depth
        .iter()
        .any(|bd| matches!(bd, BitDepth::FloatSample { .. }));

    ExportChannelSelection {
        channel_indices,
        bit_depth,
        start_offset_xy,
        grid_shifts,
        spot_colors,
        has_float_sample,
    }
}

enum ExportGpuInput<'a> {
    Resident(&'a GpuImageWithRegion),
    Materialized(GpuImageWithRegion),
}

impl ExportGpuInput<'_> {
    fn image(&self) -> &GpuImageWithRegion {
        match self {
            Self::Resident(g) => g,
            Self::Materialized(g) => g,
        }
    }
}

fn finalize_pixels(
    gpu: crate::pipeline::gpu::GpuPixelBuffer,
    options: &DecodeOptions,
) -> std::result::Result<DecodedPixels, DecodeError> {
    match options.destination {
        Destination::Gpu => Ok(DecodedPixels::Gpu(gpu)),
        Destination::Cpu => download_pixels(&gpu)
            .map(DecodedPixels::Cpu)
            .map_err(DecodeError::new),
    }
}

/// Computes the output layout, selects the color/black/alpha/spot channels,
/// allocates the pixel buffer, and runs the export transform, wrapping the
/// result into a [`DecodedImage`].
fn build_decoded_image(
    rendered: &RenderedFrame,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> std::result::Result<DecodedImage, DecodeError> {
    let _build_decoded_image = crate::phase_guard!("build_decoded_image");
    let orientation = rendered.orientation;
    debug_assert!((1..=8).contains(&orientation));

    let Region {
        left,
        top,
        mut width,
        mut height,
    } = rendered.target_frame_region;
    if orientation >= 5 {
        std::mem::swap(&mut width, &mut height);
    }

    let (image, spots_fused) = {
        let _fuse = crate::phase_guard!("fuse_spot_colors");
        fuse_spot_colors(rendered, options, env)?
    };

    let color_channels = image.color_channels();
    let needs_cpu_spots = !spots_fused
        && rendered.render_spot_color
        && color_channels == 3
        && rendered
            .extra_channels
            .iter()
            .any(|(ec, _)| matches!(ec, ExtraChannelType::SpotColour { .. }));

    let cpu_for_spots = if needs_cpu_spots {
        Some(into_cpu_arc(Arc::clone(&image)).map_err(|e| DecodeError::new(e.to_string()))?)
    } else {
        None
    };

    let selection = gather_export_channels(
        rendered,
        image.as_ref(),
        cpu_for_spots.as_deref(),
        left,
        top,
        spots_fused,
    );
    let channels = selection.channel_indices.len();
    let width_us = width as usize;
    let height_us = height as usize;
    let plane_size = width_us * height_us;
    let has_spot_colors = !selection.spot_colors.is_empty();

    let use_gpu_export = match options.layout {
        PixelLayout::Interleaved => availability::run_interleave_available(
            image.as_ref(),
            orientation,
            width,
            height,
            channels,
            options.layout,
            options,
            env,
            has_spot_colors,
            selection.has_float_sample,
        ),
        PixelLayout::Planar => availability::run_export_planar_available(
            image.as_ref(),
            orientation,
            width,
            height,
            channels,
            options.layout,
            options,
            env,
            has_spot_colors,
            selection.has_float_sample,
        ),
    };

    if use_gpu_export {
        let gpu_input = match image.as_ref() {
            DeviceImage::Gpu(gpu) => ExportGpuInput::Resident(gpu),
            DeviceImage::Cpu(cpu) => {
                let cpu_ref = cpu_for_spots.as_deref().unwrap_or(cpu);
                let gpu = {
                    let _upload = crate::phase_guard!("export_gpu_materialize");
                    GpuImageWithRegion::from_cpu(cpu_ref).map_err(DecodeError::new)?
                };
                ExportGpuInput::Materialized(gpu)
            }
        };
        let gpu = gpu_input.image();
        let gpu_pixels = {
            let _export = crate::phase_guard!("export_gpu");
            match options.layout {
                PixelLayout::Interleaved => kernels::run_interleave_on_gpu(
                    gpu,
                    &selection.channel_indices,
                    &selection.bit_depth,
                    &selection.start_offset_xy,
                    orientation,
                    width,
                    height,
                    channels,
                ),
                PixelLayout::Planar => kernels::run_export_planar_on_gpu(
                    gpu,
                    &selection.channel_indices,
                    &selection.bit_depth,
                    &selection.start_offset_xy,
                    orientation,
                    width,
                    height,
                    channels,
                    plane_size,
                ),
            }
        }
        .map_err(DecodeError::new)?;

        return Ok(DecodedImage {
            height: height_us,
            width: width_us,
            channels,
            pixels: finalize_pixels(gpu_pixels, options)?,
            metadata: DecodeMetadata::with_version(env!("CARGO_PKG_VERSION")),
        });
    }

    let cpu_image =
        into_cpu_arc(Arc::clone(&image)).map_err(|e| DecodeError::new(e.to_string()))?;

    if options.destination == Destination::Gpu && !use_gpu_export {
        let fb = cpu_image.buffer();
        let grids: Vec<&ImageBuffer> = selection
            .channel_indices
            .iter()
            .map(|&idx| &fb[idx])
            .collect();
        let mut pixels = vec![0.0f32; plane_size * channels];
        let count = match options.layout {
            PixelLayout::Interleaved => run_interleave(
                &mut pixels,
                &grids,
                &selection.bit_depth,
                &selection.start_offset_xy,
                &selection.spot_colors,
                orientation,
                width,
                height,
            ),
            PixelLayout::Planar => {
                if let Some(memcpy) = analyze_planar_memcpy(
                    &grids,
                    &selection.start_offset_xy,
                    &selection.grid_shifts,
                    orientation,
                    width_us,
                    height_us,
                    spots_fused,
                    has_spot_colors,
                ) {
                    export_planar_memcpy(&mut pixels, &memcpy.channels, plane_size);
                    pixels.len()
                } else {
                    export_planar_sample(
                        &mut pixels,
                        &grids,
                        &selection.bit_depth,
                        &selection.start_offset_xy,
                        &selection.spot_colors,
                        orientation,
                        width,
                        height,
                    )
                }
            }
        };
        if count != pixels.len() {
            return Err(DecodeError::new(format!(
                "expected to write {} samples, wrote {count}",
                pixels.len()
            )));
        }
        let gpu = upload_pixels(&pixels, width, height, channels as u32, options.layout)
            .map_err(DecodeError::new)?;
        return Ok(DecodedImage {
            height: height_us,
            width: width_us,
            channels,
            pixels: DecodedPixels::Gpu(gpu),
            metadata: DecodeMetadata::with_version(env!("CARGO_PKG_VERSION")),
        });
    }

    let fb = cpu_image.buffer();
    let grids: Vec<&ImageBuffer> = selection
        .channel_indices
        .iter()
        .map(|&idx| &fb[idx])
        .collect();

    let mut pixels = vec![0.0f32; plane_size * channels];

    let count = match options.layout {
        PixelLayout::Interleaved => run_interleave(
            &mut pixels,
            &grids,
            &selection.bit_depth,
            &selection.start_offset_xy,
            &selection.spot_colors,
            orientation,
            width,
            height,
        ),
        PixelLayout::Planar => {
            if let Some(memcpy) = analyze_planar_memcpy(
                &grids,
                &selection.start_offset_xy,
                &selection.grid_shifts,
                orientation,
                width_us,
                height_us,
                spots_fused,
                has_spot_colors,
            ) {
                export_planar_memcpy(&mut pixels, &memcpy.channels, plane_size);
                pixels.len()
            } else {
                export_planar_sample(
                    &mut pixels,
                    &grids,
                    &selection.bit_depth,
                    &selection.start_offset_xy,
                    &selection.spot_colors,
                    orientation,
                    width,
                    height,
                )
            }
        }
    };

    if count != pixels.len() {
        return Err(DecodeError::new(format!(
            "expected to write {} samples, wrote {count}",
            pixels.len()
        )));
    }

    Ok(DecodedImage {
        height: height_us,
        width: width_us,
        channels,
        pixels: DecodedPixels::Cpu(pixels),
        metadata: DecodeMetadata::with_version(env!("CARGO_PKG_VERSION")),
    })
}

/// Decodes the keyframe to a blended, color-transformed planar image. Forks
/// `render_by_index` + `postprocess_keyframe`: decode the color buffer, inject
/// it into the vendored blend machinery, then run the color transform.
fn render_keyframe_image(
    ctx: &RenderContext,
    idx: usize,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<Arc<DeviceImage>> {
    let _render_keyframe = crate::phase_guard!("render_keyframe");
    let frame = &ctx.frames()[idx];
    let image_region = ctx.requested_image_region();
    let frame_visibility = ctx.get_previous_frames_visibility(frame);
    let pool = ctx.pool();

    let grid = if ctx.narrow_modular() {
        let refs = ctx.reference_frames_narrow(idx);
        let mut cache = RenderCache::<i16>::new(frame);
        decode_color_buffer::<i16>(
            frame,
            refs,
            &mut cache,
            image_region,
            pool,
            frame_visibility,
            options,
            env,
        )?
    } else {
        let refs = ctx.reference_frames_wide(idx);
        let mut cache = RenderCache::<i32>::new(frame);
        decode_color_buffer::<i32>(
            frame,
            refs,
            &mut cache,
            image_region,
            pool,
            frame_visibility,
            options,
            env,
        )?
    };

    let blended = {
        let _blend = crate::phase_guard!("blend");
        run_blend(ctx, idx, grid, options, env)?
    };
    let _xyb2rgb = crate::phase_guard!("xyb2rgb");
    process::xyb2rgb::run_xyb2rgb(ctx, frame, blended, options, env)
}

/// Decodes a single frame's color buffer: dispatch to the VarDCT/Modular flow,
/// then run the shared post-decode stage (filters, features, upsampling and
/// color-for-record).
#[allow(clippy::too_many_arguments)]
fn decode_color_buffer<S: Sample>(
    frame: &IndexedFrame,
    reference_frames: ReferenceFrames<S>,
    cache: &mut RenderCache<S>,
    image_region: Region,
    pool: &JxlThreadPool,
    frame_visibility: (usize, usize),
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<DeviceImage> {
    let _decode_color_buffer = crate::phase_guard!("decode_color_buffer");
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let device = Device::select(options, frame_header, env);

    let regions = render::region::render_region(frame, image_region);
    let color_padded_region = regions.color_padded_region;
    let upsampling_valid_region = regions.upsampling_valid_region;

    let mut fb = match frame_header.encoding {
        Encoding::Modular => {
            render::flows::modular::run_modular_flow(frame, cache, color_padded_region, pool)?
        }
        Encoding::VarDct => {
            let result = render::flows::vardct::run_vardct_flow(
                device,
                frame,
                reference_frames.lf.as_ref(),
                cache,
                color_padded_region,
                pool,
                options,
                env,
            );
            match (result, reference_frames.lf) {
                (Ok(grid), _) => grid,
                (Err(e), Some(lf)) if matches!(e, Error::IncompleteFrame) || e.unexpected_eof() => {
                    let render = lf.image.run_with_image()?;
                    let render = render.blend(None, pool)?;
                    let mut render = render.upsample_lf(1)?;
                    render.fill_opaque_alpha(&image_header.metadata.ec_info);
                    from_cpu(render)
                }
                (Err(e), _) => return Err(e),
            }
        }
    };

    if frame_header.do_ycbcr {
        let _jpeg_upsample = crate::phase_guard!("jpeg_upsample");
        process::upsample::run_jpeg_upsample(
            &mut fb,
            color_padded_region,
            image_header.metadata.bit_depth,
            options,
            env,
        )?;
    }

    let color_channels = fb.color_channels();
    {
        let _loop_filters = crate::phase_guard!("loop_filters");
        process::filters::run_loop_filters(
            frame,
            &mut fb,
            color_padded_region,
            &cache.lf_groups,
            pool,
            options,
            env,
        )?;
    }
    fb.remove_color_channels(color_channels);
    fb.prepare_color_upsampling(frame_header);
    {
        let _features = crate::phase_guard!("features");
        process::features::run_features(
            frame,
            &mut fb,
            upsampling_valid_region,
            reference_frames.refs.clone(),
            cache.lf_global.as_ref(),
            frame_visibility.0,
            frame_visibility.1,
            pool,
            options,
            env,
        )?;
    }
    {
        let _nonsep_upsample = crate::phase_guard!("nonsep_upsample");
        process::upsample::run_nonseparable_upsample(
            &mut fb,
            image_header,
            frame_header,
            upsampling_valid_region,
            options,
            env,
        )?;
    }
    if !frame_header.save_before_ct && !frame_header.is_last {
        let _color_for_record = crate::phase_guard!("color_for_record");
        process::xyb2rgb::run_color_for_record(
            image_header,
            frame_header.do_ycbcr,
            &mut fb,
            pool,
            options,
            env,
        )?;
    }

    Ok(fb)
}

/// Allocates the (empty) XYB high-frequency coefficient buffer that pass-group
/// decode writes into, shaped to the modular region and channel shifts.
pub fn build_coefficient_buffer_cpu(
    frame_header: &FrameHeader,
    modular_region: Region,
    tracker: Option<&AllocTracker>,
) -> Result<ImageWithRegion> {
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let Region { width, height, .. } = modular_region;

    let mut color_buffer = ImageWithRegion::new(3, tracker);
    for shift in shifts_cbycr {
        let (w8, h8) = shift.shift_size((width.div_ceil(8), height.div_ceil(8)));
        let width = w8 * 8;
        let height = h8 * 8;
        let buffer = AlignedGrid::with_alloc_tracker(width as usize, height as usize, tracker)?;
        color_buffer.append_channel_shifted(ImageBuffer::F32(buffer), modular_region, shift);
    }
    Ok(color_buffer)
}

/// Reads the LF groups and produces the dequantized low-frequency image, either
/// from this frame's LF data (with chroma-from-luma + adaptive smoothing) or
/// from a referenced LF frame.
#[allow(clippy::too_many_arguments)]
pub fn build_low_frequency_image<S: Sample>(
    device: Device,
    frame: &IndexedFrame,
    low_frequency_global: &LfGlobal<S>,
    low_frequency_groups: &mut HashMap<u32, LfGroup<S>>,
    modular_lf_groups: Vec<TransformedModularSubimage<S>>,
    modular_lf_region: Region,
    lf_frame: Option<&Reference<S>>,
    low_frequency_global_vardct: &LfGlobalVarDct,
    subsampled: bool,
    pool: &JxlThreadPool,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<DeviceImage> {
    let low_frequency_image = parse::frames::read_low_frequency_groups(
        frame,
        low_frequency_global,
        low_frequency_groups,
        modular_lf_groups,
        modular_lf_region,
        pool,
    )?;

    if let Some(lf) = lf_frame {
        let lf_frame = Arc::clone(&lf.image).run_with_image()?;
        Ok(from_cpu(lf_frame.blend(None, pool)?.try_clone()?))
    } else {
        let mut low_frequency_image = low_frequency_image.unwrap();
        decode::dequant::run_low_frequency_dequant_cpu(
            &mut low_frequency_image,
            &low_frequency_global.lf_dequant,
            &low_frequency_global_vardct.quantizer,
            &low_frequency_global_vardct.lf_chan_corr,
            subsampled,
            frame.header().flags.skip_adaptive_lf_smoothing(),
        )?;
        let use_gpu = device.is_gpu()
            && availability::build_low_frequency_image_available(frame.header(), options, env);
        Ok(match if use_gpu { Device::Gpu } else { Device::Cpu } {
            Device::Cpu => from_cpu(low_frequency_image),
            Device::Gpu => DeviceImage::Gpu(
                GpuImageWithRegion::from_cpu(&low_frequency_image)
                    .map_err(|_| Error::NotSupported("GPU LF image materialization failed"))?,
            ),
        })
    }
}

/// Blends the decoded frame onto the canvas via the vendored staged blender.
pub fn run_blend(
    ctx: &RenderContext,
    idx: usize,
    grid: DeviceImage,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<Arc<DeviceImage>> {
    if availability::run_blend_available(ctx, idx, &grid, options, env) {
        return kernels::run_blend_on_gpu(ctx, idx, grid);
    }

    let DeviceImage::Cpu(image) = grid else {
        unreachable!("blend GPU fallback requires CPU-resident image");
    };
    let blended = ctx.blend_staged(idx, image)?;
    Ok(from_cpu_arc(blended))
}
