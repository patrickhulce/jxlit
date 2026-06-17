use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

#[cfg(feature = "gpu")]
use std::sync::Arc;

use crate::{DecodeOptions, Hardware, PixelLayout, decode, decode_with_options};
#[cfg(feature = "gpu")]
use crate::{
    Destination,
    pipeline::gpu::{GpuEnvironment, download_pixels},
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn assets_dir() -> PathBuf {
    repo_root().join("assets")
}

#[derive(Debug, Deserialize)]
struct ManifestFixture {
    slug: String,
    jxl: String,
    reference_exr: String,
    mae_tolerance: f32,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<ManifestFixture>,
}

fn load_manifest() -> Manifest {
    let path = assets_dir().join("manifest.json");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

fn parse_pfm_rgb_f32(data: &[u8]) -> (usize, usize, Vec<f32>) {
    let first_end = data
        .iter()
        .position(|&b| b == b'\n')
        .expect("PF magic line");
    assert_eq!(&data[..first_end], b"PF");

    let second_start = first_end + 1;
    let second_end = data[second_start..]
        .iter()
        .position(|&b| b == b'\n')
        .expect("dimensions line")
        + second_start;
    let dims = std::str::from_utf8(&data[second_start..second_end]).expect("utf8 dims");
    let mut parts = dims.split_whitespace();
    let width: usize = parts.next().expect("width").parse().expect("width");
    let height: usize = parts.next().expect("height").parse().expect("height");

    let third_start = second_end + 1;
    let third_end = data[third_start..]
        .iter()
        .position(|&b| b == b'\n')
        .expect("endianness line")
        + third_start;
    let scale = std::str::from_utf8(&data[third_start..third_end]).expect("utf8 scale");
    assert!(
        scale.starts_with('-'),
        "expected little-endian PFM scale, got {scale}"
    );

    let pixel_offset = third_end + 1;
    let expected_len = width * height * 3 * 4;
    assert_eq!(
        data.len() - pixel_offset,
        expected_len,
        "unexpected PFM payload length"
    );

    let mut pixels = Vec::with_capacity(width * height * 3);
    for chunk in data[pixel_offset..].chunks_exact(4) {
        pixels.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    (height, width, pixels)
}

fn load_reference_pfm_f32(exr_path: &Path) -> (usize, usize, Vec<f32>) {
    let output = Command::new("uv")
        .args([
            "run",
            "scripts/exr_to_pfm.py",
            exr_path.to_str().expect("utf8 exr path"),
            "--stdout",
        ])
        .current_dir(repo_root())
        .output()
        .unwrap_or_else(|err| panic!("run exr_to_pfm for {}: {err}", exr_path.display()));

    assert!(
        output.status.success(),
        "exr_to_pfm failed for {}: {}",
        exr_path.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    parse_pfm_rgb_f32(&output.stdout)
}

fn load_png_rgb_f32(path: &PathBuf) -> (usize, usize, usize, Vec<f32>) {
    let decoder = png::Decoder::new(fs::File::open(path).expect("open png"));
    let mut reader = decoder.read_info().expect("read png info");
    let width = reader.info().width as usize;
    let height = reader.info().height as usize;
    let channels = match reader.info().color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unsupported png color type: {other:?}"),
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut buf).expect("read png frame");

    let pixels = buf
        .iter()
        .map(|&v| f32::from(v) / 255.0)
        .collect::<Vec<_>>();

    (height, width, channels, pixels)
}

fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

#[test]
fn decode_rejects_invalid_input() {
    let err = decode(b"not-a-jxl").unwrap_err();
    assert!(!err.to_string().is_empty());
}

#[test]
fn decode_colorspace_fixtures_match_manifest_references() {
    let root = repo_root();
    for fixture in load_manifest().fixtures {
        let jxl_path = root.join(&fixture.jxl);
        let exr_path = root.join(&fixture.reference_exr);

        let jxl_bytes =
            fs::read(&jxl_path).unwrap_or_else(|err| panic!("read {}: {err}", jxl_path.display()));
        let decoded =
            decode(&jxl_bytes).unwrap_or_else(|err| panic!("decode {}: {err}", fixture.slug));

        let (ref_height, ref_width, ref_pixels) = load_reference_pfm_f32(&exr_path);

        assert_eq!(decoded.height, ref_height, "{}", fixture.slug);
        assert_eq!(decoded.width, ref_width, "{}", fixture.slug);
        assert_eq!(decoded.channels, 3, "{}", fixture.slug);
        assert_eq!(
            decoded.pixels.as_cpu().expect("cpu pixels").len(),
            ref_pixels.len(),
            "{}",
            fixture.slug
        );

        let mae = mean_abs_error(decoded.pixels.as_cpu().expect("cpu pixels"), &ref_pixels);
        assert!(
            mae < fixture.mae_tolerance,
            "{} mean absolute error {mae} exceeds tolerance {}",
            fixture.slug,
            fixture.mae_tolerance
        );
    }
}

#[test]
fn decode_colors_fixture_is_close_to_png() {
    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let png_path = assets.join("colors.png");

    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");
    let decoded = decode(&jxl_bytes).expect("decode jxl fixture");

    let (png_height, png_width, png_channels, png_pixels) = load_png_rgb_f32(&png_path);

    assert_eq!(decoded.height, png_height);
    assert_eq!(decoded.width, png_width);
    assert_eq!(decoded.channels, png_channels);
    assert_eq!(decoded.pixels.len(), png_pixels.len());

    let mae = mean_abs_error(decoded.pixels.as_cpu().expect("cpu pixels"), &png_pixels);
    assert!(
        mae < 0.02,
        "mean absolute error {mae} exceeds tolerance 0.02"
    );
}

#[test]
fn decode_metadata_includes_version() {
    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");

    let decoded = decode(&jxl_bytes).expect("decode jxl fixture");
    assert_eq!(decoded.metadata.jxlit.version, env!("CARGO_PKG_VERSION"));
    assert!(decoded.metadata.jxlit.telemetry.is_none());
}

#[test]
fn decode_planar_colors_fixture_uses_memcpy_export() {
    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");

    let decoded = decode_with_options(
        &jxl_bytes,
        &DecodeOptions {
            layout: PixelLayout::Planar,
            telemetry: true,
            ..DecodeOptions::default()
        },
    )
    .expect("decode jxl fixture");

    let telemetry = decoded
        .metadata
        .jxlit
        .telemetry
        .as_ref()
        .expect("telemetry enabled");
    let names: Vec<_> = telemetry.measures.iter().map(|m| m.name).collect();
    assert!(
        names.contains(&"export_planar_memcpy"),
        "expected export_planar_memcpy in telemetry, got {names:?}"
    );
}

#[test]
fn decode_gpu_hardware_falls_back_to_cpu_without_panic() {
    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");

    let decoded = decode_with_options(
        &jxl_bytes,
        &DecodeOptions {
            hardware: Hardware::Gpu,
            ..DecodeOptions::default()
        },
    )
    .expect("GPU hardware request should fall back to CPU decode");

    assert!(decoded.width > 0);
    assert!(decoded.height > 0);
    assert_eq!(decoded.channels, 3);
    assert_eq!(
        decoded.pixels.len(),
        decoded.width * decoded.height * decoded.channels
    );
}

#[test]
fn decode_telemetry_collects_flat_measures() {
    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");

    let decoded = decode_with_options(
        &jxl_bytes,
        &DecodeOptions {
            telemetry: true,
            ..DecodeOptions::default()
        },
    )
    .expect("decode jxl fixture");

    let telemetry = decoded
        .metadata
        .jxlit
        .telemetry
        .as_ref()
        .expect("telemetry enabled");
    assert!(telemetry.rust_timebase > 0.0);
    assert!(telemetry.total_ms > 0.0);
    assert!(!telemetry.measures.is_empty());

    let names: Vec<_> = telemetry.measures.iter().map(|m| m.name).collect();
    assert!(names.contains(&"decode"));
    assert!(names.contains(&"parse"));
    assert!(names.contains(&"render"));

    for measure in &telemetry.measures {
        assert!(measure.start_ms <= telemetry.total_ms);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn decode_gpu_export_matches_cpu_pixels() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping decode_gpu_export_matches_cpu_pixels: no GPU adapter");
        return;
    }

    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");

    let cpu = decode_with_options(
        &jxl_bytes,
        &DecodeOptions {
            hardware: Hardware::Gpu,
            destination: Destination::Cpu,
            ..DecodeOptions::default()
        },
    )
    .expect("GPU decode with CPU destination");

    let cpu_default = decode(&jxl_bytes).expect("default CPU decode");
    let cpu_pixels = cpu.pixels.as_cpu().expect("cpu pixels");
    let default_pixels = cpu_default.pixels.as_cpu().expect("cpu pixels");
    assert_eq!(cpu_pixels.len(), default_pixels.len());

    let mae = mean_abs_error(cpu_pixels, default_pixels);
    assert!(
        mae < 5e-5,
        "GPU export CPU destination MAE {mae} exceeds tolerance"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn decode_gpu_destination_returns_gpu_handle() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping decode_gpu_destination_returns_gpu_handle: no GPU adapter");
        return;
    }

    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");

    let decoded = decode_with_options(
        &jxl_bytes,
        &DecodeOptions {
            hardware: Hardware::Gpu,
            destination: Destination::Gpu,
            ..DecodeOptions::default()
        },
    )
    .expect("GPU decode with GPU destination");

    let gpu = match decoded.pixels {
        crate::DecodedPixels::Gpu(g) => g,
        crate::DecodedPixels::Cpu(_) => panic!("expected GPU pixel buffer"),
    };
    assert_eq!(gpu.len(), decoded.width * decoded.height * decoded.channels);

    let downloaded = download_pixels(&gpu).expect("download GPU pixels");
    let reference_decoded = decode(&jxl_bytes).expect("reference decode");
    let reference = reference_decoded
        .pixels
        .as_cpu()
        .expect("reference cpu pixels");
    let mae = mean_abs_error(&downloaded, reference);
    assert!(
        mae < 5e-5,
        "GPU destination download MAE {mae} exceeds tolerance"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn decode_gpu_full_tail_matches_cpu_pixels() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping decode_gpu_full_tail_matches_cpu_pixels: no GPU adapter");
        return;
    }

    let assets = assets_dir();
    let jxl_path = assets.join("colors_e1_d0p5_fd4.jxl");
    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");

    let reference = decode(&jxl_bytes).expect("reference decode");
    let gpu = decode_with_options(
        &jxl_bytes,
        &DecodeOptions {
            hardware: Hardware::Gpu,
            destination: Destination::Cpu,
            ..DecodeOptions::default()
        },
    )
    .expect("GPU tail decode");

    let mae = mean_abs_error(
        gpu.pixels.as_cpu().expect("cpu pixels"),
        reference.pixels.as_cpu().expect("reference pixels"),
    );
    assert!(mae < 0.02, "GPU full tail MAE {mae} exceeds tolerance 0.02");
}

#[cfg(feature = "gpu")]
#[test]
fn fuse_spot_colors_gpu_matches_cpu() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping fuse_spot_colors_gpu_matches_cpu: no GPU adapter");
        return;
    }

    use jxl_grid::AlignedGrid;
    use jxl_image::BitDepth;

    use crate::pipeline::gpu::{DeviceImage, GpuImageWithRegion, kernels};
    use crate::vendor::jxl_render::{
        ImageBuffer, ImageWithRegion, Region, features::render_spot_color,
    };

    const W: usize = 16;
    const H: usize = 16;
    let len = W * H;

    let mut cpu_image = ImageWithRegion::new(3, None);
    for c in 0..3 {
        let mut grid = AlignedGrid::with_alloc_tracker(W, H, None).expect("grid");
        for (i, v) in grid.buf_mut().iter_mut().enumerate() {
            *v = (i as f32 + c as f32 * 0.01) / len as f32;
        }
        cpu_image.append_channel_shifted(
            ImageBuffer::F32(grid),
            Region::with_size(W as u32, H as u32),
            jxl_modular::ChannelShift::from_shift(0),
        );
    }
    let mut spot = AlignedGrid::with_alloc_tracker(W, H, None).expect("spot");
    for (i, v) in spot.buf_mut().iter_mut().enumerate() {
        *v = (i as f32) / len as f32 * 0.5;
    }
    cpu_image.append_channel_shifted(
        ImageBuffer::F32(spot),
        Region::with_size(W as u32, H as u32),
        jxl_modular::ChannelShift::from_shift(0),
    );

    let gpu = GpuImageWithRegion::from_cpu(&cpu_image).expect("upload");

    let mut cpu_ref = cpu_image;
    let ec = jxl_image::ExtraChannelType::SpotColour {
        red: 0.9,
        green: 0.1,
        blue: 0.2,
        solidity: 0.75,
    };
    let (c0, rest) = cpu_ref.buffer_mut().split_at_mut(1);
    let (c1, c2) = rest.split_at_mut(1);
    let (c2, spot_buf) = c2.split_at_mut(1);
    render_spot_color(
        [
            c0[0].as_float_mut().expect("f32"),
            c1[0].as_float_mut().expect("f32"),
            c2[0].as_float_mut().expect("f32"),
        ],
        spot_buf[0].as_float().expect("f32"),
        &ec,
    )
    .expect("cpu spot render");

    let bit_depth = BitDepth::FloatSample {
        bits_per_sample: 32,
        exp_bits: 8,
    };
    let (gpu_out, fused) = kernels::fuse_spot_colors_on_gpu(
        Arc::new(DeviceImage::Gpu(gpu)),
        bit_depth,
        &[(ec, bit_depth)],
    )
    .expect("gpu fuse");
    assert!(fused);

    let gpu_cpu = match gpu_out.as_ref() {
        DeviceImage::Gpu(g) => g.to_cpu().expect("download"),
        DeviceImage::Cpu(_) => panic!("expected gpu"),
    };

    for ch in 0..3 {
        let gpu_buf = gpu_cpu.buffer()[ch].as_float().expect("f32").buf();
        let cpu_buf = cpu_ref.buffer()[ch].as_float().expect("f32").buf();
        let mae = mean_abs_error(gpu_buf, cpu_buf);
        assert!(mae < 1e-4, "spot fusion channel {ch} MAE {mae}");
    }
}

#[cfg(feature = "gpu")]
fn fixture_image_header(name: &str) -> std::sync::Arc<jxl_image::ImageHeader> {
    use jxl_threadpool::JxlThreadPool;

    let bytes = fs::read(assets_dir().join(name)).expect("read jxl fixture");
    let codestream =
        crate::pipeline::parse::container::read_codestream(&bytes).expect("read codestream");
    crate::pipeline::parse::container::read_header(&codestream, JxlThreadPool::none())
        .expect("read header")
        .image_header
}

#[cfg(feature = "gpu")]
#[test]
fn nonseparable_upsample_gpu_matches_cpu() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping nonseparable_upsample_gpu_matches_cpu: no GPU adapter");
        return;
    }

    use jxl_grid::AlignedGrid;

    use crate::pipeline::gpu::{GpuImageWithRegion, upsample::gpu_upsample_channel_for_test};
    use crate::vendor::jxl_render::{ImageBuffer, ImageWithRegion, Region, features::upsample};

    const W: usize = 16;
    const H: usize = 16;
    let image_header = fixture_image_header("colors_e1_d0p5_fd4.jxl");

    let mut pattern = AlignedGrid::with_alloc_tracker(W, H, None).expect("grid");
    for (i, v) in pattern.buf_mut().iter_mut().enumerate() {
        *v = ((i % W) as f32 + 1.0) / W as f32 * ((i / W) as f32 + 1.0) / H as f32;
    }

    let region = Region::with_size(W as u32, H as u32);
    let mut cpu_region = region;
    let cpu_out = upsample(
        pattern.as_subgrid(),
        &mut cpu_region,
        &image_header,
        2,
        None,
    )
    .expect("cpu upsample")
    .expect("upsample output");

    let cpu_image = ImageWithRegion::new(1, None);
    let mut cpu_image = cpu_image;
    cpu_image.append_channel_shifted(
        ImageBuffer::F32(pattern),
        region,
        jxl_modular::ChannelShift::from_shift(0),
    );
    let gpu_input = GpuImageWithRegion::from_cpu(&cpu_image)
        .expect("upload")
        .buffer()[0]
        .try_clone()
        .expect("clone channel");
    let gpu_out = gpu_upsample_channel_for_test(gpu_input, &image_header, 2).expect("gpu upsample");

    let mut gpu_image = GpuImageWithRegion::new(1, None);
    gpu_image.append_channel_shifted(
        gpu_out,
        cpu_region,
        jxl_modular::ChannelShift::from_shift(0),
    );
    let gpu_downloaded = gpu_image.to_cpu().expect("download");
    let gpu_buf = gpu_downloaded.buffer()[0].as_float().expect("f32").buf();
    let cpu_buf = cpu_out.buf();
    assert_eq!(gpu_buf.len(), cpu_buf.len());
    let mae = mean_abs_error(gpu_buf, cpu_buf);
    assert!(mae < 1e-4, "nonseparable upsample MAE {mae}");
}

#[cfg(feature = "gpu")]
#[test]
fn blend_crop_gpu_matches_cpu() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping blend_crop_gpu_matches_cpu: no GPU adapter");
        return;
    }

    use jxl_grid::AlignedGrid;
    use jxl_modular::ChannelShift;

    use crate::pipeline::gpu::GpuImageWithRegion;
    use crate::pipeline::gpu::crop::dispatch_crop_f32;
    use crate::vendor::jxl_render::{ImageBuffer, ImageWithRegion, Region};

    const W: usize = 32;
    const H: usize = 32;
    let full = Region::with_size(W as u32, H as u32);
    let crop = Region {
        left: 4,
        top: 6,
        width: 20,
        height: 18,
    };

    let mut cpu_image = ImageWithRegion::new(1, None);
    let mut grid = AlignedGrid::with_alloc_tracker(W, H, None).expect("grid");
    for (i, v) in grid.buf_mut().iter_mut().enumerate() {
        *v = i as f32 / (W * H) as f32;
    }
    cpu_image.append_channel_shifted(ImageBuffer::F32(grid), full, ChannelShift::from_shift(0));

    let left = (crop.left - full.left) as u32;
    let top = (crop.top - full.top) as u32;
    let cpu_sub = cpu_image.buffer()[0]
        .as_float()
        .expect("f32")
        .as_subgrid()
        .subgrid(
            left as usize..(left + crop.width) as usize,
            top as usize..(top + crop.height) as usize,
        );
    let cpu_pixels: Vec<f32> = (0..crop.height as usize)
        .flat_map(|y| cpu_sub.get_row(y).iter().copied())
        .collect();

    let gpu = GpuImageWithRegion::from_cpu(&cpu_image).expect("upload");
    let cropped =
        dispatch_crop_f32(&gpu.buffer()[0], left, top, crop.width, crop.height).expect("gpu crop");
    let mut cropped_image = GpuImageWithRegion::new(1, None);
    cropped_image.append_channel_shifted(cropped, crop, ChannelShift::from_shift(0));
    let cropped_cpu = cropped_image.to_cpu().expect("download crop");
    let gpu_pixels = cropped_cpu.buffer()[0].as_float().expect("f32").buf();
    let mae = mean_abs_error(gpu_pixels, &cpu_pixels);
    assert!(mae < 1e-6, "crop MAE {mae}");
}

#[cfg(feature = "gpu")]
fn fixture_frame_header(
    name: &str,
) -> (
    std::sync::Arc<jxl_image::ImageHeader>,
    crate::vendor::jxl_frame::FrameHeader,
) {
    use jxl_bitstream::Bitstream;
    use jxl_oxide_common::Bundle;
    use jxl_threadpool::JxlThreadPool;

    use crate::vendor::jxl_frame::FrameHeader;

    let bytes = fs::read(assets_dir().join(name)).expect("read jxl fixture");
    let codestream =
        crate::pipeline::parse::container::read_codestream(&bytes).expect("read codestream");
    let decl = crate::pipeline::parse::container::read_header(&codestream, JxlThreadPool::none())
        .expect("read header");
    let mut bitstream = Bitstream::new(&codestream[decl.offset..]);
    let frame_header =
        FrameHeader::parse(&mut bitstream, &decl.image_header).expect("parse frame header");
    (decl.image_header, frame_header)
}

#[cfg(feature = "gpu")]
#[test]
fn features_noise_gpu_matches_cpu() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping features_noise_gpu_matches_cpu: no GPU adapter");
        return;
    }

    use jxl_grid::AlignedGrid;
    use jxl_modular::ChannelShift;
    use jxl_threadpool::JxlThreadPool;

    use crate::pipeline::gpu::{GpuImageWithRegion, noise::dispatch_noise_for_test};
    use crate::vendor::jxl_frame::data::NoiseParameters;
    use crate::vendor::jxl_render::{ImageBuffer, ImageWithRegion, Region, features::render_noise};

    const W: u32 = 64;
    const H: u32 = 64;
    let (_image_header, mut frame_header) = fixture_frame_header("colors_e1_d0p5_fd4.jxl");
    frame_header.width = W;
    frame_header.height = H;

    let region = Region::with_size(W, H);
    let make_image = || -> ImageWithRegion {
        let mut image = ImageWithRegion::new(3, None);
        for ch in 0..3 {
            let mut grid =
                AlignedGrid::with_alloc_tracker(W as usize, H as usize, None).expect("grid");
            for (i, v) in grid.buf_mut().iter_mut().enumerate() {
                *v = ((i % W as usize) as f32 + 1.0) / W as f32 * ((i / W as usize) as f32 + 1.0)
                    / H as f32
                    + ch as f32 * 0.01;
            }
            image.append_channel_shifted(
                ImageBuffer::F32(grid),
                region,
                ChannelShift::from_shift(0),
            );
        }
        image
    };

    let noise_params = NoiseParameters {
        lut: std::array::from_fn(|i| (i as f32 + 1.0) / 9.0),
    };
    let base_corr = Some((0.1, 0.9));

    let mut cpu_image = make_image();
    render_noise(
        &frame_header,
        2,
        1,
        base_corr,
        &mut cpu_image,
        &noise_params,
        &JxlThreadPool::none(),
    )
    .expect("cpu noise");

    let mut gpu_image = make_image();
    let mut gpu = GpuImageWithRegion::from_cpu(&gpu_image).expect("upload");
    gpu.convert_modular_color(_image_header.metadata.bit_depth)
        .expect("convert");
    dispatch_noise_for_test(&mut gpu, &frame_header, &noise_params, 2, 1, base_corr)
        .expect("gpu noise");
    gpu_image = gpu.to_cpu().expect("download");

    for ch in 0..3 {
        let cpu_buf = cpu_image.buffer()[ch].as_float().expect("f32").buf();
        let gpu_buf = gpu_image.buffer()[ch].as_float().expect("f32").buf();
        let mae = mean_abs_error(gpu_buf, cpu_buf);
        assert!(mae < 0.05, "noise channel {ch} MAE {mae}");
    }
}

#[cfg(feature = "gpu")]
#[test]
fn decode_gpu_up2_fixture_matches_cpu_pixels() {
    if !GpuEnvironment::current().device_available {
        eprintln!("skipping decode_gpu_up2_fixture_matches_cpu_pixels: no GPU adapter");
        return;
    }

    let jxl_path = assets_dir().join("colors_up2_e1_d0p5_fd4.jxl");
    if !jxl_path.exists() {
        eprintln!("skipping decode_gpu_up2_fixture_matches_cpu_pixels: fixture missing");
        return;
    }

    let jxl_bytes = fs::read(&jxl_path).expect("read jxl fixture");
    let reference = decode(&jxl_bytes).expect("reference decode");
    let gpu = decode_with_options(
        &jxl_bytes,
        &DecodeOptions {
            hardware: Hardware::Gpu,
            destination: Destination::Cpu,
            ..DecodeOptions::default()
        },
    )
    .expect("GPU decode");

    let mae = mean_abs_error(
        gpu.pixels.as_cpu().expect("cpu pixels"),
        reference.pixels.as_cpu().expect("reference pixels"),
    );
    assert!(mae < 0.02, "GPU up2 fixture MAE {mae} exceeds tolerance");
}
