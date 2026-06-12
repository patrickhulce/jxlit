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
