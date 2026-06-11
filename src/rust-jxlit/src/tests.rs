use std::fs;
use std::path::PathBuf;

use crate::{DecodeOptions, Hardware, PixelLayout, decode, decode_with_options};
#[cfg(feature = "gpu")]
use crate::{Destination, pipeline::gpu::{download_pixels, GpuEnvironment}};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("assets")
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
        mae < 1e-5,
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
    assert_eq!(
        gpu.len(),
        decoded.width * decoded.height * decoded.channels
    );

    let downloaded = download_pixels(&gpu).expect("download GPU pixels");
    let reference_decoded = decode(&jxl_bytes).expect("reference decode");
    let reference = reference_decoded
        .pixels
        .as_cpu()
        .expect("reference cpu pixels");
    let mae = mean_abs_error(&downloaded, reference);
    assert!(
        mae < 1e-5,
        "GPU destination download MAE {mae} exceeds tolerance"
    );
}
