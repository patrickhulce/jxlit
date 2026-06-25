//! jxl-rs candidate: decode a JPEG XL file with the pure-Rust `jxl` crate
//! (libjxl/jxl-rs). The crate decodes single-threaded, so `--threads` is
//! accepted for a uniform CLI but does not change behavior.
//!
//! Emits a single JSON line matching the jxlit benchmark schema.

use std::time::Instant;

use jxl::api::{
    states, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult,
};

const WARMUP_DECODES: usize = 3;

struct Args {
    file: String,
    iterations: usize,
    action: String,
}

fn parse_args() -> Args {
    let mut file = None;
    let mut iterations = 100usize;
    let mut action = "decode_cpu".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next().unwrap_or_else(|| {
                eprintln!("missing value for {arg}");
                std::process::exit(1);
            })
        };
        match arg.as_str() {
            "--file" => file = Some(value()),
            "--iterations" => iterations = value().parse().expect("iterations"),
            "--action" => action = value(),
            // Accepted for a uniform CLI across candidates, but unused here.
            "--threads" | "--layout" | "--hardware" | "--destination" => {
                let _ = value();
            }
            "--no-telemetry" => {}
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }
    let file = file.unwrap_or_else(|| {
        eprintln!("--file is required");
        std::process::exit(1);
    });
    if iterations == 0 {
        eprintln!("--iterations must be greater than 0");
        std::process::exit(1);
    }
    Args {
        file,
        iterations,
        action,
    }
}

/// Decodes the first frame to f32 pixels and materializes the color buffer.
fn decode_once(bytes: &[u8]) -> (u32, u32, u32) {
    let mut input: &[u8] = bytes;
    let init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());

    let mut with_info = match init.process(&mut input).expect("process image info") {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("unexpected end of input (image info)"),
    };

    let (width, height) = with_info.basic_info().size;
    let width = width as usize;
    let height = height as usize;

    // Request f32 output for color and any extra channels.
    let default_format = with_info.current_pixel_format().clone();
    let requested_format = JxlPixelFormat {
        color_type: default_format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: default_format
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    };
    with_info.set_pixel_format(requested_format);
    let pixel_format = with_info.current_pixel_format().clone();
    let num_channels = pixel_format.color_type.samples_per_pixel();

    let color_bytes_per_row = width * num_channels * 4;
    let mut color_buf = vec![0u8; color_bytes_per_row * height];
    let mut extra_bufs: Vec<Vec<u8>> = pixel_format
        .extra_channel_format
        .iter()
        .filter(|ec| ec.is_some())
        .map(|_| vec![0u8; width * 4 * height])
        .collect();

    let with_frame = match with_info.process(&mut input).expect("process frame info") {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("unexpected end of input (frame info)"),
    };

    {
        let mut buffers: Vec<JxlOutputBuffer> =
            vec![JxlOutputBuffer::new(&mut color_buf, height, color_bytes_per_row)];
        for buf in extra_bufs.iter_mut() {
            buffers.push(JxlOutputBuffer::new(buf, height, width * 4));
        }
        match with_frame
            .process(&mut input, &mut buffers)
            .expect("process pixels")
        {
            ProcessingResult::Complete { .. } => {}
            ProcessingResult::NeedsMoreInput { .. } => panic!("unexpected end of input (pixels)"),
        }
    }

    std::hint::black_box(&color_buf);
    (width as u32, height as u32, num_channels as u32)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let weight = rank - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn main() {
    let args = parse_args();
    let bytes = std::fs::read(&args.file).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", args.file);
        std::process::exit(1);
    });

    let (mut width, mut height, mut channels) = (0u32, 0u32, 0u32);
    for _ in 0..WARMUP_DECODES {
        let (w, h, c) = decode_once(&bytes);
        width = w;
        height = h;
        channels = c;
    }

    let mut latencies_ms = Vec::with_capacity(args.iterations);
    let decode_start = Instant::now();
    for _ in 0..args.iterations {
        let start = Instant::now();
        let decoded = decode_once(&bytes);
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(decoded);
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    let mut sorted = latencies_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
    let megapixels = (width as f64 * height as f64) / 1_000_000.0;

    println!(
        "{{\"decoder\":\"jxl-rs\",\"action\":\"{}\",\"hardware\":\"cpu\",\
\"destination\":\"cpu\",\"iterations\":{},\"width\":{},\"height\":{},\
\"channels\":{},\"megapixels\":{:.6},\"decode_seconds\":{:.6},\
\"latency_ms\":{{\"mean\":{:.3},\"p50\":{:.3},\"p95\":{:.3},\"min\":{:.3},\
\"max\":{:.3}}}}}",
        args.action,
        args.iterations,
        width,
        height,
        channels,
        megapixels,
        decode_seconds,
        mean,
        percentile(&sorted, 50.0),
        percentile(&sorted, 95.0),
        sorted[0],
        sorted[sorted.len() - 1],
    );
}
