//! jxl-oxide candidate: decode a JPEG XL file with the upstream `jxl-oxide`
//! crate as shipped (multithreaded via rayon). This is distinct from the jxlit
//! pipeline, which is built on vendored jxl-oxide sub-crates.
//!
//! Emits a single JSON line matching the jxlit benchmark schema.

use std::time::Instant;

use jxl_oxide::{JxlImage, JxlThreadPool};

const WARMUP_DECODES: usize = 3;

struct Args {
    file: String,
    iterations: usize,
    threads: usize,
    action: String,
}

fn parse_args() -> Args {
    let mut file = None;
    let mut iterations = 100usize;
    let mut threads = 0usize; // 0 == auto
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
            "--threads" => threads = value().parse().expect("threads"),
            "--action" => action = value(),
            // Accepted for a uniform CLI across candidates, but unused here.
            "--layout" | "--hardware" | "--destination" => {
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
        threads,
        action,
    }
}

/// Decodes the first keyframe and materializes all channel samples.
fn decode_once(bytes: &[u8], pool: &JxlThreadPool) -> (u32, u32, u32) {
    let image = JxlImage::builder()
        .pool(pool.clone())
        .read(bytes)
        .expect("read jxl image");
    let render = image.render_frame(0).expect("render frame");
    let fb = render.image_all_channels();
    let width = fb.width() as u32;
    let height = fb.height() as u32;
    let channels = fb.channels() as u32;
    std::hint::black_box(fb.buf());
    (width, height, channels)
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

    let num_threads = if args.threads > 0 {
        Some(args.threads)
    } else {
        None // jxl-oxide picks a default rayon pool sized to the machine
    };
    let pool = JxlThreadPool::rayon(num_threads);

    let (mut width, mut height, mut channels) = (0u32, 0u32, 0u32);
    for _ in 0..WARMUP_DECODES {
        let (w, h, c) = decode_once(&bytes, &pool);
        width = w;
        height = h;
        channels = c;
    }

    let mut latencies_ms = Vec::with_capacity(args.iterations);
    let decode_start = Instant::now();
    for _ in 0..args.iterations {
        let start = Instant::now();
        let decoded = decode_once(&bytes, &pool);
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(decoded);
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    let mut sorted = latencies_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
    let megapixels = (width as f64 * height as f64) / 1_000_000.0;

    println!(
        "{{\"decoder\":\"jxl-oxide\",\"action\":\"{}\",\"hardware\":\"cpu\",\
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
