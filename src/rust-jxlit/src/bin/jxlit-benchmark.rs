use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use jxlit::{DecodeOptions, decode_with_options};

const WARMUP_DECODES: usize = 3;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_FILE: &str = "assets/frame_4K_10bit_e1_d0p5_fd4.jxl";

struct Options {
    file: String,
    action: String,
    iterations: usize,
    threads: Option<usize>,
}

fn parse_args() -> Options {
    let mut file: Option<String> = None;
    let mut action: Option<String> = None;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut threads: Option<usize> = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--file" => {
                file = Some(args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --file");
                    process::exit(1);
                }));
            }
            "--action" => {
                action = Some(args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --action");
                    process::exit(1);
                }));
            }
            "--iterations" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --iterations");
                    process::exit(1);
                });
                iterations = value.parse().unwrap_or_else(|_| {
                    eprintln!("invalid --iterations value: {value}");
                    process::exit(1);
                });
            }
            "--threads" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --threads");
                    process::exit(1);
                });
                threads = Some(value.parse().unwrap_or_else(|_| {
                    eprintln!("invalid --threads value: {value}");
                    process::exit(1);
                }));
            }
            other => {
                eprintln!("unknown argument: {other}");
                process::exit(1);
            }
        }
    }

    let file = file.unwrap_or_else(|| DEFAULT_FILE.to_string());
    let action = action.unwrap_or_else(|| "decode_cpu".to_string());

    if action != "decode_cpu" {
        eprintln!("unsupported action: {action}");
        process::exit(1);
    }

    if iterations == 0 {
        eprintln!("--iterations must be greater than 0");
        process::exit(1);
    }

    Options {
        file,
        action,
        iterations,
        threads,
    }
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
    let upper = rank.ceil() as usize;
    let weight = rank - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn resolve_file(path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_file() {
        return candidate;
    }

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let from_manifest = PathBuf::from(&manifest_dir)
            .join("..")
            .join("..")
            .join(path);
        if from_manifest.is_file() {
            return from_manifest;
        }
    }

    candidate
}

fn main() {
    let options = parse_args();
    let file_path = resolve_file(&options.file);

    let bytes = fs::read(&file_path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", file_path.display());
        process::exit(1);
    });

    let decode_options = DecodeOptions {
        threads: options.threads,
    };

    let warmup = decode_with_options(&bytes, &decode_options).unwrap_or_else(|e| {
        eprintln!("warmup decode failed: {e}");
        process::exit(1);
    });
    for _ in 1..WARMUP_DECODES {
        decode_with_options(&bytes, &decode_options).unwrap_or_else(|e| {
            eprintln!("warmup decode failed: {e}");
            process::exit(1);
        });
    }
    let width = warmup.width;
    let height = warmup.height;
    let channels = warmup.channels;
    let megapixels = (width * height) as f64 / 1_000_000.0;

    let mut latencies_ms = Vec::with_capacity(options.iterations);
    let decode_start = Instant::now();

    for _ in 0..options.iterations {
        let start = Instant::now();
        let decoded = decode_with_options(&bytes, &decode_options).unwrap_or_else(|e| {
            eprintln!("decode failed: {e}");
            process::exit(1);
        });
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(elapsed_ms);
        std::hint::black_box(decoded);
    }

    let decode_seconds = decode_start.elapsed().as_secs_f64();

    let mut sorted = latencies_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    println!(
        "{{\"lang\":\"rust\",\"action\":\"{}\",\"iterations\":{},\"width\":{},\"height\":{},\"channels\":{},\"megapixels\":{:.6},\"decode_seconds\":{:.6},\"latency_ms\":{{\"mean\":{:.3},\"p50\":{:.3},\"p95\":{:.3},\"min\":{:.3},\"max\":{:.3}}}}}",
        options.action,
        options.iterations,
        width,
        height,
        channels,
        megapixels,
        decode_seconds,
        mean,
        p50,
        p95,
        min,
        max,
    );
}
