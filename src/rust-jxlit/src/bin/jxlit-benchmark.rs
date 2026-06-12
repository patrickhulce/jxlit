use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use jxlit::{
    DecodeOptions, Destination, Hardware, PixelLayout, RebasingTelemetry, decode_with_options,
    rebase_telemetry,
};

const WARMUP_DECODES: usize = 3;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_FILE: &str = "assets/frame_4K_10bit_e1_d0p5_fd4.jxl";

struct Options {
    file: String,
    action: String,
    iterations: usize,
    threads: Option<usize>,
    layout: PixelLayout,
    hardware: Hardware,
    destination: Destination,
    no_telemetry: bool,
}

fn parse_hardware(value: &str) -> Hardware {
    match value {
        "cpu" => Hardware::Cpu,
        "gpu" => Hardware::Gpu,
        _ => {
            eprintln!("invalid --hardware value: {value} (expected cpu or gpu)");
            process::exit(1);
        }
    }
}

fn parse_destination(value: &str) -> Destination {
    match value {
        "cpu" => Destination::Cpu,
        "gpu" => Destination::Gpu,
        _ => {
            eprintln!("invalid --destination value: {value} (expected cpu or gpu)");
            process::exit(1);
        }
    }
}

fn hardware_label(hardware: Hardware) -> &'static str {
    match hardware {
        Hardware::Cpu => "cpu",
        Hardware::Gpu => "gpu",
    }
}

fn destination_label(destination: Destination) -> &'static str {
    match destination {
        Destination::Cpu => "cpu",
        Destination::Gpu => "gpu",
    }
}

fn unix_time_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

fn parse_args() -> Options {
    let mut file: Option<String> = None;
    let mut action: Option<String> = None;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut threads: Option<usize> = None;
    let mut layout = PixelLayout::Interleaved;
    let mut hardware = Hardware::Cpu;
    let mut destination = Destination::Cpu;
    let mut no_telemetry = false;
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
            "--layout" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --layout");
                    process::exit(1);
                });
                layout = match value.as_str() {
                    "interleaved" => PixelLayout::Interleaved,
                    "planar" => PixelLayout::Planar,
                    _ => {
                        eprintln!(
                            "invalid --layout value: {value} (expected interleaved or planar)"
                        );
                        process::exit(1);
                    }
                };
            }
            "--hardware" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --hardware");
                    process::exit(1);
                });
                hardware = parse_hardware(&value);
            }
            "--destination" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --destination");
                    process::exit(1);
                });
                destination = parse_destination(&value);
            }
            "--no-telemetry" => no_telemetry = true,
            other => {
                eprintln!("unknown argument: {other}");
                process::exit(1);
            }
        }
    }

    let file = file.unwrap_or_else(|| DEFAULT_FILE.to_string());
    let action = action.unwrap_or_else(|| "decode_cpu".to_string());

    if action != "decode_cpu" && action != "decode_gpu" {
        eprintln!("unsupported action: {action} (expected decode_cpu or decode_gpu)");
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
        layout,
        hardware,
        destination,
        no_telemetry,
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

fn print_phase_summary(telemetry: &RebasingTelemetry, lang: &str, top_n: usize) {
    let outer = telemetry
        .measures
        .iter()
        .find(|measure| measure.name.ends_with("_decode"));
    let total_ms = outer.map(|m| m.duration_ms).unwrap_or(telemetry.total_ms);
    let mut ranked: Vec<_> = telemetry.measures.iter().collect();
    ranked.sort_by(|a, b| {
        b.duration_ms
            .partial_cmp(&a.duration_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_n);

    eprintln!("\n== phase breakdown ({lang}) ==");
    for measure in ranked {
        let pct = if total_ms > 0.0 {
            100.0 * measure.duration_ms / total_ms
        } else {
            0.0
        };
        eprintln!(
            "{:<16} {:>8.2}ms {:>6.1}%",
            measure.name, measure.duration_ms, pct
        );
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn telemetry_to_json(telemetry: &RebasingTelemetry) -> String {
    let measures: Vec<String> = telemetry
        .measures
        .iter()
        .map(|measure| {
            format!(
                "{{\"name\":\"{}\",\"start_ms\":{:.6},\"duration_ms\":{:.6}}}",
                json_escape(&measure.name),
                measure.start_ms,
                measure.duration_ms
            )
        })
        .collect();
    format!(
        "{{\"timebase\":{:.6},\"total_ms\":{:.6},\"measures\":[{}]}}",
        telemetry.timebase,
        telemetry.total_ms,
        measures.join(",")
    )
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
        layout: options.layout,
        hardware: options.hardware,
        destination: options.destination,
        ..DecodeOptions::default()
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

    let telemetry_json = if options.no_telemetry {
        String::new()
    } else {
        let telemetry_options = DecodeOptions {
            threads: options.threads,
            layout: options.layout,
            hardware: options.hardware,
            destination: options.destination,
            telemetry: true,
        };
        let timebase = unix_time_ms();
        let wall_start = Instant::now();
        let telemetry_decode =
            decode_with_options(&bytes, &telemetry_options).unwrap_or_else(|e| {
                eprintln!("telemetry decode failed: {e}");
                process::exit(1);
            });
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        let native = telemetry_decode
            .metadata
            .jxlit
            .telemetry
            .as_ref()
            .expect("telemetry decode must return telemetry");
        std::hint::black_box(&telemetry_decode);
        let rebased = rebase_telemetry(native, timebase, "rust_decode", wall_ms);
        print_phase_summary(&rebased, "rust", 25);
        format!(",\"telemetry\":{}", telemetry_to_json(&rebased))
    };

    println!(
        "{{\"lang\":\"rust\",\"action\":\"{}\",\"hardware\":\"{}\",\"destination\":\"{}\",\"iterations\":{},\"width\":{},\"height\":{},\"channels\":{},\"megapixels\":{:.6},\"decode_seconds\":{:.6},\"latency_ms\":{{\"mean\":{:.3},\"p50\":{:.3},\"p95\":{:.3},\"min\":{:.3},\"max\":{:.3}}}{}}}",
        options.action,
        hardware_label(options.hardware),
        destination_label(options.destination),
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
        telemetry_json,
    );
}
