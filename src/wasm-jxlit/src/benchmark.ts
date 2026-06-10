import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";

import { decode } from "./index.js";
import { printPhaseSummary, telemetryToJson } from "./telemetry.js";

const WARMUP_DECODES = 3;
const DEFAULT_ITERATIONS = 100;
const DEFAULT_FILE = join(
  process.cwd(),
  "..",
  "..",
  "assets",
  "frame_4K_10bit_e1_d0p5_fd4.jxl",
);

interface LatencyStats {
  mean: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
}

interface BenchmarkResult {
  lang: string;
  action: string;
  iterations: number;
  width: number;
  height: number;
  channels: number;
  megapixels: number;
  decode_seconds: number;
  latency_ms: LatencyStats;
  telemetry?: Record<string, unknown>;
}

interface Options {
  file: string;
  action: string;
  iterations: number;
  threads?: number;
  noTelemetry: boolean;
}

function parseArgs(argv: string[]): Options {
  let file: string | undefined;
  let action = "decode_cpu";
  let iterations = DEFAULT_ITERATIONS;
  let threads: number | undefined;
  let noTelemetry = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--file":
        file = argv[++i];
        break;
      case "--action":
        action = argv[++i] ?? action;
        break;
      case "--iterations": {
        const value = argv[++i];
        iterations = Number.parseInt(value ?? "", 10);
        break;
      }
      case "--threads": {
        const value = argv[++i];
        threads = Number.parseInt(value ?? "", 10);
        break;
      }
      case "--no-telemetry":
        noTelemetry = true;
        break;
      default:
        console.error(`unknown argument: ${arg}`);
        process.exit(1);
    }
  }

  if (!file) {
    file = resolve(DEFAULT_FILE);
  }

  if (action !== "decode_cpu") {
    console.error(`unsupported action: ${action}`);
    process.exit(1);
  }

  if (!Number.isFinite(iterations) || iterations <= 0) {
    console.error("--iterations must be greater than 0");
    process.exit(1);
  }

  return { file, action, iterations, threads, noTelemetry };
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  if (sorted.length === 1) {
    return sorted[0];
  }
  const rank = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  const weight = rank - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const bytes = readFileSync(options.file);
  const decodeOptions =
    options.threads === undefined ? undefined : { threads: options.threads };

  const warmup = decode(bytes, decodeOptions);
  for (let i = 1; i < WARMUP_DECODES; i++) {
    decode(bytes, decodeOptions);
  }
  const width = warmup.width;
  const height = warmup.height;
  const channels = warmup.channels;
  const megapixels = (width * height) / 1_000_000;

  const latenciesMs: number[] = [];
  const decodeStart = performance.now();

  for (let i = 0; i < options.iterations; i++) {
    const start = performance.now();
    const decoded = decode(bytes, decodeOptions);
    const elapsedMs = performance.now() - start;
    latenciesMs.push(elapsedMs);
    void decoded;
  }

  const decodeSeconds = (performance.now() - decodeStart) / 1000;

  const sorted = [...latenciesMs].sort((a, b) => a - b);
  const result: BenchmarkResult = {
    lang: "wasm",
    action: options.action,
    iterations: options.iterations,
    width,
    height,
    channels,
    megapixels,
    decode_seconds: decodeSeconds,
    latency_ms: {
      mean:
        latenciesMs.reduce((sum, value) => sum + value, 0) / latenciesMs.length,
      p50: percentile(sorted, 50),
      p95: percentile(sorted, 95),
      min: sorted[0],
      max: sorted[sorted.length - 1],
    },
  };

  if (!options.noTelemetry) {
    const telemetryOptions = {
      ...(decodeOptions ?? {}),
      telemetry: true,
    };
    const telemetryDecode = decode(bytes, telemetryOptions);
    const telemetry = telemetryDecode.metadata._jxlit.telemetry;
    if (telemetry !== undefined) {
      printPhaseSummary(telemetry, "wasm");
      result.telemetry = telemetryToJson(telemetry);
    }
  }

  console.log(JSON.stringify(result));
}

main();
