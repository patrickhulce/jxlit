import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";

import { decode } from "./index.js";

const WARMUP_DECODES = 3;
const DEFAULT_ITERATIONS = 100;

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
}

interface Options {
  file: string;
  action: string;
  iterations: number;
}

function parseArgs(argv: string[]): Options {
  let file: string | undefined;
  let action = "decode_cpu";
  let iterations = DEFAULT_ITERATIONS;

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
      default:
        console.error(`unknown argument: ${arg}`);
        process.exit(1);
    }
  }

  if (!file) {
    console.error("--file is required");
    process.exit(1);
  }

  if (action !== "decode_cpu") {
    console.error(`unsupported action: ${action}`);
    process.exit(1);
  }

  if (!Number.isFinite(iterations) || iterations <= 0) {
    console.error("--iterations must be greater than 0");
    process.exit(1);
  }

  return { file, action, iterations };
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

  const warmup = decode(bytes);
  for (let i = 1; i < WARMUP_DECODES; i++) {
    decode(bytes);
  }
  const width = warmup.width;
  const height = warmup.height;
  const channels = warmup.channels;
  const megapixels = (width * height) / 1_000_000;

  const latenciesMs: number[] = [];
  const decodeStart = performance.now();

  for (let i = 0; i < options.iterations; i++) {
    const start = performance.now();
    const decoded = decode(bytes);
    const elapsedMs = performance.now() - start;
    latenciesMs.push(elapsedMs);
    void decoded;
  }

  const decodeSeconds = (performance.now() - decodeStart) / 1000;

  const sorted = [...latenciesMs].sort((a, b) => a - b);
  const result: BenchmarkResult = {
    lang: "node",
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

  console.log(JSON.stringify(result));
}

main();
