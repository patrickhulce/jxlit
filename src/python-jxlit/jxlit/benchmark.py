"""Benchmark CLI for the Python jxlit bindings."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from jxlit import decode

WARMUP_DECODES = 3
DEFAULT_ITERATIONS = 100


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (p / 100.0) * (len(values) - 1)
    lower = int(rank // 1)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark jxlit decode performance")
    parser.add_argument("--file", required=True, help="Path to a JPEG XL file")
    parser.add_argument(
        "--action",
        default="decode_cpu",
        help="Benchmark action (only decode_cpu is supported)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of measured decode iterations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action != "decode_cpu":
        print(f"unsupported action: {args.action}", file=sys.stderr)
        raise SystemExit(1)

    if args.iterations <= 0:
        print("--iterations must be greater than 0", file=sys.stderr)
        raise SystemExit(1)

    data = Path(args.file).read_bytes()

    warmup = decode(data)
    for _ in range(1, WARMUP_DECODES):
        decode(data)

    width = warmup.width
    height = warmup.height
    channels = warmup.channels
    megapixels = (width * height) / 1_000_000.0

    latencies_ms: list[float] = []
    decode_start = time.perf_counter()

    for _ in range(args.iterations):
        start = time.perf_counter()
        decoded = decode(data)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        _ = decoded

    decode_seconds = time.perf_counter() - decode_start

    sorted_latencies = sorted(latencies_ms)
    result = {
        "lang": "python",
        "action": args.action,
        "iterations": args.iterations,
        "width": width,
        "height": height,
        "channels": channels,
        "megapixels": megapixels,
        "decode_seconds": decode_seconds,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "p50": percentile(sorted_latencies, 50.0),
            "p95": percentile(sorted_latencies, 95.0),
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
