"""imagecodecs candidate: decode JPEG XL via libjxl and (optionally) upload to GPU.

imagecodecs wraps the libjxl reference decoder, so decode always runs on CPU.
Two destinations are supported:

* ``cpu``: decode into a NumPy array.
* ``gpu``: decode, convert to float32, and upload to a PyTorch device
  (CUDA on Linux, MPS on macOS), synchronizing so the transfer is included in
  the measured time. If no GPU device is available, the worker prints a single
  ``{"skipped": true, ...}`` JSON line and exits 0.

Emits a single JSON line matching the jxlit benchmark schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import imagecodecs
import numpy as np

WARMUP_DECODES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="imagecodecs decode benchmark")
    parser.add_argument("--file", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--action", default="decode_cpu")
    parser.add_argument("--destination", choices=("cpu", "gpu"), default="cpu")
    # Accepted for a uniform CLI across candidates, but unused here.
    parser.add_argument("--hardware", default="cpu")
    parser.add_argument("--layout", default="interleaved")
    parser.add_argument("--no-telemetry", action="store_true")
    return parser.parse_args()


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


def emit_skip(reason: str) -> None:
    print(json.dumps({"decoder": "imagecodecs", "skipped": True, "reason": reason}))


def pick_torch_device():
    """Return (torch_module, device) or (None, None) if no GPU is available."""
    try:
        import torch
    except ImportError:
        return None, None
    if torch.cuda.is_available():
        return torch, torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch, torch.device("mps")
    return None, None


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    with open(args.file, "rb") as handle:
        data = handle.read()

    numthreads = args.threads if args.threads and args.threads > 0 else None

    torch = None
    device = None
    if args.destination == "gpu":
        torch, device = pick_torch_device()
        if device is None:
            emit_skip("no CUDA or MPS device available for GPU destination")
            return

    # Warmup also fixes the output array shape/dtype so the timed loop can decode
    # into a preallocated buffer.
    cpu_out = imagecodecs.jpegxl_decode(data, numthreads=numthreads)
    for _ in range(WARMUP_DECODES - 1):
        imagecodecs.jpegxl_decode(data, numthreads=numthreads, out=cpu_out)
    if device is not None:
        host = np.ascontiguousarray(cpu_out, dtype=np.float32)
        warm = torch.from_numpy(host).to(device)
        synchronize(torch, device)
        del warm

    height = cpu_out.shape[0]
    width = cpu_out.shape[1]
    channels = cpu_out.shape[2] if cpu_out.ndim == 3 else 1
    megapixels = width * height / 1e6

    latencies_ms: list[float] = []
    decode_start = time.perf_counter()
    for _ in range(args.iterations):
        start = time.perf_counter()
        arr = imagecodecs.jpegxl_decode(data, numthreads=numthreads, out=cpu_out)
        if device is not None:
            host = np.ascontiguousarray(arr, dtype=np.float32)
            tensor = torch.from_numpy(host).to(device)
            synchronize(torch, device)
            del tensor
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    decode_seconds = time.perf_counter() - decode_start

    sorted_ms = sorted(latencies_ms)
    payload = {
        "decoder": "imagecodecs",
        "action": args.action,
        "hardware": "cpu",
        "destination": args.destination,
        "iterations": args.iterations,
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "megapixels": megapixels,
        "decode_seconds": decode_seconds,
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms),
            "p50": percentile(sorted_ms, 50.0),
            "p95": percentile(sorted_ms, 95.0),
            "min": sorted_ms[0],
            "max": sorted_ms[-1],
        },
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
