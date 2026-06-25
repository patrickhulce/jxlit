"""Shared helpers for jxlit decode benchmark orchestrators.

Both `benchmark.py` (cross-language jxlit bindings) and `benchmark_compare.py`
(cross-decoder comparison) parse the same single-line JSON worker output and
aggregate per-worker latencies into throughput summaries. The logic lives here
so the two orchestrators stay in lockstep.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkerResult:
    """One worker process's decode result, parsed from its JSON stdout line."""

    name: str
    action: str
    iterations: int
    width: int
    height: int
    channels: int
    megapixels: float
    decode_seconds: float
    latency_ms: dict[str, float]
    hardware: str = "cpu"
    destination: str = "cpu"
    telemetry: dict[str, Any] | None = None


@dataclass(frozen=True)
class Summary:
    """Aggregate of a batch of workers running the same config."""

    name: str
    workers: int
    iterations_per_worker: int
    total_iterations: int
    width: int
    height: int
    channels: int
    megapixels: float
    latency_ms: dict[str, float]
    per_worker_fps: float
    per_worker_mps: float
    aggregate_fps: float
    aggregate_mps: float
    total_worker_seconds: float
    batch_wall_seconds: float
    overhead: float
    hardware: str = "cpu"
    destination: str = "cpu"


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


def parse_worker_result(payload: dict[str, Any]) -> WorkerResult:
    """Build a `WorkerResult` from a worker's JSON payload.

    Accepts either a `decoder` key (compare candidates) or a `lang` key (the
    existing jxlit binding workers), so both harnesses share this parser.
    """
    name = payload.get("decoder") or payload.get("lang") or payload.get("name")
    telemetry = payload.get("telemetry")
    return WorkerResult(
        name=str(name),
        action=str(payload.get("action", "decode")),
        iterations=int(payload["iterations"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        channels=int(payload["channels"]),
        megapixels=float(payload["megapixels"]),
        decode_seconds=float(payload["decode_seconds"]),
        latency_ms={key: float(value) for key, value in payload["latency_ms"].items()},
        hardware=str(payload.get("hardware", "cpu")),
        destination=str(payload.get("destination", "cpu")),
        telemetry=telemetry if isinstance(telemetry, dict) else None,
    )


def aggregate_results(
    name: str,
    workers: int,
    results: list[WorkerResult],
    batch_wall_seconds: float,
    *,
    hardware: str = "cpu",
    destination: str = "cpu",
) -> Summary:
    if not results:
        raise ValueError(f"no worker results for {name}")

    iterations_per_worker = results[0].iterations
    total_iterations = iterations_per_worker * workers
    width = results[0].width
    height = results[0].height
    channels = results[0].channels
    megapixels = results[0].megapixels

    pooled_means = [result.latency_ms["mean"] for result in results]
    pooled_p50 = [result.latency_ms["p50"] for result in results]
    pooled_p95 = [result.latency_ms["p95"] for result in results]
    pooled_min = [result.latency_ms["min"] for result in results]
    pooled_max = [result.latency_ms["max"] for result in results]

    per_worker_fps = statistics.mean(
        result.iterations / result.decode_seconds for result in results
    )
    per_worker_mps = statistics.mean(
        result.megapixels * result.iterations / result.decode_seconds
        for result in results
    )

    total_worker_seconds = sum(result.decode_seconds for result in results)
    total_megapixels = megapixels * total_iterations
    aggregate_fps = total_iterations / batch_wall_seconds
    aggregate_mps = total_megapixels / batch_wall_seconds
    # Share of wall time not spent in measured decode work (0-100%).
    overhead = (1.0 - total_worker_seconds / (batch_wall_seconds * workers)) * 100.0

    return Summary(
        name=name,
        workers=workers,
        iterations_per_worker=iterations_per_worker,
        total_iterations=total_iterations,
        width=width,
        height=height,
        channels=channels,
        megapixels=megapixels,
        latency_ms={
            "mean": statistics.mean(pooled_means),
            "p50": percentile(sorted(pooled_p50), 50.0),
            "p95": percentile(sorted(pooled_p95), 95.0),
            "min": min(pooled_min),
            "max": max(pooled_max),
        },
        per_worker_fps=per_worker_fps,
        per_worker_mps=per_worker_mps,
        aggregate_fps=aggregate_fps,
        aggregate_mps=aggregate_mps,
        total_worker_seconds=total_worker_seconds,
        batch_wall_seconds=batch_wall_seconds,
        overhead=overhead,
        hardware=hardware,
        destination=destination,
    )


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"
