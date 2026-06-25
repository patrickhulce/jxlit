"""Compare JPEG XL decoders, each running its fastest path, on the same inputs.

This orchestrator discovers self-contained decoder candidates under
``benchmarks/candidates/`` (each with a ``manifest.toml``), runs every config a
candidate declares, and aggregates per-worker latencies into a cross-decoder
table - reusing the exact timing methodology of ``benchmark.py`` via
``benchmark_common``.

Fairness rules:

* The input file is read once per worker, not re-read per iteration.
* ``--threads`` is forwarded to every candidate that supports intra-decode
  parallelism (all of them use the same flag name).
* ``--workers`` stays a first-class knob: it launches N concurrent worker
  processes per config so multi-instance scaling is visible. Use
  ``--worker-sweep`` to chart how each decoder scales.
* Every candidate fully materializes f32/native pixels and keeps a live
  reference so the decode cannot be optimized away.
* GPU configs are skipped (not failed) when no device is available.

The candidate's pixel output classes differ on purpose (libjxl forces f32 RGB,
djxl preserves native bit depth, imagecodecs adds a torch upload for GPU dest,
jxlit can leave pixels in a GPU buffer); that is what we want to compare.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_common import (
    Summary,
    aggregate_results,
    format_float,
    format_percent,
    parse_worker_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = REPO_ROOT / "benchmarks" / "candidates"
COMPARE_DIR = REPO_ROOT / ".data" / "benchmarks" / "compare"
DEFAULT_FILE = REPO_ROOT / "assets" / "frame_4K_10bit_e1_d0p5_fd4.jxl"
DEFAULT_ITERATIONS = 100
DEFAULT_WORKERS = 1
DEFAULT_LAYOUT = "planar"

# Preferred display order; unknown candidates are appended alphabetically.
CANDIDATE_ORDER = ("libjxl", "djxl", "jxl-oxide", "jxl-rs", "imagecodecs", "jxlit")


@dataclass(frozen=True)
class Config:
    hardware: str
    destination: str
    layout: str | None
    action: str
    requires: str | None


@dataclass(frozen=True)
class Candidate:
    name: str
    directory: Path
    program: str
    base_args: list[str]
    cwd: str | None
    configs: list[Config]


@dataclass(frozen=True)
class SkipResult:
    name: str
    hardware: str
    destination: str
    reason: str


def load_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for manifest_path in sorted(CANDIDATES_DIR.glob("*/manifest.toml")):
        data = tomllib.loads(manifest_path.read_text())
        configs = [
            Config(
                hardware=str(cfg.get("hardware", "cpu")),
                destination=str(cfg.get("destination", "cpu")),
                layout=cfg.get("layout"),
                action=str(cfg.get("action", "decode_cpu")),
                requires=cfg.get("requires"),
            )
            for cfg in data.get("configs", [])
        ]
        candidates.append(
            Candidate(
                name=str(data["name"]),
                directory=manifest_path.parent,
                program=str(data["program"]),
                base_args=[str(a) for a in data.get("base_args", [])],
                cwd=data.get("cwd"),
                configs=configs,
            )
        )

    def order_key(candidate: Candidate) -> tuple[int, str]:
        try:
            return (CANDIDATE_ORDER.index(candidate.name), "")
        except ValueError:
            return (len(CANDIDATE_ORDER), candidate.name)

    return sorted(candidates, key=order_key)


def gpu_available() -> bool:
    """Best-effort detection of any GPU usable by jxlit (wgpu) or torch."""
    import platform
    import shutil

    if platform.system() == "Darwin":
        return True  # Metal (wgpu) and MPS (torch) on Apple platforms
    if shutil.which("nvidia-smi") or shutil.which("vulkaninfo"):
        return True
    dri = Path("/dev/dri")
    if dri.is_dir() and any(p.name.startswith("renderD") for p in dri.iterdir()):
        return True
    return False


def substitute(value: str, candidate: Candidate) -> str:
    return value.format(
        repo_root=str(REPO_ROOT),
        candidate_dir=str(candidate.directory),
    )


def resolve_program(candidate: Candidate) -> str:
    program = substitute(candidate.program, candidate)
    path = Path(program)
    if not path.is_absolute() and ("/" in candidate.program):
        path = candidate.directory / program
        return str(path)
    return program


def resolve_cwd(candidate: Candidate) -> Path:
    if candidate.cwd is None:
        return candidate.directory
    resolved = substitute(candidate.cwd, candidate)
    if resolved == ".":
        return candidate.directory
    return Path(resolved)


def build_command(
    candidate: Candidate,
    config: Config,
    *,
    file_path: Path,
    iterations: int,
    threads: int | None,
    layout: str,
) -> list[str]:
    program = resolve_program(candidate)
    base_args = [substitute(arg, candidate) for arg in candidate.base_args]
    standard = [
        "--file",
        str(file_path),
        "--action",
        config.action,
        "--iterations",
        str(iterations),
        "--layout",
        config.layout or layout,
        "--hardware",
        config.hardware,
        "--destination",
        config.destination,
        "--no-telemetry",
    ]
    if threads is not None:
        standard.extend(["--threads", str(threads)])
    return [program, *base_args, *standard]


def run_worker(
    candidate: Candidate,
    config: Config,
    *,
    file_path: Path,
    iterations: int,
    threads: int | None,
    layout: str,
) -> Any:
    """Run one worker; returns a parsed WorkerResult or a SkipResult."""
    import subprocess

    command = build_command(
        candidate,
        config,
        file_path=file_path,
        iterations=iterations,
        threads=threads,
        layout=layout,
    )
    cwd = resolve_cwd(candidate)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return SkipResult(
            candidate.name,
            config.hardware,
            config.destination,
            "binary not built (run `make build-bench-candidates`)",
        )

    if completed.returncode != 0:
        stderr = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
        reason = stderr[-1] if stderr else f"exit {completed.returncode}"
        return SkipResult(candidate.name, config.hardware, config.destination, reason)

    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return SkipResult(
            candidate.name, config.hardware, config.destination, "no output"
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return SkipResult(
            candidate.name, config.hardware, config.destination, "unparseable output"
        )

    if payload.get("skipped"):
        return SkipResult(
            candidate.name,
            config.hardware,
            config.destination,
            str(payload.get("reason", "skipped")),
        )
    # jxlit-benchmark emits `"lang":"rust"`; always label rows by the candidate name.
    parsed = parse_worker_result(payload)
    return replace(parsed, name=candidate.name)


def run_config(
    candidate: Candidate,
    config: Config,
    *,
    file_path: Path,
    iterations: int,
    workers: int,
    threads: int | None,
    layout: str,
) -> Summary | SkipResult:
    if config.requires == "gpu" and not gpu_available():
        return SkipResult(
            candidate.name, config.hardware, config.destination, "no GPU detected"
        )

    batch_start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_worker,
                candidate,
                config,
                file_path=file_path,
                iterations=iterations,
                threads=threads,
                layout=layout,
            )
            for _ in range(workers)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    batch_wall_seconds = time.perf_counter() - batch_start

    skip = next((r for r in results if isinstance(r, SkipResult)), None)
    if skip is not None:
        return skip

    return aggregate_results(
        candidate.name,
        workers,
        results,
        batch_wall_seconds,
        hardware=config.hardware,
        destination=config.destination,
    )


def config_label(summary_or_skip: Summary | SkipResult) -> str:
    return f"{summary_or_skip.name} [{summary_or_skip.hardware}/{summary_or_skip.destination}]"


def print_config_summary(summary: Summary) -> None:
    print(f"\n== {config_label(summary)} ==")
    print(
        f"workers={summary.workers} "
        f"iterations/worker={summary.iterations_per_worker} "
        f"total_iterations={summary.total_iterations}"
    )
    print(
        f"image={summary.width}x{summary.height} "
        f"channels={summary.channels} "
        f"megapixels={format_float(summary.megapixels, 6)}"
    )
    print(
        "latency_ms "
        f"mean={format_float(summary.latency_ms['mean'], 3)} "
        f"p50={format_float(summary.latency_ms['p50'], 3)} "
        f"p95={format_float(summary.latency_ms['p95'], 3)} "
        f"min={format_float(summary.latency_ms['min'], 3)} "
        f"max={format_float(summary.latency_ms['max'], 3)}"
    )
    print(
        "aggregate "
        f"fps={format_float(summary.aggregate_fps)} "
        f"MP/s={format_float(summary.aggregate_mps)} "
        f"per_worker_MP/s={format_float(summary.per_worker_mps)} "
        f"overhead={format_percent(summary.overhead)}"
    )


def print_compare_table(
    summaries: list[Summary], skips: list[SkipResult], workers: int
) -> None:
    print(f"\n== decoder comparison (workers={workers}) ==")
    header = (
        f"{'decoder':<12} {'hw':>4} {'dest':>5} {'p50_ms':>9} "
        f"{'agg_MP/s':>10} {'1w_MP/s':>10} {'overhead':>9}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.name:<12} "
            f"{summary.hardware:>4} "
            f"{summary.destination:>5} "
            f"{format_float(summary.latency_ms['p50'], 2):>9} "
            f"{format_float(summary.aggregate_mps):>10} "
            f"{format_float(summary.per_worker_mps):>10} "
            f"{format_percent(summary.overhead):>9}"
        )
    for skip in skips:
        print(
            f"{skip.name:<12} {skip.hardware:>4} {skip.destination:>5} "
            f"{'skipped':>9}  ({skip.reason})"
        )


def print_scaling_table(
    sweep_results: dict[int, list[Summary]], worker_counts: list[int]
) -> None:
    print("\n== worker scaling (aggregate MP/s) ==")
    keys: list[tuple[str, str, str]] = []
    by_key: dict[tuple[str, str, str], dict[int, float]] = {}
    for workers in worker_counts:
        for summary in sweep_results.get(workers, []):
            key = (summary.name, summary.hardware, summary.destination)
            if key not in by_key:
                by_key[key] = {}
                keys.append(key)
            by_key[key][workers] = summary.aggregate_mps

    header = f"{'decoder [hw/dest]':<26}" + "".join(
        f"{f'{w}w':>10}" for w in worker_counts
    )
    print(header)
    print("-" * len(header))
    for key in keys:
        name, hardware, destination = key
        label = f"{name} [{hardware}/{destination}]"
        row = f"{label:<26}"
        for workers in worker_counts:
            value = by_key[key].get(workers)
            row += f"{format_float(value) if value is not None else '-':>10}"
        print(row)


def save_summary(summary: Summary, stamp: str, *, file_name: str, threads: int | None) -> Path:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "decoder": summary.name,
        "hardware": summary.hardware,
        "destination": summary.destination,
        "file": file_name,
        "threads": threads,
        "workers": summary.workers,
        "iterations_per_worker": summary.iterations_per_worker,
        "total_iterations": summary.total_iterations,
        "width": summary.width,
        "height": summary.height,
        "channels": summary.channels,
        "megapixels": summary.megapixels,
        "latency_ms": summary.latency_ms,
        "per_worker_fps": summary.per_worker_fps,
        "per_worker_mps": summary.per_worker_mps,
        "aggregate_fps": summary.aggregate_fps,
        "aggregate_mps": summary.aggregate_mps,
        "total_worker_seconds": summary.total_worker_seconds,
        "batch_wall_seconds": summary.batch_wall_seconds,
        "overhead": summary.overhead,
    }
    name = (
        f"{summary.name}_{summary.hardware}-{summary.destination}"
        f"_{summary.workers}w_{stamp}.json"
    )
    out_path = COMPARE_DIR / name
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def run_matrix(
    candidates: list[Candidate],
    *,
    file_path: Path,
    iterations: int,
    workers: int,
    threads: int | None,
    layout: str,
) -> tuple[list[Summary], list[SkipResult]]:
    summaries: list[Summary] = []
    skips: list[SkipResult] = []
    for candidate in candidates:
        for config in candidate.configs:
            result = run_config(
                candidate,
                config,
                file_path=file_path,
                iterations=iterations,
                workers=workers,
                threads=threads,
                layout=layout,
            )
            if isinstance(result, SkipResult):
                skips.append(result)
                print(f"\n-- {config_label(result)} skipped: {result.reason}")
            else:
                summaries.append(result)
                print_config_summary(result)
    return summaries, skips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare JPEG XL decoders on identical inputs",
    )
    parser.add_argument("--file", default=str(DEFAULT_FILE))
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument(
        "--decoders",
        default="",
        help="Comma-separated subset of candidate names (default: all)",
    )
    parser.add_argument(
        "--worker-sweep",
        default="",
        help="Comma-separated worker counts to chart scaling, e.g. 1,2,4,8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.is_file():
        print(f"file not found: {file_path}", file=sys.stderr)
        raise SystemExit(1)

    candidates = load_candidates()
    if not candidates:
        print(f"no candidates found under {CANDIDATES_DIR}", file=sys.stderr)
        raise SystemExit(1)

    if args.decoders.strip():
        wanted = {name.strip() for name in args.decoders.split(",") if name.strip()}
        unknown = wanted - {c.name for c in candidates}
        if unknown:
            print(f"unknown decoders: {', '.join(sorted(unknown))}", file=sys.stderr)
            raise SystemExit(1)
        candidates = [c for c in candidates if c.name in wanted]

    sweep = [int(w) for w in args.worker_sweep.split(",") if w.strip()]
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    print(
        f"compare file={file_path.name} iterations/worker={args.iterations} "
        f"threads={args.threads if args.threads is not None else 'auto'} "
        f"layout={args.layout} decoders={','.join(c.name for c in candidates)}"
    )

    if sweep:
        sweep_results: dict[int, list[Summary]] = {}
        for workers in sweep:
            print(f"\n######## worker count: {workers} ########")
            summaries, skips = run_matrix(
                candidates,
                file_path=file_path,
                iterations=args.iterations,
                workers=workers,
                threads=args.threads,
                layout=args.layout,
            )
            print_compare_table(summaries, skips, workers)
            for summary in summaries:
                save_summary(summary, stamp, file_name=file_path.name, threads=args.threads)
            sweep_results[workers] = summaries
        print_scaling_table(sweep_results, sweep)
        return

    summaries, skips = run_matrix(
        candidates,
        file_path=file_path,
        iterations=args.iterations,
        workers=args.workers,
        threads=args.threads,
        layout=args.layout,
    )
    print_compare_table(summaries, skips, args.workers)
    for summary in summaries:
        out_path = save_summary(
            summary, stamp, file_name=file_path.name, threads=args.threads
        )
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
