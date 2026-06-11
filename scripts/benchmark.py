"""Orchestrate cross-language jxlit decode benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / ".data" / "benchmarks" / "raw"
DEFAULT_FILE = REPO_ROOT / "assets" / "frame_4K_10bit_e1_d0p5_fd4.jxl"
DEFAULT_ACTION = "decode_cpu"
DEFAULT_ITERATIONS = 100
DEFAULT_WORKERS = 1
DEFAULT_LANGS = ("rust", "python", "node", "wasm")
SUPPORTED_LANGS = frozenset(DEFAULT_LANGS)


@dataclass(frozen=True)
class WorkerResult:
    lang: str
    action: str
    iterations: int
    width: int
    height: int
    channels: int
    megapixels: float
    decode_seconds: float
    latency_ms: dict[str, float]
    telemetry: dict[str, Any] | None = None


@dataclass(frozen=True)
class LanguageSummary:
    lang: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run jxlit decode benchmarks across language bindings",
    )
    parser.add_argument(
        "--file",
        default=str(DEFAULT_FILE),
        help="Path to a JPEG XL file",
    )
    parser.add_argument(
        "--action",
        default=DEFAULT_ACTION,
        help="Benchmark action label (decode_cpu or decode_gpu)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Measured decode iterations per worker",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Thread count for decode (default: available CPU cores)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel worker processes per language",
    )
    parser.add_argument(
        "--langs",
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated languages to benchmark",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable post-loop phase telemetry collection",
    )
    parser.add_argument(
        "--layout",
        choices=("interleaved", "planar"),
        default="interleaved",
        help="Pixel buffer layout (default: interleaved)",
    )
    parser.add_argument(
        "--hardware",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Decode compute backend (default: cpu)",
    )
    parser.add_argument(
        "--destination",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Where decoded pixels reside (default: cpu)",
    )
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


def parse_worker_result(payload: dict[str, Any]) -> WorkerResult:
    telemetry = payload.get("telemetry")
    return WorkerResult(
        lang=str(payload["lang"]),
        action=str(payload["action"]),
        iterations=int(payload["iterations"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        channels=int(payload["channels"]),
        megapixels=float(payload["megapixels"]),
        decode_seconds=float(payload["decode_seconds"]),
        latency_ms={key: float(value) for key, value in payload["latency_ms"].items()},
        telemetry=telemetry if isinstance(telemetry, dict) else None,
    )


def build_command(
    lang: str,
    file_path: Path,
    action: str,
    iterations: int,
    threads: int | None,
    no_telemetry: bool,
    layout: str,
    hardware: str,
    destination: str,
) -> list[str]:
    common = [
        "--file",
        str(file_path),
        "--action",
        action,
        "--iterations",
        str(iterations),
        "--layout",
        layout,
    ]
    if threads is not None:
        common.extend(["--threads", str(threads)])
    if no_telemetry:
        common.append("--no-telemetry")

    if lang == "rust":
        common.extend(["--hardware", hardware, "--destination", destination])
        binary = REPO_ROOT / "target" / "release" / "jxlit-benchmark"
        return [str(binary), *common]

    if lang == "python":
        return ["uv", "run", "python", "-m", "jxlit.benchmark", *common]

    if lang == "node":
        return ["node", "--import", "tsx", "src/benchmark.ts", *common]

    if lang == "wasm":
        return [
            "node",
            "--experimental-wasm-modules",
            "--import",
            "tsx",
            "src/benchmark.ts",
            *common,
        ]

    raise ValueError(f"unsupported language: {lang}")


def run_worker(
    lang: str,
    file_path: Path,
    action: str,
    iterations: int,
    threads: int | None,
    no_telemetry: bool,
    layout: str,
    hardware: str,
    destination: str,
) -> WorkerResult:
    if lang != "rust" and (hardware != "cpu" or destination != "cpu"):
        raise RuntimeError(
            f"{lang} bindings do not support --hardware/--destination yet "
            f"(requested hardware={hardware}, destination={destination})",
        )

    command = build_command(
        lang,
        file_path,
        action,
        iterations,
        threads,
        no_telemetry,
        layout,
        hardware,
        destination,
    )
    if lang == "python":
        cwd = REPO_ROOT / "src" / "python-jxlit"
    elif lang in {"node", "wasm"}:
        cwd = REPO_ROOT / "src" / f"{lang}-jxlit"
    else:
        cwd = REPO_ROOT

    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{lang} worker failed (exit {completed.returncode}): {stderr}",
        )

    line = completed.stdout.strip().splitlines()[-1]
    payload = json.loads(line)
    return parse_worker_result(payload)


def aggregate_language(
    lang: str,
    workers: int,
    results: list[WorkerResult],
    batch_wall_seconds: float,
) -> LanguageSummary:
    if not results:
        raise ValueError(f"no worker results for {lang}")

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
    overhead = (
        1.0 - total_worker_seconds / (batch_wall_seconds * workers)
    ) * 100.0

    return LanguageSummary(
        lang=lang,
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
    )


def run_language_batch(
    lang: str,
    file_path: Path,
    action: str,
    iterations: int,
    workers: int,
    threads: int | None,
    no_telemetry: bool,
    layout: str,
    hardware: str,
    destination: str,
) -> tuple[LanguageSummary, dict[str, Any] | None]:
    batch_start = time.perf_counter()
    results: list[WorkerResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_worker,
                lang,
                file_path,
                action,
                iterations,
                threads,
                no_telemetry,
                layout,
                hardware,
                destination,
            )
            for _ in range(workers)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    batch_wall_seconds = time.perf_counter() - batch_start
    telemetry = next((result.telemetry for result in results if result.telemetry), None)
    return aggregate_language(lang, workers, results, batch_wall_seconds), telemetry


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def print_language_summary(summary: LanguageSummary) -> None:
    print(f"\n== {summary.lang} ==")
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
        "per_worker "
        f"fps={format_float(summary.per_worker_fps)} "
        f"MP/s={format_float(summary.per_worker_mps)}"
    )
    print(
        "aggregate "
        f"fps={format_float(summary.aggregate_fps)} "
        f"MP/s={format_float(summary.aggregate_mps)} "
        f"worker_seconds={format_float(summary.total_worker_seconds)} "
        f"wall_seconds={format_float(summary.batch_wall_seconds)} "
        f"overhead={format_percent(summary.overhead)}"
    )


def print_phase_summary(
    lang: str,
    telemetry: dict[str, Any],
    top_n: int = 10,
) -> None:
    measures = telemetry.get("measures")
    if not isinstance(measures, list) or not measures:
        return

    outer = next(
        (
            measure
            for measure in measures
            if isinstance(measure, dict)
            and str(measure.get("name", "")).endswith("_decode")
        ),
        None,
    )
    total_ms = float(outer["duration_ms"]) if isinstance(outer, dict) else float(
        telemetry.get("total_ms", 0)
    )

    ranked = sorted(
        (measure for measure in measures if isinstance(measure, dict)),
        key=lambda measure: float(measure.get("duration_ms", 0)),
        reverse=True,
    )[:top_n]

    print(f"\n== phase breakdown ({lang}) ==")
    for measure in ranked:
        name = str(measure.get("name", ""))
        duration_ms = float(measure.get("duration_ms", 0))
        pct = 100.0 * duration_ms / total_ms if total_ms else 0.0
        print(f"{name:<16} {duration_ms:8.2f}ms {pct:6.1f}%")


def save_raw_telemetry(
    lang: str,
    telemetry: dict[str, Any],
    stamp: str,
    *,
    file_name: str,
    threads: int | None,
    iterations: int,
    workers: int,
    layout: str,
    hardware: str,
    destination: str,
) -> Path:
    """Persist a language's rebased telemetry plus run context to disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "lang": lang,
        "file": file_name,
        "threads": threads,
        "iterations": iterations,
        "workers": workers,
        "layout": layout,
        "hardware": hardware,
        "destination": destination,
        **telemetry,
    }
    out_path = RAW_DIR / f"{lang}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_cross_language_table(summaries: list[LanguageSummary]) -> None:
    print("\n== cross-language summary ==")
    header = (
        f"{'lang':<8} {'workers':>7} {'fps':>10} {'MP/s':>10} "
        f"{'worker_s':>10} {'wall_s':>10} {'overhead%':>10}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.lang:<8} "
            f"{summary.workers:>7} "
            f"{format_float(summary.aggregate_fps):>10} "
            f"{format_float(summary.aggregate_mps):>10} "
            f"{format_float(summary.total_worker_seconds):>10} "
            f"{format_float(summary.batch_wall_seconds):>10} "
            f"{format_percent(summary.overhead):>10}"
        )


def main() -> None:
    args = parse_args()

    if args.action not in {"decode_cpu", "decode_gpu"}:
        print(
            f"unsupported action: {args.action} (expected decode_cpu or decode_gpu)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.iterations <= 0:
        print("--iterations must be greater than 0", file=sys.stderr)
        raise SystemExit(1)

    if args.workers <= 0:
        print("--workers must be greater than 0", file=sys.stderr)
        raise SystemExit(1)

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    unknown = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if unknown:
        print(f"unsupported languages: {', '.join(unknown)}", file=sys.stderr)
        raise SystemExit(1)

    file_path = Path(args.file).resolve()
    if not file_path.is_file():
        print(f"file not found: {file_path}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"benchmark action={args.action} file={file_path.name} "
        f"iterations/worker={args.iterations} workers={args.workers} "
        f"threads={args.threads if args.threads is not None else 'auto'} "
        f"layout={args.layout} hardware={args.hardware} destination={args.destination}"
    )

    summaries: list[LanguageSummary] = []
    telemetry_by_lang: dict[str, dict[str, Any]] = {}
    for lang in langs:
        summary, telemetry = run_language_batch(
            lang=lang,
            file_path=file_path,
            action=args.action,
            iterations=args.iterations,
            workers=args.workers,
            threads=args.threads,
            no_telemetry=args.no_telemetry,
            layout=args.layout,
            hardware=args.hardware,
            destination=args.destination,
        )
        summaries.append(summary)
        print_language_summary(summary)
        if telemetry is not None:
            telemetry_by_lang[lang] = telemetry

    print_cross_language_table(summaries)

    if not args.no_telemetry:
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        for lang in langs:
            telemetry = telemetry_by_lang.get(lang)
            if telemetry is not None:
                print_phase_summary(lang, telemetry)
                out_path = save_raw_telemetry(
                    lang,
                    telemetry,
                    stamp,
                    file_name=file_path.name,
                    threads=args.threads,
                    iterations=args.iterations,
                    workers=args.workers,
                    layout=args.layout,
                    hardware=args.hardware,
                    destination=args.destination,
                )
                print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
