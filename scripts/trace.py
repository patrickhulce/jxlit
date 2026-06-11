"""Convert saved jxlit benchmark measures into a Chrome Trace Event profile.

Reads the per-language telemetry JSON written by ``benchmark.py`` (under
``.data/benchmarks/raw``) and emits a single combined trace
(``.data/benchmarks/profiles/trace_<ISO>.json``) loadable in chrome://tracing
or https://ui.perfetto.dev. Each language becomes its own process track; the
nested phase measures render as a flame chart.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / ".data" / "benchmarks" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".data" / "benchmarks" / "profiles"

# Single decode thread per language; measures are already nested by time.
DECODE_TID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved benchmark measures into a Chrome trace",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Directory of per-language telemetry JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the combined trace into",
    )
    return parser.parse_args()


def latest_per_language(input_dir: Path) -> dict[str, dict[str, Any]]:
    """Return the most recent telemetry payload for each language.

    Filenames are suffixed with an ISO timestamp, so the lexicographically
    greatest filename per language is the newest run.
    """
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        lang = str(payload.get("lang", path.stem))
        existing = latest.get(lang)
        if existing is None or path.name > existing[0]:
            latest[lang] = (path.name, payload)
    return {lang: payload for lang, (_, payload) in latest.items()}


def build_trace_events(by_lang: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pid, lang in enumerate(sorted(by_lang), start=1):
        payload = by_lang[lang]
        # Label the process track with the language name.
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": DECODE_TID,
                "args": {"name": lang},
            }
        )
        for measure in payload.get("measures", []):
            if not isinstance(measure, dict):
                continue
            start_ms = float(measure.get("start_ms", 0))
            duration_ms = float(measure.get("duration_ms", 0))
            events.append(
                {
                    "name": str(measure.get("name", "")),
                    "cat": "decode",
                    "ph": "X",
                    "ts": start_ms * 1000.0,
                    "dur": duration_ms * 1000.0,
                    "pid": pid,
                    "tid": DECODE_TID,
                }
            )
    return events


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")

    by_lang = latest_per_language(input_dir)
    if not by_lang:
        raise SystemExit(f"no telemetry JSON files found in {input_dir}")

    trace = {
        "traceEvents": build_trace_events(by_lang),
        "displayTimeUnit": "ns",
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = output_dir / f"trace_{stamp}.json"
    out_path.write_text(json.dumps(trace, indent=2))

    langs = ", ".join(sorted(by_lang))
    print(f"wrote to {out_path.parent} ({langs})")
    print("open it in chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
