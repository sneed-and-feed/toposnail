#!/usr/bin/env python3
"""
scripts/run_blind_cad_benchmark.py: Command-Line Runner for the TopoSnail Reference-Free
Blind CAD Ingestion & Intrinsic Physics/Manifold Benchmark Suite.

Usage:
  1. Run full 5-part blind CAD corpus benchmark:
       python scripts/run_blind_cad_benchmark.py
  2. Run on any custom unseen .STL or .OBJ CAD file cold:
       python scripts/run_blind_cad_benchmark.py --input path/to/part.stl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

# Ensure repository root is on sys.path when invoked directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toposnail.blind_cad_benchmark import (
    format_blind_cad_benchmark_tables,
    run_blind_cad_corpus_benchmark,
)
from toposnail.scenarios import CANONICAL_GRID


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_blind_cad_benchmark",
        description=(
            "TopoSnail Reference-Free Blind CAD Ingestion & 8-Law Intrinsic "
            "Physics/Manifold Benchmark Suite (Zero Example / Zero Template Required)"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to a custom unseen .STL or .OBJ file (or folder of blind CAD files). Defaults to full 5-part blind corpus.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="results/blind_cad_inputs",
        help="Directory containing the 5-part blind CAD corpus (default: results/blind_cad_inputs)",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="results/blind_cad_benchmark_summary.json",
        help="Output path for the JSON benchmark summary (default: results/blind_cad_benchmark_summary.json)",
    )
    parser.add_argument(
        "--audit-png",
        type=str,
        default="results/audit_blind_cad_benchmark.png",
        help="Output path for the 5x4 visual progression & intrinsic telemetry PNG sheet (default: results/audit_blind_cad_benchmark.png)",
    )
    parser.add_argument(
        "-v",
        "--v-budget",
        type=float,
        default=0.25,
        help="Target Stage 4 solid volume fraction relative to Stage 1 stock (default: 0.25)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=25,
        help="Maximum optimization iterations per solver rollout (default: 25)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="32,24,24",
        help="3D hexahedral voxel grid resolution Z,Y,X (default: 32,24,24)",
    )

    args = parser.parse_args(argv)

    try:
        parts = [int(x.strip()) for x in args.resolution.split(",")]
        grid_shape = (parts[0], parts[1], parts[2])
    except Exception:
        grid_shape = CANONICAL_GRID

    summary = run_blind_cad_corpus_benchmark(
        input_path=args.input,
        corpus_dir=args.corpus_dir,
        summary_json_path=args.summary_json,
        audit_png_path=args.audit_png,
        grid_shape=grid_shape,
        v_budget=args.v_budget,
        max_iters=args.max_iters,
    )

    table_report = format_blind_cad_benchmark_tables(summary)
    print(table_report)
    print(
        f"\n[SUMMARY] Evaluated {summary['num_parts_evaluated']} blind CAD part(s) cold in {summary['elapsed_sec']:.2f}s | "
        f"TopoSnail 8/8 Intrinsic Pass Rate: {summary['toposnail_parts_passed_8_of_8']}/{summary['num_parts_evaluated']} "
        f"({summary['toposnail_pass_rate_pct']:.1f}%)"
    )
    if "summary_json" in summary:
        print(f"[JSON]    Saved benchmark summary -> {summary['summary_json']}")
    if "audit_sheet_png" in summary:
        print(f"[VISUAL]  Saved 4-stage visual progression & telemetry sheet -> {summary['audit_sheet_png']}")

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
