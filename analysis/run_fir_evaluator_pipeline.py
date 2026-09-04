"""Run the FIR evaluator pipeline in a reproducible order.

Use:
    python run_fir_evaluator_pipeline.py --mode summary
    python run_fir_evaluator_pipeline.py --mode full

`summary` refreshes the combined diagnostics from existing model-count CSVs.
`full` also rebuilds the heavier model-count products.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


NB_DIR = Path(__file__).resolve().parent

ALWAYS_STEPS = [
    "compile_external_spire_differential_counts.py",
    "popcosmos_wang_catalog_checks.py",
    "plot_wang_raw_count_context.py",
    "plot_external_count_source_coverage.py",
]

FULL_ONLY_STEPS = [
    "popcosmos_restframe_hybrid_sed.py",
    "popcosmos_differential_count_evaluator.py",
    "popcosmos_mbb_temperature_grid.py",
    "popcosmos_casey_like_template_grid.py",
]

SUMMARY_STEPS = [
    "popcosmos_model_family_score_summary.py",
    "popcosmos_evaluator_chi2_dof_check.py",
    "popcosmos_model_family_pd_sensitivity.py",
    "popcosmos_formal_evaluator_summary.py",
    "popcosmos_model_family_leave_one_out.py",
    "popcosmos_model_family_source_tension.py",
    "popcosmos_model_family_flux_regime_diagnostics.py",
    "popcosmos_thesis_evaluator_snapshot.py",
    "prepare_fir_thesis_figure_package.py",
]


def run_step(script_name: str) -> None:
    script_path = NB_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing pipeline step: {script_path}")

    print(f"\n=== {script_name} ===", flush=True)
    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(script_path)], cwd=NB_DIR)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")
    print(f"--- completed in {elapsed:.1f}s ---", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pop-cosmos FIR evaluator pipeline.")
    parser.add_argument(
        "--mode",
        choices=["summary", "full"],
        default="summary",
        help="summary refreshes downstream diagnostics; full rebuilds heavy model grids too.",
    )
    args = parser.parse_args()

    steps = list(ALWAYS_STEPS)
    if args.mode == "full":
        steps.extend(FULL_ONLY_STEPS)
    else:
        steps.append("popcosmos_differential_count_evaluator.py")
    steps.extend(SUMMARY_STEPS)

    print(f"Running FIR evaluator pipeline in {args.mode!r} mode.")
    print(f"Working directory: {NB_DIR}")
    print("Steps:")
    for step in steps:
        print(f"- {step}")

    start = time.perf_counter()
    for step in steps:
        run_step(step)
    elapsed = time.perf_counter() - start

    print("\nPipeline complete.")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"Current snapshot: {NB_DIR / 'outputs' / 'popcosmos_thesis_evaluator_snapshot.md'}")


if __name__ == "__main__":
    main()
