"""Prepare a stable package of core FIR thesis figures.

The analysis scripts write many exploratory plots into outputs/. This helper
copies the small set of thesis-facing plots into a named folder with a README,
so the report/viva figures are easier to find.
"""

from __future__ import annotations

import shutil
from pathlib import Path


NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"
PACKAGE_DIR = OUT_DIR / "thesis_figure_package"

FIGURES = [
    {
        "order": 1,
        "source": "external_count_source_flux_coverage.png",
        "target": "fig01_external_count_source_coverage.png",
        "title": "External SPIRE Count Source Coverage",
        "use": "Methods / data-source justification",
        "caption": (
            "Flux-density coverage of the external SPIRE count sources used in "
            "the evaluator. Resolved/prior count products cover the directly "
            "detected regime, while P(D) analyses extend to much fainter fluxes "
            "as statistical constraints."
        ),
    },
    {
        "order": 2,
        "source": "popcosmos_model_family_score_comparison.png",
        "target": "fig02_model_family_score_comparison.png",
        "title": "Model Family Score Comparison",
        "use": "Main result figure",
        "caption": (
            "Comparison of far-IR SED model variants using the published SPIRE "
            "differential-count evaluator. The best current models are Casey-like "
            "templates around T ~ 30 K, while the baseline FSPS/pop-cosmos SED is "
            "substantially worse."
        ),
    },
    {
        "order": 3,
        "source": "popcosmos_model_family_flux_regime_residual_heatmap.png",
        "target": "fig03_fsps_flux_regime_residual_heatmap.png",
        "title": "FSPS Flux-Regime Residuals",
        "use": "Diagnostic result figure",
        "caption": (
            "Median log residuals between model and observed SPIRE differential "
            "counts, split by wavelength and flux regime. Positive residuals mean "
            "the model predicts too many counts. The FSPS mismatch gets worse at "
            "longer wavelength and high flux."
        ),
    },
    {
        "order": 4,
        "source": "popcosmos_casey_like_template_grid_counts.png",
        "target": "fig04_casey_like_count_comparison.png",
        "title": "Casey-Like Template Count Comparison",
        "use": "Model-extension result figure",
        "caption": (
            "Casey-like far-IR SED grid used as a physically motivated template "
            "extension. Each template is normalised to preserve the original "
            "pop-cosmos L_IR before predicting SPIRE differential counts."
        ),
    },
    {
        "order": 5,
        "source": "popcosmos_model_family_leave_one_source_out.png",
        "target": "fig05_leave_one_source_out.png",
        "title": "Leave-One-Source-Out Validation",
        "use": "Overfitting guard / robustness figure",
        "caption": (
            "Leave-one-source-out test for the count evaluator. For each observed "
            "count source, the best model is selected using all other sources and "
            "then evaluated on the held-out source."
        ),
    },
    {
        "order": 6,
        "source": "popcosmos_model_family_pd_sensitivity.png",
        "target": "fig06_pd_sensitivity.png",
        "title": "P(D) Sensitivity",
        "use": "Robustness / caveat figure",
        "caption": (
            "Sensitivity of the model ranking to the inclusion of P(D) statistical "
            "count constraints. P(D) changes the exact best alpha, but not the "
            "broad preference for a warm Casey-like dust SED."
        ),
    },
    {
        "order": 7,
        "source": "popcosmos_model_family_source_tension.png",
        "target": "fig07_source_to_source_tension.png",
        "title": "Source-To-Source Tension",
        "use": "Robustness / caveat figure",
        "caption": (
            "Best-fitting model family for each external count source. All sources "
            "prefer a model variant over baseline FSPS, but the exact preferred "
            "template differs between count products."
        ),
    },
    {
        "order": 8,
        "source": "popcosmos_evaluator_chi2_dof_check.png",
        "target": "fig08_chi_square_dof_check.png",
        "title": "Chi-Square Degrees-Of-Freedom Check",
        "use": "Methods appendix / supervisor discussion",
        "caption": (
            "Comparison between chi2/N and a simple chi2/(N-k) degrees-of-freedom "
            "correction. The correction barely changes the ranking."
        ),
    },
    {
        "order": 9,
        "source": "popcosmos_formal_evaluator_summary.png",
        "target": "fig09_formal_evaluator_summary.png",
        "title": "Formal Resolved/Prior Evaluator Summary",
        "use": "Thesis-facing main-score figure",
        "caption": (
            "Thesis-facing evaluator summary using corrected resolved/prior "
            "SPIRE differential counts as the formal score. P(D) constraints are "
            "shown as a sensitivity comparison rather than the main score."
        ),
    },
    {
        "order": 10,
        "source": "wang_raw_count_context.png",
        "target": "fig10_wang_raw_count_context.png",
        "title": "Wang Raw Count Context",
        "use": "Appendix / Wang diagnostic context",
        "caption": (
            "Raw cumulative counts from the Wang COSMOS deblended catalogue under "
            "different area and selection assumptions, compared with direct "
            "published integral-count points from Clements and Pearson. This is "
            "a diagnostic plot showing why Wang should not replace corrected "
            "published differential counts in the formal evaluator."
        ),
    },
]


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for fig in FIGURES:
        source_path = OUT_DIR / fig["source"]
        target_path = PACKAGE_DIR / fig["target"]
        if not source_path.exists():
            missing.append(source_path)
            continue
        shutil.copy2(source_path, target_path)
        rows.append({**fig, "path": target_path.name})

    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing figure source(s):\n{missing_text}")

    readme_lines = [
        "# FIR Thesis Figure Package",
        "",
        "Stable copies of the main FIR/SPIRE figures for the thesis and viva.",
        "",
        "| # | file | title | use |",
        "|---:|---|---|---|",
    ]
    for row in rows:
        readme_lines.append(
            f"| {row['order']} | `{row['path']}` | {row['title']} | {row['use']} |"
        )

    readme_lines.extend(["", "## Captions", ""])
    for row in rows:
        readme_lines.extend(
            [
                f"### {row['order']}. {row['title']}",
                "",
                f"File: `{row['path']}`",
                "",
                row["caption"],
                "",
            ]
        )

    (PACKAGE_DIR / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    manifest_lines = ["order,source,target,title,use"]
    for row in rows:
        manifest_lines.append(
            f"{row['order']},{row['source']},{row['target']},\"{row['title']}\",\"{row['use']}\""
        )
    (PACKAGE_DIR / "manifest.csv").write_text("\n".join(manifest_lines), encoding="utf-8")

    print(PACKAGE_DIR)
    print(PACKAGE_DIR / "README.md")
    print(PACKAGE_DIR / "manifest.csv")


if __name__ == "__main__":
    main()
