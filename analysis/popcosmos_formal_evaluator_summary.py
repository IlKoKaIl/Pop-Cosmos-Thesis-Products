"""Write the thesis-facing FIR evaluator summary.

The exploratory scripts keep several useful scores around:

- all external counts together
- resolved/prior extracted counts only
- P(D) statistical counts only

For the thesis, the clean default is slightly more specific:

- formal score: resolved/prior differential counts
- sensitivity check: P(D) counts
- diagnostics: Wang matched-object checks

This script does not recompute model fluxes. It just turns the existing score
tables into a small markdown/CSV/plot that is easier to defend in writing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"

POOLED = OUT_DIR / "popcosmos_model_family_score_summary.csv"
REGIME = OUT_DIR / "popcosmos_model_family_regime_summary.csv"
PER_SOURCE = OUT_DIR / "popcosmos_model_family_per_source_scores.csv"

OUT_CSV = OUT_DIR / "popcosmos_formal_evaluator_summary.csv"
OUT_MD = OUT_DIR / "popcosmos_formal_evaluator_summary.md"
OUT_PNG = OUT_DIR / "popcosmos_formal_evaluator_summary.png"

FORMAL_REGIME = "resolved/prior counts"
PD_REGIME = "P(D) statistical counts"
ALL_REGIME = "all scored counts"

TEMPLATE_PARAM_COUNTS = {
    "baseline": 0,
    "empirical template": 0,
    "FSPS/ALESS hybrid": 1,
    "modified blackbody": 1,
    "Casey-like template": 2,
}


def add_simple_dof(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["k_simple"] = out["model_family"].map(TEMPLATE_PARAM_COUNTS).fillna(0).astype(int)
    out["dof_simple"] = np.maximum(out["N_points"].astype(float) - out["k_simple"], 1)
    out["chi2_over_dof_simple"] = out["chi2_log"].astype(float) / out["dof_simple"]
    return out


def fmt(value, digits=2):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def pick_model(df: pd.DataFrame, label: str):
    sub = df[df["model_label"] == label]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_summary_table(regime_scores: pd.DataFrame) -> pd.DataFrame:
    formal = regime_scores[regime_scores["regime"] == FORMAL_REGIME].sort_values("chi2_over_dof_simple")
    pd_scores = regime_scores[regime_scores["regime"] == PD_REGIME].sort_values("chi2_over_dof_simple")
    all_scores = regime_scores[regime_scores["regime"] == ALL_REGIME].sort_values("chi2_over_dof_simple")

    selected_labels = list(formal.head(8)["model_label"])
    for label in ["FSPS", "ALESS", "25% ALESS", "50% ALESS", "MBB 35 K"]:
        if label not in selected_labels:
            selected_labels.append(label)

    rows = []
    for label in selected_labels:
        f = pick_model(formal, label)
        p = pick_model(pd_scores, label)
        a = pick_model(all_scores, label)
        if f is None:
            continue
        rows.append(
            {
                "model_label": label,
                "model_family": f["model_family"],
                "formal_resolved_prior_chi2_over_n": f["reduced_chi2_log"],
                "formal_resolved_prior_chi2_over_dof": f["chi2_over_dof_simple"],
                "formal_resolved_prior_median_log_model_over_obs": f[
                    "median_log10_model_over_obs"
                ],
                "pd_sensitivity_chi2_over_n": p["reduced_chi2_log"] if p is not None else np.nan,
                "all_counts_chi2_over_n": a["reduced_chi2_log"] if a is not None else np.nan,
                "N_formal_points": int(f["N_points"]),
                "k_simple": int(f["k_simple"]),
            }
        )
    return pd.DataFrame(rows).sort_values("formal_resolved_prior_chi2_over_dof")


def make_plot(summary: pd.DataFrame):
    plot_df = summary.head(10).copy()
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(
        y,
        plot_df["formal_resolved_prior_chi2_over_dof"],
        color="#4C78A8",
        label=r"formal score: resolved/prior $\chi^2/(N-k)$",
    )
    ax.scatter(
        plot_df["pd_sensitivity_chi2_over_n"],
        y,
        marker="D",
        s=46,
        color="#E15759",
        label=r"P(D) sensitivity $\chi^2/N$",
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(r"rough reduced $\chi^2$ in log-count space")
    ax.set_title("Thesis-facing FIR evaluator summary")
    ax.grid(True, axis="x", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def source_count_table() -> list[dict]:
    if not PER_SOURCE.exists():
        return []
    per_source = pd.read_csv(PER_SOURCE)
    source_col = "source" if "source" in per_source.columns else "external_source"
    sources = []
    for source, group in per_source.groupby(source_col):
        kind = "P(D) sensitivity" if "P(D)" in source else "formal resolved/prior"
        if "Wang raw" in source:
            kind = "diagnostic only"
        sources.append({"source": source, "role": kind, "models scored": group["model_label"].nunique()})
    return sorted(sources, key=lambda r: (r["role"], r["source"]))


def write_markdown(summary: pd.DataFrame):
    best = summary.iloc[0]
    fsps = summary[summary["model_label"] == "FSPS"].iloc[0]
    improvement = fsps["formal_resolved_prior_chi2_over_dof"] / best[
        "formal_resolved_prior_chi2_over_dof"
    ]

    top_rows = []
    for _, row in summary.head(10).iterrows():
        top_rows.append(
            {
                "model": row["model_label"],
                "family": row["model_family"],
                "formal chi2/(N-k)": fmt(row["formal_resolved_prior_chi2_over_dof"], 2),
                "formal median log ratio": fmt(
                    row["formal_resolved_prior_median_log_model_over_obs"], 2
                ),
                "P(D) chi2/N": fmt(row["pd_sensitivity_chi2_over_n"], 2),
            }
        )

    source_rows = source_count_table()

    text = f"""# Thesis-Facing FIR Evaluator Summary

This is the version of the evaluator I would currently defend in the thesis.

## Default Scoring Choice

Formal score:

> corrected/published resolved-or-prior differential SPIRE counts only.

That means:

- Clements et al. 2010
- Oliver et al. 2010
- Pearson et al. 2025 SUSSEXtractor
- Pearson et al. 2025 XID

Sensitivity check:

- Glenn et al. 2010 P(D)
- Varnish et al. 2025 P(D)

Diagnostic-only:

- Wang raw/matched catalogue checks

Reason:

> differential count bins are the cleanest thing to score, while P(D) and Wang are valuable but have extra correlation/selection complications.

## Score Definition

The evaluator compares model and observed differential counts in log space:

```text
chi2 = sum((log10(model_count) - log10(observed_count))^2 / sigma_log^2)
```

The simple thesis-facing score here is:

```text
rough reduced chi2 = chi2 / (N_points - k)
```

where `k` is a small template-parameter count:

- FSPS / ALESS fixed templates: `0`
- FSPS-ALESS hybrid fraction: `1`
- modified blackbody temperature: `1`
- Casey-like template: `2`

This is still a diagnostic score, not a full likelihood model.

## Current Formal Result

Best formal resolved/prior model:

- `{best['model_label']}`
- formal rough chi2/(N-k): `{best['formal_resolved_prior_chi2_over_dof']:.2f}`
- median log10(model/observed): `{best['formal_resolved_prior_median_log_model_over_obs']:.2f} dex`

Baseline FSPS:

- formal rough chi2/(N-k): `{fsps['formal_resolved_prior_chi2_over_dof']:.2f}`
- median log10(model/observed): `{fsps['formal_resolved_prior_median_log_model_over_obs']:.2f} dex`

So by this formal resolved/prior score, baseline FSPS is about `{improvement:.1f}x` worse than the current best model.

## Formal Ranking

{markdown_table(top_rows, ["model", "family", "formal chi2/(N-k)", "formal median log ratio", "P(D) chi2/N"])}

## Interpretation

Simple thesis line:

> Using corrected resolved/prior SPIRE differential counts as the formal evaluator, the baseline FSPS/pop-cosmos FIR SED is worse than warm/broader dust-template variants. The current best formal model is Casey-like around 30 K, but the safe conclusion is dust-SED flexibility, not one unique template.

Important nuance:

- P(D) points are useful but correlated, so they are better as a sensitivity check unless supervisors say otherwise.
- Wang is useful for matched-object residuals, not the formal corrected count score.
- A high chi-square does not mean overfitting. It means mismatch relative to the adopted errors/model assumptions.
- Overfitting is checked by leave-one-source-out, by not tuning separate templates per paper/band, and by keeping physical constraints like fixed `L_IR`.

## Count Source Roles In Current Score Files

{markdown_table(source_rows, ["source", "role", "models scored"])}

## Report Wording

Use:

> I define the main evaluator using corrected published differential counts from resolved/prior extraction methods, because these flux bins are closest to independent measurements. I then repeat the comparison with P(D) constraints as a sensitivity test. This prevents the model choice from being driven only by correlated faint-end statistical constraints.

Avoid:

> I fit pop-cosmos to all available count data and found the true dust temperature.
"""
    OUT_MD.write_text(text, encoding="utf-8")


def main():
    pooled = pd.read_csv(POOLED)
    regime = pd.read_csv(REGIME)
    pooled["regime"] = ALL_REGIME
    all_scores = pd.concat([pooled, regime], ignore_index=True, sort=False)
    all_scores = add_simple_dof(all_scores)

    summary = build_summary_table(all_scores)
    summary.to_csv(OUT_CSV, index=False)
    make_plot(summary)
    write_markdown(summary)

    best = summary.iloc[0]
    print(OUT_CSV)
    print(OUT_MD)
    print(OUT_PNG)
    print(
        f"Best formal model: {best['model_label']} "
        f"(chi2/(N-k)={best['formal_resolved_prior_chi2_over_dof']:.2f})"
    )


if __name__ == "__main__":
    main()
