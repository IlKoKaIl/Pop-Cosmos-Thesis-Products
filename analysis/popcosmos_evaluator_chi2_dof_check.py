"""Check how the FIR evaluator changes with a simple degrees-of-freedom correction.

The existing evaluator reports a rough score:

    reduced_chi2_log = chi2_log / N_points

That is useful as a ranking score, but in a formal reduced-chi-square sense we
would normally divide by:

    dof = N_points - N_fitted_parameters

This script keeps that distinction explicit for thesis/report writing. The
parameter counts below are deliberately simple and conservative. They are not a
claim that the grid search is a full likelihood fit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"

SCORE_SUMMARY = OUT_DIR / "popcosmos_model_family_score_summary.csv"
REGIME_SUMMARY = OUT_DIR / "popcosmos_model_family_regime_summary.csv"

OUT_CSV = OUT_DIR / "popcosmos_evaluator_chi2_dof_check.csv"
OUT_MD = OUT_DIR / "popcosmos_evaluator_chi2_dof_check.md"
OUT_PNG = OUT_DIR / "popcosmos_evaluator_chi2_dof_check.png"

TEMPLATE_PARAM_COUNTS = {
    "baseline": 0,
    "empirical template": 0,
    "FSPS/ALESS hybrid": 1,
    "modified blackbody": 1,
    "Casey-like template": 2,
}


def fmt(value, digits=2):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def add_dof_columns(df):
    out = df.copy()
    out["template_param_count_simple"] = out["model_family"].map(TEMPLATE_PARAM_COUNTS).fillna(0).astype(int)
    out["dof_simple"] = np.maximum(
        pd.to_numeric(out["N_points"], errors="coerce") - out["template_param_count_simple"],
        1,
    )
    out["reduced_chi2_log_simple_dof"] = pd.to_numeric(out["chi2_log"], errors="coerce") / out["dof_simple"]
    out["rank_original"] = out.groupby("regime")["reduced_chi2_log"].rank(method="min")
    out["rank_simple_dof"] = out.groupby("regime")["reduced_chi2_log_simple_dof"].rank(method="min")
    out["rank_shift_simple_dof_minus_original"] = out["rank_simple_dof"] - out["rank_original"]
    return out.sort_values(["regime", "reduced_chi2_log_simple_dof"])


def make_plot(pooled):
    key = pooled.head(10).copy()
    anchors = ["FSPS", "ALESS", "25% ALESS", "50% ALESS", "MBB 35 K", "Casey T30K a=2.5", "Casey T30K a=3.0"]
    key = pd.concat([key, pooled[pooled["model_label"].isin(anchors)]], ignore_index=True)
    key = key.drop_duplicates(subset=["model_label"])
    key = key.sort_values("reduced_chi2_log")

    x = np.arange(len(key))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar(
        x - width / 2,
        key["reduced_chi2_log"],
        width=width,
        label=r"score used so far: $\chi^2/N$",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        key["reduced_chi2_log_simple_dof"],
        width=width,
        label=r"simple dof check: $\chi^2/(N-k)$",
        color="#F28E2B",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"rough reduced $\chi^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(key["model_label"], rotation=35, ha="right")
    ax.set_title("Does subtracting simple template parameters change the ranking?")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def make_markdown(all_scores):
    pooled = all_scores[all_scores["regime"] == "all scored counts"].copy()
    pooled = pooled.sort_values("reduced_chi2_log_simple_dof")

    top_rows = []
    for _, row in pooled.head(8).iterrows():
        top_rows.append(
            {
                "model": row["model_label"],
                "family": row["model_family"],
                "N": int(row["N_points"]),
                "k": int(row["template_param_count_simple"]),
                "chi2/N": fmt(row["reduced_chi2_log"], 2),
                "chi2/(N-k)": fmt(row["reduced_chi2_log_simple_dof"], 2),
                "rank shift": fmt(row["rank_shift_simple_dof_minus_original"], 0),
            }
        )

    best_by_regime_rows = []
    for regime, group in all_scores.groupby("regime"):
        best = group.sort_values("reduced_chi2_log_simple_dof").iloc[0]
        best_by_regime_rows.append(
            {
                "regime": regime,
                "best model": best["model_label"],
                "N": int(best["N_points"]),
                "k": int(best["template_param_count_simple"]),
                "chi2/(N-k)": fmt(best["reduced_chi2_log_simple_dof"], 2),
            }
        )

    text = f"""# FIR evaluator chi-square / degrees-of-freedom check

This is a simple bookkeeping check for thesis wording.

The current evaluator score is:

```text
chi2 = sum((log10(model) - log10(observed))^2 / sigma_log^2)
rough reduced chi2 = chi2 / N_points
```

That is okay as a model-ranking score. If I want formal reduced chi-square wording, I should be more careful and use something like:

```text
reduced chi2 = chi2 / (N_points - k)
```

where `k` is the number of fitted template parameters.

Simple `k` used here:

- FSPS baseline: `0`
- pure ALESS fixed template: `0`
- FSPS/ALESS hybrid fraction: `1`
- modified blackbody temperature grid: `1`
- Casey-like template grid: `2` (`T_dust`, `alpha`)

Important caveat:

> These are diagnostic template grids, not a full likelihood fit. So I should call the current number a "rough reduced chi-square score" unless Boris/Dave want a stricter statistical treatment.

## Top Models With Simple Dof Correction

{markdown_table(top_rows, ["model", "family", "N", "k", "chi2/N", "chi2/(N-k)", "rank shift"])}

## Best Model By Regime With Simple Dof Correction

{markdown_table(best_by_regime_rows, ["regime", "best model", "N", "k", "chi2/(N-k)"])}

Main read:

- subtracting these simple template-parameter counts barely changes the result because `N_points` is large compared with `k`
- the broad best model family remains Casey-like around `30 K`
- this supports using the chi-square score as a ranking diagnostic, while being careful not to overstate it as a final formal likelihood

Overfitting wording:

> A high chi-square does not mean overfitting. It means the model is not matching the data within the adopted errors, or the errors/model assumptions are incomplete. Overfitting is tested by held-out count sources and by keeping physical guardrails on the template family.
"""
    OUT_MD.write_text(text, encoding="utf-8")


def main():
    pooled = pd.read_csv(SCORE_SUMMARY)
    regime = pd.read_csv(REGIME_SUMMARY)
    all_scores = pd.concat([pooled, regime], ignore_index=True, sort=False)
    all_scores = add_dof_columns(all_scores)
    all_scores.to_csv(OUT_CSV, index=False)
    make_plot(all_scores[all_scores["regime"] == "all scored counts"].copy())
    make_markdown(all_scores)

    print(OUT_CSV)
    print(OUT_MD)
    print(OUT_PNG)
    print("\nBest with simple dof correction:")
    for regime, group in all_scores.groupby("regime"):
        best = group.sort_values("reduced_chi2_log_simple_dof").iloc[0]
        print(
            f"- {regime}: {best['model_label']} "
            f"(chi2/(N-k)={best['reduced_chi2_log_simple_dof']:.2f}, "
            f"N={int(best['N_points'])}, k={int(best['template_param_count_simple'])})"
        )


if __name__ == "__main__":
    main()
