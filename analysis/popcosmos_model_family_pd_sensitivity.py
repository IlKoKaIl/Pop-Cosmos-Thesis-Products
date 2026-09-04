"""Summarise how much the preferred FIR model changes when P(D) counts are included.

This is a bookkeeping/meeting script. It does not recompute any model fluxes.

It reads the already generated model-family score summaries and makes a small
table/plot comparing:

- all published differential counts together
- resolved/prior extracted counts only
- P(D) statistical counts only

The point is to keep the thesis claim honest: if the headline model only wins
because of P(D), that should be obvious. If the broad result is stable, that is
useful evidence that the evaluator is not just chasing one count product.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"

POOLED = OUT_DIR / "popcosmos_model_family_score_summary.csv"
REGIME = OUT_DIR / "popcosmos_model_family_regime_summary.csv"
OUT_CSV = OUT_DIR / "popcosmos_model_family_pd_sensitivity.csv"
OUT_PNG = OUT_DIR / "popcosmos_model_family_pd_sensitivity.png"

REGIME_ORDER = [
    "all scored counts",
    "resolved/prior counts",
    "P(D) statistical counts",
]


def load_scores():
    pooled = pd.read_csv(POOLED).copy()
    pooled["regime"] = "all scored counts"

    regime = pd.read_csv(REGIME).copy()
    combined = pd.concat([pooled, regime], ignore_index=True, sort=False)
    combined = combined[combined["regime"].isin(REGIME_ORDER)].copy()
    combined["model_id"] = combined["model_family"] + " | " + combined["model_label"]
    return combined


def make_pivot(scores):
    rows = []
    for model_id, group in scores.groupby("model_id"):
        row = {
            "model_id": model_id,
            "model_family": group["model_family"].iloc[0],
            "model_label": group["model_label"].iloc[0],
        }
        for regime in REGIME_ORDER:
            sub = group[group["regime"] == regime]
            if sub.empty:
                row[f"{regime} reduced_chi2_log"] = np.nan
                row[f"{regime} median_log10_model_over_obs"] = np.nan
            else:
                best_row = sub.sort_values("reduced_chi2_log").iloc[0]
                row[f"{regime} reduced_chi2_log"] = best_row["reduced_chi2_log"]
                row[f"{regime} median_log10_model_over_obs"] = best_row[
                    "median_log10_model_over_obs"
                ]
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("all scored counts reduced_chi2_log")
    return out


def plot_sensitivity(pivot):
    key = pivot.head(10).copy()

    # Always include the current anchor models, even if they are not in the top 10.
    anchors = ["FSPS", "25% ALESS", "50% ALESS", "ALESS", "MBB 35 K", "Casey T30K a=2.5", "Casey T30K a=3.0"]
    key = pd.concat([key, pivot[pivot["model_label"].isin(anchors)]], ignore_index=True)
    key = key.drop_duplicates(subset=["model_id"])
    key = key.sort_values("all scored counts reduced_chi2_log")

    x = np.arange(len(key))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = {
        "all scored counts": "#4C78A8",
        "resolved/prior counts": "#59A14F",
        "P(D) statistical counts": "#E15759",
    }

    for i, regime in enumerate(REGIME_ORDER):
        vals = key[f"{regime} reduced_chi2_log"].to_numpy(float)
        ax.bar(x + (i - 1) * width, vals, width=width, label=regime, color=colors[regime])

    ax.set_yscale("log")
    ax.set_ylabel(r"rough reduced $\chi^2$ in log-count space")
    ax.set_xticks(x)
    ax.set_xticklabels(key["model_label"], rotation=35, ha="right")
    ax.set_title("Does the preferred FIR model change when P(D) counts are included?")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def main():
    scores = load_scores()
    pivot = make_pivot(scores)
    pivot.to_csv(OUT_CSV, index=False)
    plot_sensitivity(pivot)

    print(OUT_CSV)
    print(OUT_PNG)
    print("\nBest model by regime:")
    for regime in REGIME_ORDER:
        best = scores[scores["regime"] == regime].sort_values("reduced_chi2_log").iloc[0]
        print(
            f"- {regime}: {best['model_label']} "
            f"(reduced chi2={best['reduced_chi2_log']:.2f}, "
            f"median log ratio={best['median_log10_model_over_obs']:.2f})"
        )


if __name__ == "__main__":
    main()
