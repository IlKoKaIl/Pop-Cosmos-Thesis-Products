"""Summarise source-to-source tension in the FIR model-family evaluator.

This is a thesis-facing diagnostic, not another model fit.

Question:
    Do all observed count papers prefer the exact same dust template?

Answer we want to show cleanly:
    No, the exact best template shifts between count products, but every
    source is better fit by a warmer/broader dust correction than by baseline
    FSPS.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"

PER_SOURCE = OUT_DIR / "popcosmos_model_family_per_source_scores.csv"
OUT_SUMMARY = OUT_DIR / "popcosmos_model_family_source_tension_summary.csv"
OUT_MATRIX = OUT_DIR / "popcosmos_model_family_key_model_matrix.csv"
OUT_PLOT = OUT_DIR / "popcosmos_model_family_source_tension.png"

KEY_MODELS = [
    "FSPS",
    "25% ALESS",
    "50% ALESS",
    "MBB 35 K",
    "Casey T30K a=2.5",
    "Casey T30K a=3.0",
    "ALESS",
]

SOURCE_ORDER = [
    "Clements et al. / Table 1",
    "Oliver et al. / Table 2",
    "Pearson et al. / Table 3 SUSSEXtractor",
    "Pearson et al. / Table 4 XID",
    "Glenn et al. / Table 4 P(D) spline no FIRAS",
    "Varnish et al. / Table 4 P(D) best-fit spline",
]

SHORT_SOURCE = {
    "Clements et al. / Table 1": "Clements",
    "Oliver et al. / Table 2": "Oliver",
    "Pearson et al. / Table 3 SUSSEXtractor": "Pearson SUSSEX",
    "Pearson et al. / Table 4 XID": "Pearson XID",
    "Glenn et al. / Table 4 P(D) spline no FIRAS": "Glenn P(D)",
    "Varnish et al. / Table 4 P(D) best-fit spline": "Varnish P(D)",
}


def source_regime(source):
    return "P(D)" if "P(D)" in source else "resolved/prior"


def load_scores():
    df = pd.read_csv(PER_SOURCE)
    df["reduced_chi2_log"] = pd.to_numeric(df["reduced_chi2_log"], errors="coerce")
    df["median_log10_model_over_obs"] = pd.to_numeric(
        df["median_log10_model_over_obs"], errors="coerce"
    )
    df["N_points"] = pd.to_numeric(df["N_points"], errors="coerce")
    df = df.dropna(subset=["external_source", "model_label", "reduced_chi2_log"])
    return df


def make_summary(df):
    rows = []
    for source, group in df.groupby("external_source"):
        group = group.sort_values("reduced_chi2_log")
        best = group.iloc[0]
        fsps = group[group["model_label"] == "FSPS"]
        casey25 = group[group["model_label"] == "Casey T30K a=2.5"]
        casey30 = group[group["model_label"] == "Casey T30K a=3.0"]

        fsps_chi2 = float(fsps["reduced_chi2_log"].iloc[0]) if not fsps.empty else np.nan
        casey25_chi2 = (
            float(casey25["reduced_chi2_log"].iloc[0]) if not casey25.empty else np.nan
        )
        casey30_chi2 = (
            float(casey30["reduced_chi2_log"].iloc[0]) if not casey30.empty else np.nan
        )

        rows.append(
            {
                "external_source": source,
                "source_short": SHORT_SOURCE.get(source, source),
                "count_regime": source_regime(source),
                "best_model_family": best["model_family"],
                "best_model_label": best["model_label"],
                "best_reduced_chi2": float(best["reduced_chi2_log"]),
                "best_median_log10_model_over_obs": float(
                    best["median_log10_model_over_obs"]
                ),
                "fsps_reduced_chi2": fsps_chi2,
                "fsps_over_best_chi2_ratio": fsps_chi2 / float(best["reduced_chi2_log"])
                if np.isfinite(fsps_chi2) and best["reduced_chi2_log"] > 0
                else np.nan,
                "casey_T30_a25_reduced_chi2": casey25_chi2,
                "casey_T30_a30_reduced_chi2": casey30_chi2,
                "N_points_best": int(best["N_points"]),
            }
        )

    out = pd.DataFrame(rows)
    out["source_rank"] = out["external_source"].map(
        {source: i for i, source in enumerate(SOURCE_ORDER)}
    )
    return out.sort_values("source_rank").drop(columns=["source_rank"])


def make_key_model_matrix(df):
    matrix = df[df["model_label"].isin(KEY_MODELS)].copy()
    matrix["source_short"] = matrix["external_source"].map(SHORT_SOURCE)
    pivot = matrix.pivot_table(
        index="source_short",
        columns="model_label",
        values="reduced_chi2_log",
        aggfunc="first",
    )
    source_short_order = [SHORT_SOURCE[s] for s in SOURCE_ORDER if SHORT_SOURCE[s] in pivot.index]
    model_order = [m for m in KEY_MODELS if m in pivot.columns]
    pivot = pivot.loc[source_short_order, model_order]
    return pivot


def plot_summary(summary, matrix):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.45, 1]})

    log_matrix = np.log10(matrix.to_numpy(dtype=float))
    im = axes[0].imshow(log_matrix, aspect="auto", cmap="viridis_r")
    axes[0].set_xticks(np.arange(matrix.shape[1]))
    axes[0].set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    axes[0].set_yticks(np.arange(matrix.shape[0]))
    axes[0].set_yticklabels(matrix.index, fontsize=9)
    axes[0].set_title(r"Key models by observed count source, log10 reduced $\chi^2$")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                axes[0].text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label(r"log10 rough reduced $\chi^2$")

    x = np.arange(len(summary))
    colors = summary["count_regime"].map({"resolved/prior": "#0072B2", "P(D)": "#D55E00"})
    axes[1].bar(x, summary["fsps_over_best_chi2_ratio"], color=colors)
    axes[1].axhline(1.0, color="0.2", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["source_short"], rotation=25, ha="right")
    axes[1].set_ylabel(r"FSPS reduced $\chi^2$ / best reduced $\chi^2$")
    axes[1].set_title("How much worse is baseline FSPS?")
    axes[1].set_ylim(0, summary["fsps_over_best_chi2_ratio"].max() * 1.35)
    axes[1].grid(True, axis="y", alpha=0.25)
    for xi, row in summary.reset_index(drop=True).iterrows():
        axes[1].text(
            xi,
            row["fsps_over_best_chi2_ratio"] + 0.08,
            row["best_model_label"],
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )

    fig.suptitle("Observed SPIRE count sources prefer the same broad correction, not the same exact template")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT_PLOT, dpi=180)
    plt.close(fig)
    return OUT_PLOT


def main():
    df = load_scores()
    summary = make_summary(df)
    matrix = make_key_model_matrix(df)

    summary.to_csv(OUT_SUMMARY, index=False)
    matrix.to_csv(OUT_MATRIX)
    plot_path = plot_summary(summary, matrix)

    print(OUT_SUMMARY)
    print(OUT_MATRIX)
    print(plot_path)
    print("\nBest model by external source:")
    print(
        summary[
            [
                "source_short",
                "count_regime",
                "best_model_label",
                "best_reduced_chi2",
                "fsps_reduced_chi2",
                "fsps_over_best_chi2_ratio",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
