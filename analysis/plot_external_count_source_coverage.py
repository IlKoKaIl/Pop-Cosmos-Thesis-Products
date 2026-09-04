"""Plot flux coverage of the external SPIRE count sources.

This is not part of the model evaluator. It is a planning/helper plot for the
thesis: it shows which observed count source constrains which flux range.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
COUNT_PATH = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
OUT_DIR = NB_DIR / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def source_label(row):
    paper = row["paper"].replace(" et al.", "")
    method = row["method_or_table"]
    if "SUSSEX" in method:
        return "Pearson SUSSEX"
    if "XID" in method:
        return "Pearson XID"
    if "P(D)" in method:
        return f"{paper} P(D)"
    return paper


def source_role(label):
    if "P(D)" in label:
        return "P(D) statistical / faint-end sensitivity"
    if "Pearson" in label:
        return "deep resolved/prior extraction"
    if "Clements" in label:
        return "wide bright-end resolved counts"
    if "Oliver" in label:
        return "HerMES resolved-count bridge"
    return "external count source"


def main():
    df = pd.read_csv(COUNT_PATH)
    df["source_label"] = df.apply(source_label, axis=1)

    summary = (
        df.groupby(["source_label", "paper", "method_or_table", "band_um"], as_index=False)
        .agg(
            n_points=("flux_mjy", "size"),
            min_flux_mjy=("flux_mjy", "min"),
            max_flux_mjy=("flux_mjy", "max"),
        )
        .sort_values(["band_um", "min_flux_mjy", "source_label"])
    )
    summary["role"] = summary["source_label"].map(source_role)

    summary_path = OUT_DIR / "external_count_source_flux_coverage.csv"
    plot_path = OUT_DIR / "external_count_source_flux_coverage.png"
    summary.to_csv(summary_path, index=False)

    labels = [
        "Varnish P(D)",
        "Glenn P(D)",
        "Pearson XID",
        "Pearson SUSSEX",
        "Oliver",
        "Clements",
    ]
    colors = {250: "#0072B2", 350: "#D55E00", 500: "#009E73"}
    offsets = {250: -0.18, 350: 0.0, 500: 0.18}

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, label in enumerate(labels):
        sub = summary[summary["source_label"] == label]
        for row in sub.itertuples(index=False):
            y = i + offsets[int(row.band_um)]
            ax.hlines(
                y,
                row.min_flux_mjy,
                row.max_flux_mjy,
                color=colors[int(row.band_um)],
                linewidth=4,
                alpha=0.9,
            )
            ax.plot(
                [row.min_flux_mjy, row.max_flux_mjy],
                [y, y],
                "o",
                color=colors[int(row.band_um)],
                markersize=4,
            )
            ax.text(
                row.max_flux_mjy * 1.04,
                y,
                f"{int(row.band_um)}",
                va="center",
                fontsize=8,
                color=colors[int(row.band_um)],
            )

    ax.set_xscale("log")
    ax.set_xlabel("Flux density covered by published count points [mJy]")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("External SPIRE Count-Source Flux Coverage")
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.set_xlim(0.01, 1600)

    handles = [
        plt.Line2D([0], [0], color=colors[band], linewidth=4, label=f"{band} um")
        for band in [250, 350, 500]
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(summary_path)
    print(plot_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
