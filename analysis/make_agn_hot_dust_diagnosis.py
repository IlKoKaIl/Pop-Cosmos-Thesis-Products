"""Rebuild the thesis AGN/hot-dust diagnostic with an unobstructed legend."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "fir_validation_aug2026" / "tables"
FIGURE_DIR = ROOT / "fir_validation_aug2026" / "figures"


def main():
    sample = pd.read_csv(TABLE_DIR / "drew_casey_peak_comparison.csv")
    low_agn = sample.loc[sample["fAGN"] < 0.1, "peak_um"].dropna()
    high_agn = sample.loc[sample["fAGN"] >= 0.3, "peak_um"].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    fig.suptitle(
        "Short-wavelength SED peaks are associated with AGN-heated dust, "
        "and stacked SEDs conceal them",
        fontsize=13,
    )

    # Panel 1: the same saved 8,000-galaxy sample used by the thesis analysis.
    ax = axes[0]
    bins = np.geomspace(5, 500, 42)
    ax.hist(
        low_agn,
        bins=bins,
        color="#3f93c2",
        alpha=0.95,
        label=rf"low AGN ($f_{{AGN}}<0.1$, N={len(low_agn)})",
    )
    ax.hist(
        high_agn,
        bins=bins,
        color="#e2863a",
        alpha=0.95,
        label=rf"high AGN ($f_{{AGN}}\geq0.3$, N={len(high_agn)})",
    )
    ax.axvline(20, color="black", linestyle="--", linewidth=1.2, zorder=0)
    ax.annotate(
        "20 $\mu$m\n($\sim$145 K)",
        xy=(20, 1),
        xycoords=("data", "axes fraction"),
        xytext=(5, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.set_xscale("log")
    ax.set_xlim(5, 500)
    ax.set_title("Hot peaks are associated with high $f_{AGN}$", fontsize=11)
    ax.set_xlabel("rest-frame FIR peak wavelength [$\mu$m]")
    ax.set_ylabel("number of galaxies")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        fontsize=7.8,
        frameon=True,
        framealpha=1.0,
    )

    # Panel 2: rounded fractions retained from the original diagnostic summary.
    ax = axes[1]
    fagn_labels = ["<0.01", "0.01-0.05", "0.05-0.1", "0.1-0.3", "0.3-1", "$\geq$1"]
    hot_fractions = np.array([0.00, 0.00, 0.01, 0.13, 0.67, 1.00])
    bars = ax.bar(np.arange(len(fagn_labels)), hot_fractions, color="#d95f02", width=0.60)
    for bar, fraction in zip(bars, hot_fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            fraction + 0.018,
            f"{fraction:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 1.06)
    ax.set_xticks(np.arange(len(fagn_labels)), fagn_labels, rotation=28, ha="right")
    ax.set_title("Hot-peak fraction rises with the AGN parameter", fontsize=11)
    ax.set_xlabel("AGN luminosity ratio $f_{AGN}$")
    ax.set_ylabel("fraction with SED peak $<40\,\mu$m")

    # Panel 3: stored summary values from the original luminosity-bin stack test.
    ax = axes[2]
    lir_labels = ["9.0-9.5", "9.5-10.0", "10.0-10.5", "10.5-11.0", "11.0-13.0"]
    x = np.arange(len(lir_labels))
    stacked_peaks = np.full(len(x), 135.5)
    median_individual_peaks = np.array([135.5, 88.72, 88.72, 105.5, 88.72])
    hot_percent = [24, 36, 39, 40, 50]
    ax.plot(x, stacked_peaks, "s-", color="#087eb5", label="peak of the STACKED (mean) SED")
    ax.plot(
        x,
        median_individual_peaks,
        "o--",
        color="#d95f02",
        label="median of INDIVIDUAL galaxy peaks",
    )
    for xi, yi, percentage in zip(x, median_individual_peaks, hot_percent):
        ax.annotate(
            f"{percentage}% hot",
            (xi, yi),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            color="#d95f02",
            fontsize=8,
        )
    ax.set_xticks(x, lir_labels, rotation=20, ha="right")
    ax.set_ylim(60, 155)
    ax.set_title("Stacking hides the hot population", fontsize=11)
    ax.set_xlabel(r"$\log_{10}(L_{IR}/L_\odot)$ bin")
    ax.set_ylabel("rest-frame FIR peak [$\mu$m]")
    ax.legend(loc="center left", fontsize=8, frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.78, bottom=0.28, wspace=0.23)
    output = FIGURE_DIR / "agn_hot_dust_diagnosis.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
