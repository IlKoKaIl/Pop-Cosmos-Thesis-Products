"""Rebuild the Valiante bright-end diagnostic used in the thesis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
COUNTS_PATH = (
    NOTEBOOK_DIR.parent
    / "catalog data"
    / "external_number_counts"
    / "external_spire_differential_counts_compiled.csv"
)
OUTPUT_PATH = (
    NOTEBOOK_DIR
    / "fir_validation_aug2026"
    / "figures"
    / "lensing_bright_end_assessment.png"
)

COLORS = {250: "#0072B2", 350: "#009E73", 500: "#D55E00"}


def main() -> None:
    counts = pd.read_csv(COUNTS_PATH)
    valiante = counts.loc[counts["paper"].eq("Valiante et al.")].copy()

    fig, (ax_counts, ax_ratio) = plt.subplots(1, 2, figsize=(14, 5.7))
    bands = [250, 350, 500]
    ratios = []
    ratio_errors = []

    for band in bands:
        band_data = valiante.loc[valiante["band_um"].eq(band)].sort_values("flux_mjy")
        flux = band_data["flux_mjy"].to_numpy(float)
        value = band_data["euclidean_best_jy15_deg2"].to_numpy(float)
        error = band_data["euclidean_err_jy15_deg2"].to_numpy(float)

        ax_counts.errorbar(
            flux,
            value,
            yerr=error,
            marker="o",
            markersize=4.5,
            linewidth=1.7,
            capsize=2.5,
            color=COLORS[band],
            label=rf"{band} $\mu$m",
        )

        previous = band_data.loc[np.isclose(band_data["flux_mjy"], 244.2)].iloc[0]
        final = band_data.loc[np.isclose(band_data["flux_mjy"], 300.0)].iloc[0]
        ratio = final["euclidean_best_jy15_deg2"] / previous["euclidean_best_jy15_deg2"]
        ratio_error = ratio * np.hypot(
            final["euclidean_err_jy15_deg2"] / final["euclidean_best_jy15_deg2"],
            previous["euclidean_err_jy15_deg2"] / previous["euclidean_best_jy15_deg2"],
        )
        ratios.append(ratio)
        ratio_errors.append(ratio_error)

    ax_counts.axvspan(80, 150, color="0.85", alpha=0.55, zorder=0)
    ax_counts.text(
        108,
        5.5,
        "strong lensing becomes relevant\nin bright wide-field samples",
        ha="center",
        va="center",
        fontsize=9,
        color="0.3",
    )
    ax_counts.annotate(
        "300 mJy bin",
        xy=(300, 1.756),
        xytext=(205, 0.23),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )
    ax_counts.set_xscale("log")
    ax_counts.set_yscale("log")
    ax_counts.set_xlabel(r"Flux density $S$ [mJy]")
    ax_counts.set_ylabel(r"$S^{2.5}\,\mathrm{d}N/\mathrm{d}S$ [Jy$^{1.5}$ deg$^{-2}$]")
    ax_counts.set_title("H-ATLAS DR1 published bright counts")
    ax_counts.legend(frameon=False)
    ax_counts.grid(alpha=0.18, which="both")

    x = np.arange(len(bands))
    for position, band, ratio, error in zip(x, bands, ratios, ratio_errors):
        ax_ratio.errorbar(
            position,
            ratio,
            yerr=error,
            fmt="o",
            markersize=10,
            capsize=5,
            linewidth=2,
            color=COLORS[band],
        )
        ax_ratio.text(position + 0.12, ratio, rf"{ratio:.2f} $\pm$ {error:.2f}", va="center")

    band_difference_sigma = abs(ratios[0] - ratios[-1]) / np.hypot(
        ratio_errors[0], ratio_errors[-1]
    )
    ax_ratio.axhline(1, color="0.3", linestyle=":", linewidth=1.3)
    ax_ratio.text(-0.32, 1.03, "no uptick", color="0.35", fontsize=9)
    ax_ratio.set_xticks(x, [rf"{band} $\mu$m" for band in bands])
    ax_ratio.set_ylim(0.75, 3.55)
    ax_ratio.set_ylabel("300 mJy count / 244 mJy count")
    ax_ratio.set_title(
        f"No measurable band-to-band trend (250 vs 500: {band_difference_sigma:.1f}$\sigma$)"
    )
    ax_ratio.grid(alpha=0.18, axis="y")

    fig.suptitle("Valiante bright-end uptick and the bright-flux lensing regime", fontsize=16)
    fig.text(
        0.73,
        0.02,
        "Ratio errors propagate quoted per-bin errors; bin covariance is unavailable.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
