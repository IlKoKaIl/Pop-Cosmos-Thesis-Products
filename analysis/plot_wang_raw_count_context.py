"""Plot raw Wang cumulative counts next to directly published integral counts.

This is a diagnostic, not part of the formal evaluator.

The formal evaluator uses published differential counts because those bins are
closer to independent. Here the goal is simpler: show why raw Wang `master.dat`
counts should be treated carefully when compared with corrected count papers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
PROJECT_CODING = NB_DIR.parents[1]
OUT_DIR = ROOT / "outputs"

WANG_SCENARIOS = OUT_DIR / "wang_master_catalog_count_scenarios.csv"
EXTERNAL_STARTER = PROJECT_CODING / "catalog data/external_number_counts/external_spire_number_counts_starter.csv"

OUT_PNG = OUT_DIR / "wang_raw_count_context.png"
OUT_CSV = OUT_DIR / "wang_raw_count_context_area_selection_summary.csv"
OUT_MD = OUT_DIR / "wang_raw_count_context_note.md"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_wang_scenarios() -> pd.DataFrame:
    require_file(WANG_SCENARIOS)
    df = pd.read_csv(WANG_SCENARIOS)
    return df[df["band_um"].isin([250, 350, 500])].copy()


def load_direct_integral_counts() -> pd.DataFrame:
    require_file(EXTERNAL_STARTER)
    df = pd.read_csv(EXTERNAL_STARTER)
    has_integral = df["integral_N_gt_S_per_deg2"].notna()
    useful_sources = df["paper"].isin(["Clements et al.", "Pearson et al."])
    out = df[has_integral & useful_sources & df["band_um"].isin([250, 350, 500])].copy()
    for col in ["flux_mjy", "integral_N_gt_S_per_deg2", "integral_err_per_deg2"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["flux_mjy", "integral_N_gt_S_per_deg2"])


def build_area_selection_summary(wang: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cuts = [5, 10, 20, 50, 100]
    for band in [250, 350, 500]:
        for cut in cuts:
            sub = wang[(wang["band_um"] == band) & (wang["flux_cut_mjy"] == cut)]

            def pick(population, area_name, column):
                hit = sub[(sub["population"] == population) & (sub["area_name"] == area_name)]
                if hit.empty:
                    return np.nan
                return float(hit.iloc[0][column])

            positive_1p278 = pick(
                "positive_cosmos2020_ids_only",
                "wang_farmer_flag_combined0",
                "N_positive_per_deg2",
            )
            positive_snr3_1p278 = pick(
                "positive_cosmos2020_ids_only",
                "wang_farmer_flag_combined0",
                "N_snr3_per_deg2",
            )
            positive_2deg = pick(
                "positive_cosmos2020_ids_only",
                "old_full_cosmos_approx",
                "N_positive_per_deg2",
            )
            all_1p278 = pick(
                "all_wang_prior_rows",
                "wang_farmer_flag_combined0",
                "N_positive_per_deg2",
            )
            rows.append(
                {
                    "band_um": band,
                    "flux_cut_mjy": cut,
                    "positive_ids_1p278deg2_per_deg2": positive_1p278,
                    "positive_ids_snr3_1p278deg2_per_deg2": positive_snr3_1p278,
                    "positive_ids_2deg2_per_deg2": positive_2deg,
                    "all_prior_rows_1p278deg2_per_deg2": all_1p278,
                    "area_choice_factor_1p278_vs_2deg": positive_1p278 / positive_2deg
                    if positive_2deg > 0
                    else np.nan,
                    "snr3_fraction_of_positive_count_density": positive_snr3_1p278 / positive_1p278
                    if positive_1p278 > 0
                    else np.nan,
                    "all_prior_vs_positive_factor": all_1p278 / positive_1p278
                    if positive_1p278 > 0
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    return out


def plot_context(wang: pd.DataFrame, external: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    fig.suptitle("Wang raw cumulative counts are a diagnostic, not the formal count product", y=1.02)

    for ax, band in zip(axes, [250, 350, 500]):
        w = wang[(wang["band_um"] == band)]
        pos = w[
            (w["population"] == "positive_cosmos2020_ids_only")
            & (w["area_name"] == "wang_farmer_flag_combined0")
        ].sort_values("flux_cut_mjy")
        old_area = w[
            (w["population"] == "positive_cosmos2020_ids_only")
            & (w["area_name"] == "old_full_cosmos_approx")
        ].sort_values("flux_cut_mjy")
        all_rows = w[
            (w["population"] == "all_wang_prior_rows")
            & (w["area_name"] == "wang_farmer_flag_combined0")
        ].sort_values("flux_cut_mjy")

        ax.plot(
            pos["flux_cut_mjy"],
            pos["N_positive_per_deg2"],
            marker="o",
            color="black",
            lw=2,
            label="Wang positive IDs, 1.278 deg2",
        )
        ax.plot(
            pos["flux_cut_mjy"],
            pos["N_snr3_per_deg2"],
            marker="s",
            color="black",
            lw=1.5,
            ls="--",
            label="Wang positive IDs, SNR>=3",
        )
        ax.plot(
            old_area["flux_cut_mjy"],
            old_area["N_positive_per_deg2"],
            marker=".",
            color="0.55",
            lw=1.3,
            ls=":",
            label="same rows / 2 deg2",
        )
        ax.plot(
            all_rows["flux_cut_mjy"],
            all_rows["N_positive_per_deg2"],
            marker="^",
            color="#cc6677",
            lw=1.3,
            ls="-.",
            label="all Wang prior rows",
        )

        e = external[external["band_um"] == band]
        for (paper, method), group in e.groupby(["paper", "method_or_table"]):
            group = group.sort_values("flux_mjy")
            marker = "o" if paper.startswith("Clements") else "^"
            color = "#1b9e77" if paper.startswith("Clements") else "#7570b3"
            if paper.startswith("Clements"):
                label = "Clements"
            elif "SUSSEX" in method:
                label = "Pearson Table 3 SUSSEX"
            elif "XID" in method:
                label = "Pearson Table 4 XID"
            else:
                label = f"{paper} {method}"
            yerr = group["integral_err_per_deg2"].to_numpy(float)
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            ax.errorbar(
                group["flux_mjy"],
                group["integral_N_gt_S_per_deg2"],
                yerr=yerr,
                fmt=marker,
                ms=4,
                color=color,
                alpha=0.75,
                capsize=2,
                label=label,
            )

        ax.axvline(10, color="0.25", ls="--", lw=1, alpha=0.5)
        ax.text(
            10.8,
            0.075,
            "10 mJy",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="0.25",
        )
        ax.set_title(f"{band} um")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("flux cut S [mJy]")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"cumulative counts $N(>S)$ [deg$^{-2}$]")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[-1].legend(unique.values(), unique.keys(), fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_note(summary: pd.DataFrame) -> None:
    display = summary[
        (summary["flux_cut_mjy"].isin([10, 20, 50]))
        & (summary["band_um"].isin([250, 350, 500]))
    ].copy()
    for col in [
        "positive_ids_1p278deg2_per_deg2",
        "positive_ids_snr3_1p278deg2_per_deg2",
        "area_choice_factor_1p278_vs_2deg",
        "snr3_fraction_of_positive_count_density",
        "all_prior_vs_positive_factor",
    ]:
        display[col] = display[col].round(3)

    short_cols = [
        "band_um",
        "flux_cut_mjy",
        "positive_ids_1p278deg2_per_deg2",
        "positive_ids_snr3_1p278deg2_per_deg2",
        "area_choice_factor_1p278_vs_2deg",
        "snr3_fraction_of_positive_count_density",
        "all_prior_vs_positive_factor",
    ]
    display = display[short_cols]

    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    table_lines = [header, sep]
    for row in display.itertuples(index=False):
        table_lines.append("| " + " | ".join(str(value) for value in row) + " |")

    lines = [
        "# Wang Raw Count Context",
        "",
        "This is a diagnostic note, not the formal evaluator.",
        "",
        "The formal evaluator should still use corrected published differential counts.",
        "",
        "## What I Plotted",
        "",
        "- raw cumulative Wang counts from `master.dat`",
        "- positive COSMOS2020 IDs with the Wang `1.278 deg2` area",
        "- the same positive IDs with the old rough `2 deg2` area, just to show the area effect",
        "- all Wang prior rows, including negative radio-prior IDs",
        "- direct published integral-count points from Clements and Pearson where the tables give `N(>S)`",
        "",
        "## Simple Read",
        "",
        "- the Wang units are not the issue: the catalogue fluxes are in `mJy`",
        "- using `1.278 deg2` instead of `2 deg2` moves raw count densities by a factor of about `1.56`",
        "- SNR>=3 barely changes the bright counts above about `20-50 mJy`, but matters more near the faint end",
        "- including negative radio-prior rows can noticeably lift the raw Wang curve",
        "- this is enough to explain why Wang should be treated as diagnostic rather than as a corrected count paper",
        "",
        "## Quick Numbers",
        "",
        "\n".join(table_lines),
        "",
        "## Thesis Usage",
        "",
        "> Wang is best used for matched-object residuals. Corrected Clements / Oliver / Pearson differential counts should carry the formal population evaluator.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wang = load_wang_scenarios()
    external = load_direct_integral_counts()
    summary = build_area_selection_summary(wang)
    plot_context(wang, external)
    write_note(summary)
    print(OUT_PNG)
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
