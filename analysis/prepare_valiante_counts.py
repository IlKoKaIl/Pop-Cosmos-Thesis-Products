"""Prepare Valiante et al. 2016 H-ATLAS DR1 number counts for quick use.

The hand-entered source file keeps GAMA9, GAMA12, and GAMA15 separately.
For quick plots/evaluator experiments, this script also makes one area-weighted
H-ATLAS DR1 average curve per SPIRE band.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = SCRIPT_DIR.parent
PROJECT_CODING = SCRIPT_DIR.parents[1]
COUNTS_DIR = PROJECT_CODING / "catalog data" / "external_number_counts"
OUT_DIR = NOTEBOOK_DIR / "outputs"

IN_CSV = COUNTS_DIR / "valiante_2016_hatlas_dr1_number_counts.csv"
OUT_AVG_CSV = COUNTS_DIR / "valiante_2016_hatlas_dr1_number_counts_area_weighted.csv"
OUT_QC_CSV = COUNTS_DIR / "valiante_2016_hatlas_dr1_number_counts_qc_summary.csv"
OUT_PNG = OUT_DIR / "valiante_2016_hatlas_dr1_number_counts_quicklook.png"

FIELD_AREAS_DEG2 = {
    "GAMA9": 53.43,
    "GAMA12": 53.56,
    "GAMA15": 54.56,
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def area_weighted_average(group: pd.DataFrame) -> pd.Series:
    weights = group["field"].map(FIELD_AREAS_DEG2).to_numpy(float)
    values = group["s25_dnds_jy15_deg2"].to_numpy(float)
    errors = group["s25_dnds_err_jy15_deg2"].to_numpy(float)

    total_area = float(np.sum(weights))
    norm_weights = weights / total_area
    weighted_mean = float(np.sum(norm_weights * values))

    # Propagate quoted per-field errors as if fields were independent.
    # Keep field scatter separately because the paper notes flux-bin correlations.
    propagated_err = float(np.sqrt(np.sum((norm_weights * errors) ** 2)))
    field_scatter = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan

    return pd.Series(
        {
            "paper": "Valiante et al.",
            "year": 2016,
            "survey": "H-ATLAS DR1",
            "method_or_table": f"{group['table'].iloc[0]} area-weighted GAMA9/12/15",
            "count_type": "resolved/prior",
            "band_um": int(group["band_um"].iloc[0]),
            "flux_mjy": float(group["flux_mjy"].iloc[0]),
            "flux_jy": float(group["flux_mjy"].iloc[0]) / 1000.0,
            "euclidean_best_jy15_deg2": weighted_mean,
            "euclidean_err_jy15_deg2": propagated_err,
            "field_scatter_jy15_deg2": field_scatter,
            "n_fields": int(len(group)),
            "area_deg2": total_area,
            "standard_unit": "Jy^1.5 deg^-2",
            "notes": (
                "Area-weighted average of GAMA9/GAMA12/GAMA15 from Valiante "
                "Tables 5/8/9. Quoted errors do not include flux-bin correlations; "
                "field_scatter is kept as a sanity check."
            ),
            "source_url": "https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.3146V/abstract",
        }
    )


def make_quicklook(raw: pd.DataFrame, avg: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    colors = {"GAMA9": "#0072B2", "GAMA12": "#009E73", "GAMA15": "#D55E00"}

    for ax, band in zip(axes, [250, 350, 500]):
        sub = raw[raw["band_um"] == band]
        for field, group in sub.groupby("field"):
            group = group.sort_values("flux_mjy")
            ax.errorbar(
                group["flux_mjy"],
                group["s25_dnds_jy15_deg2"],
                yerr=group["s25_dnds_err_jy15_deg2"],
                marker="o",
                ms=3,
                lw=1,
                color=colors.get(field, "0.5"),
                alpha=0.65,
                label=field,
            )

        mean = avg[avg["band_um"] == band].sort_values("flux_mjy")
        ax.plot(
            mean["flux_mjy"],
            mean["euclidean_best_jy15_deg2"],
            color="black",
            lw=2.2,
            marker="s",
            ms=3,
            label="area-weighted mean",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{band} um")
        ax.set_xlabel("Flux density S [mJy]")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"$S^{2.5} dN/dS$ [Jy$^{1.5}$ deg$^{-2}$]")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Valiante et al. 2016 H-ATLAS DR1 number counts")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    require_file(IN_CSV)
    raw = pd.read_csv(IN_CSV)
    for col in ["band_um", "flux_mjy", "s25_dnds_jy15_deg2", "s25_dnds_err_jy15_deg2"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    qc = (
        raw.groupby(["table", "band_um", "field"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["band_um", "field"])
    )
    qc.to_csv(OUT_QC_CSV, index=False)

    avg = (
        raw.groupby(["band_um", "flux_mjy"], as_index=False, group_keys=False)
        .apply(area_weighted_average)
        .reset_index(drop=True)
        .sort_values(["band_um", "flux_mjy"])
    )
    avg.to_csv(OUT_AVG_CSV, index=False)
    make_quicklook(raw, avg)

    print(OUT_AVG_CSV)
    print(OUT_QC_CSV)
    print(OUT_PNG)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
