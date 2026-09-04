from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = NB_DIR.parent
ROOT = NB_DIR.parents[1]
PROJECT_OUT = NOTEBOOK_DIR / "outputs"
NB_OUT = NOTEBOOK_DIR / "outputs"
DIFF_COUNTS_CSV = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
INTEGRAL_COUNTS_CSV = ROOT / "catalog data/external_number_counts/external_spire_number_counts_starter.csv"
PREDICTION_CACHE = NB_OUT / "popcosmos_full_sed_band_predictions.pkl"
SR_PER_DEG2 = 3282.806350011744
COSMOS_AREA_DEG2 = 1.278

PROJECT_OUT.mkdir(parents=True, exist_ok=True)
NB_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(NB_DIR))
import popcosmos_full_sed_250_counts as pc  # noqa: E402


STYLES = {
    ("Clements et al.", "Table 1"): {
        "color": "#0072B2",
        "marker": "s",
        "ls": "-",
        "label": "Clements H-ATLAS",
    },
    ("Oliver et al.", "Table 2"): {
        "color": "#009E73",
        "marker": "D",
        "ls": "-",
        "label": "Oliver HerMES",
    },
    ("Pearson et al.", "Table 3 SUSSEXtractor"): {
        "color": "#D55E00",
        "marker": "^",
        "ls": "-",
        "label": "Pearson SUSSEX",
    },
    ("Pearson et al.", "Table 4 XID"): {
        "color": "#E69F00",
        "marker": "v",
        "ls": "--",
        "label": "Pearson XID",
    },
    ("Valiante et al.", "Table 5 area-weighted GAMA9/12/15"): {
        "color": "#332288",
        "marker": "P",
        "ls": "-",
        "label": "Valiante H-ATLAS DR1",
    },
    ("Valiante et al.", "Table 8 area-weighted GAMA9/12/15"): {
        "color": "#332288",
        "marker": "P",
        "ls": "-",
        "label": "Valiante H-ATLAS DR1",
    },
    ("Valiante et al.", "Table 9 area-weighted GAMA9/12/15"): {
        "color": "#332288",
        "marker": "P",
        "ls": "-",
        "label": "Valiante H-ATLAS DR1",
    },
    ("Varnish et al.", "Table 4 P(D) best-fit spline"): {
        "color": "#CC79A7",
        "marker": "o",
        "ls": "-",
        "label": "Varnish P(D)",
    },
}


def cumulative_counts(values, grid):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.zeros_like(grid)
    values = np.sort(values)
    return values.size - np.searchsorted(values, grid, side="left")


def oliver_cumulative_from_diff(external, band):
    sub = external[(external["paper"] == "Oliver et al.") & (external["band_um"] == band)].copy()
    if sub.empty:
        return sub

    rows = []
    for _, row in sub.iterrows():
        smin = float(row["bin_min_mjy"])
        tail = sub[pd.to_numeric(sub["bin_min_mjy"]) >= smin]
        n_sr = 0.0
        err_sr2 = 0.0
        for _, tail_row in tail.iterrows():
            width_jy = (float(tail_row["bin_max_mjy"]) - float(tail_row["bin_min_mjy"])) / 1000.0
            n_sr += float(tail_row["differential_value"]) * width_jy
            err = pd.to_numeric(tail_row.get("differential_err"), errors="coerce")
            if np.isfinite(err):
                err_sr2 += (err * width_jy) ** 2
        rows.append(
            {
                "flux_mjy": smin,
                "integral_N_gt_S_per_deg2": n_sr / SR_PER_DEG2,
                "integral_err_per_deg2": np.sqrt(err_sr2) / SR_PER_DEG2 if err_sr2 > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_external_differential_counts():
    df = pd.read_csv(DIFF_COUNTS_CSV)
    df = df[df["euclidean_best_jy15_deg2"] > 0].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    for ax, band in zip(axes, [250, 350, 500]):
        sub = df[df["band_um"] == band]
        for (paper, method), group in sub.groupby(["paper", "method_or_table"]):
            group = group.sort_values("flux_mjy")
            style = STYLES.get((paper, method), {"marker": "o", "ls": "-", "label": f"{paper} {method}"})
            yerr = pd.to_numeric(group.get("euclidean_err_jy15_deg2"), errors="coerce")
            kwargs = {
                "marker": style["marker"],
                "ms": 4,
                "lw": 1.0,
                "ls": style["ls"],
                "color": style.get("color"),
                "label": style["label"],
            }
            if np.isfinite(yerr).any() and (yerr.fillna(0) > 0).any():
                ax.errorbar(
                    group["flux_mjy"],
                    group["euclidean_best_jy15_deg2"],
                    yerr=yerr,
                    capsize=2,
                    **kwargs,
                )
            else:
                ax.plot(group["flux_mjy"], group["euclidean_best_jy15_deg2"], **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5, 1000)
        ax.set_title(f"{band} um")
        ax.set_xlabel("Flux density S [mJy]")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"$S^{2.5} dN/dS$ [Jy$^{1.5}$ deg$^{-2}$]")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), fontsize=7.5, loc="best")
    fig.suptitle("Compiled external SPIRE differential counts")
    fig.tight_layout()
    out = PROJECT_OUT / "external_spire_differential_counts_compiled.png"
    checked_out = PROJECT_OUT / "external_spire_differential_counts_compiled_checked.png"
    fullrange_out = PROJECT_OUT / "external_spire_differential_counts_fullrange.png"
    july21_out = PROJECT_OUT / "external_spire_differential_counts_july21_3dex.png"
    fig.savefig(out, dpi=180)
    fig.savefig(checked_out, dpi=180)
    fig.savefig(fullrange_out, dpi=180)
    for ax in axes:
        ax.set_ylim(0.03, 30)
    fig.savefig(july21_out, dpi=180)
    plt.close(fig)
    return july21_out


def plot_corrected_cumulative_overlay():
    pred = pd.read_pickle(PREDICTION_CACHE)
    wang = pc.load_wang_bands()
    matched = pred.merge(wang, on="ID", how="inner")
    external = pd.read_csv(INTEGRAL_COUNTS_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    grid = np.logspace(np.log10(5.0), np.log10(1000.0), 180)

    for ax, band in zip(axes, [250, 350, 500]):
        obs_col = f"F{band}"
        fsps_col = f"F{band}_fsps_mjy"
        aless_col = f"F{band}_aless_mjy"

        wang_raw_flux = wang.loc[
            np.isfinite(wang[obs_col])
            & np.isfinite(wang[f"s_F{band}"])
            & (wang[f"s_F{band}"] > 0)
            & (wang[obs_col] > 0),
            obs_col,
        ]

        ax.plot(
            grid,
            cumulative_counts(matched[fsps_col], grid) / COSMOS_AREA_DEG2,
            label="pop-cosmos FSPS",
            color=pc.OKABE_ITO["blue"],
            lw=2,
        )
        ax.plot(
            grid,
            cumulative_counts(matched[aless_col], grid) / COSMOS_AREA_DEG2,
            label="pop-cosmos + ALESS",
            color=pc.OKABE_ITO["orange"],
            lw=2,
            ls="--",
        )
        ax.plot(
            grid,
            cumulative_counts(wang_raw_flux, grid) / COSMOS_AREA_DEG2,
            label="Wang raw",
            color=pc.OKABE_ITO["black"],
            lw=2,
        )

        clements = external[(external["paper"] == "Clements et al.") & (external["band_um"] == band)]
        if not clements.empty:
            ax.errorbar(
                pd.to_numeric(clements["flux_mjy"]),
                pd.to_numeric(clements["integral_N_gt_S_per_deg2"]),
                yerr=pd.to_numeric(clements["integral_err_per_deg2"]),
                fmt="s",
                ms=5,
                color=pc.OKABE_ITO["green"],
                mec="white",
                mew=0.5,
                lw=1,
                alpha=0.9,
                label="Clements H-ATLAS",
            )

        for method, marker, color, label in [
            ("Table 3 SUSSEXtractor", "^", pc.OKABE_ITO["purple"], "Pearson SUSSEX"),
            ("Table 4 XID", "v", pc.OKABE_ITO["vermillion"], "Pearson XID"),
        ]:
            pearson = external[
                (external["paper"] == "Pearson et al.")
                & (external["method_or_table"] == method)
                & (external["band_um"] == band)
            ]
            if not pearson.empty:
                ax.errorbar(
                    pd.to_numeric(pearson["flux_mjy"]),
                    pd.to_numeric(pearson["integral_N_gt_S_per_deg2"]),
                    yerr=pd.to_numeric(pearson["integral_err_per_deg2"]),
                    fmt=marker,
                    ms=5,
                    color=color,
                    mec="white",
                    mew=0.5,
                    lw=1,
                    alpha=0.85,
                    label=label,
                )

        oliver = oliver_cumulative_from_diff(external, band)
        if not oliver.empty:
            ax.errorbar(
                oliver["flux_mjy"],
                oliver["integral_N_gt_S_per_deg2"],
                yerr=oliver["integral_err_per_deg2"],
                fmt="D",
                ms=4,
                color=pc.OKABE_ITO["yellow"],
                mec="black",
                mew=0.3,
                lw=1,
                alpha=0.9,
                label="Oliver approx from dN/dS",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{band}um")
        ax.set_xlabel("observed flux cut (mJy)")
        ax.grid(alpha=0.25, which="both")

    axes[0].set_ylabel(r"cumulative counts $N(>S)$ per deg$^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=7.5)
    fig.suptitle(
        "pop-cosmos/Wang cumulative counts with corrected external SPIRE counts\n"
        "Wang/model curves use raw positive fluxes and area=1.278 deg^2"
    )
    fig.tight_layout()
    out = NB_OUT / "popcosmos_full_sed_external_counts_overlay_corrected.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    print(plot_external_differential_counts())
    try:
        print(plot_corrected_cumulative_overlay())
    except FileNotFoundError as exc:
        print(f"Skipped cumulative Wang overlay because an input file was missing: {exc}")


if __name__ == "__main__":
    main()
