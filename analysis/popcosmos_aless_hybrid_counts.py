"""ALESS-template and hybrid FIR count checks for pop-cosmos.

This script uses the cached FSPS band predictions from
`popcosmos_full_sed_250_counts.py` and asks a simple question:

If pop-cosmos keeps the same objects and the same L_IR, but the FIR SED shape
is moved towards ALESS, do the 250/350/500 um counts move in the right
direction?
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
OUT_DIR = NB_DIR / "outputs"
PROJECT_OUT = ROOT / "outputs"
PREDICTION_CACHE = OUT_DIR / "popcosmos_full_sed_band_predictions.pkl"
EXTERNAL_DIFF_COUNTS = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
FULL_COSMOS_APPROX_AREA_DEG2 = 2.0
WANG_FARMER_AREA_DEG2 = 1.278
COSMOS_AREA_DEG2 = WANG_FARMER_AREA_DEG2
BANDS_UM = [250, 350, 500]

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(NB_DIR))
import popcosmos_full_sed_250_counts as pc  # noqa: E402


def add_aless_variants_and_hybrids(pred):
    """Overwrite ALESS variant and hybrid flux columns using one clear method."""
    template = pc.load_aless_template(pc.ALESS_PATH)
    fcols = {
        "aless": "fnu_average_mjy",
        "aless_bright": "fnu_bright_mjy",
        "aless_faint": "fnu_faint_mjy",
    }
    z = pred["z_pop"].to_numpy(float)
    lir = pred["L_IR"].to_numpy(float)

    for band in pc.OBS_BANDS_UM:
        for label, fcol in fcols.items():
            pred[f"F{band}_{label}_mjy"] = pc.predict_aless_flux_mjy(
                template,
                z,
                lir,
                lambda_obs_um=band,
                fcol=fcol,
            )

        fsps = pred[f"F{band}_fsps_mjy"].to_numpy(float)
        aless = pred[f"F{band}_aless_mjy"].to_numpy(float)
        for alpha, label in [(0.25, "hybrid25"), (0.50, "hybrid50"), (0.75, "hybrid75")]:
            # Both endpoints are normalised to the same object L_IR, so a linear
            # flux mix is a simple first-pass SED-shape interpolation.
            pred[f"F{band}_{label}_mjy"] = (1.0 - alpha) * fsps + alpha * aless

    return pred


def differential_counts(flux_mjy, bins_mjy, area_deg2=COSMOS_AREA_DEG2):
    flux = np.asarray(flux_mjy, dtype=float)
    flux = flux[np.isfinite(flux) & (flux > 0)]
    counts, edges = np.histogram(flux, bins=bins_mjy)
    centres_mjy = np.sqrt(edges[:-1] * edges[1:])
    centres_jy = centres_mjy / 1000.0
    width_jy = np.diff(edges) / 1000.0
    dnds = counts / (area_deg2 * width_jy)
    euclidean = (centres_jy**2.5) * dnds
    euclidean_err = (centres_jy**2.5) * np.sqrt(counts) / (area_deg2 * width_jy)
    euclidean_err[counts == 0] = np.nan
    euclidean = np.where(counts > 0, euclidean, np.nan)
    return pd.DataFrame(
        {
            "flux_mjy": centres_mjy,
            "bin_min_mjy": edges[:-1],
            "bin_max_mjy": edges[1:],
            "N_bin": counts,
            "euclidean_jy15_deg2": euclidean,
            "euclidean_err_jy15_deg2": euclidean_err,
        }
    )


def wang_differential_counts(wang, band, bins_mjy):
    flux = wang.loc[
        np.isfinite(wang[f"F{band}"])
        & np.isfinite(wang[f"SNR{band}"])
        & (wang[f"SNR{band}"] >= 3)
        & (wang[f"F{band}"] > 0),
        f"F{band}",
    ]
    return differential_counts(flux, bins_mjy)


def plot_external_points(ax, external, band):
    styles = {
        ("Clements et al.", "Table 1"): ("Clements H-ATLAS", "s", "#0072B2", 0.85),
        ("Oliver et al.", "Table 2"): ("Oliver HerMES", "D", "#009E73", 0.85),
        ("Pearson et al.", "Table 3 SUSSEXtractor"): ("Pearson SUSSEX", "^", "#D55E00", 0.75),
        ("Pearson et al.", "Table 4 XID"): ("Pearson XID", "v", "#E69F00", 0.75),
        ("Varnish et al.", "Table 4 P(D) best-fit spline"): ("Varnish P(D)", "o", "#CC79A7", 0.45),
    }

    sub = external[
        (external["band_um"] == band)
        & (external["flux_mjy"] >= 5)
        & (external["flux_mjy"] <= 1000)
        & (external["euclidean_best_jy15_deg2"] > 0)
    ]
    for (paper, method), group in sub.groupby(["paper", "method_or_table"]):
        label, marker, color, alpha = styles.get(
            (paper, method), (f"{paper} {method}", "o", "0.55", 0.6)
        )
        yerr = pd.to_numeric(group["euclidean_err_jy15_deg2"], errors="coerce")
        ax.errorbar(
            group["flux_mjy"],
            group["euclidean_best_jy15_deg2"],
            yerr=yerr if np.isfinite(yerr).any() else None,
            fmt=marker,
            ms=3.7,
            color=color,
            alpha=alpha,
            lw=0.8,
            capsize=1.5,
            label=label,
        )


def plot_model_lines(ax, count_tables, band, labels):
    for key, label, color, linestyle, width in labels:
        table = count_tables[(band, key)]
        ax.plot(
            table["flux_mjy"],
            table["euclidean_jy15_deg2"],
            label=label,
            color=color,
            ls=linestyle,
            lw=width,
        )


def make_count_tables(sample, wang, bins_mjy):
    model_cols = {
        "fsps": "fsps",
        "aless": "aless avg",
        "aless_bright": "ALESS bright",
        "aless_faint": "ALESS faint",
        "hybrid25": "hybrid 25%",
        "hybrid50": "hybrid 50%",
        "hybrid75": "hybrid 75%",
    }
    rows = []
    tables = {}
    for band in BANDS_UM:
        for key in model_cols:
            table = differential_counts(sample[f"F{band}_{key}_mjy"], bins_mjy)
            table.insert(0, "model", key)
            table.insert(0, "band_um", band)
            rows.append(table)
            tables[(band, key)] = table

        wtable = wang_differential_counts(wang, band, bins_mjy)
        wtable.insert(0, "model", "wang_snr3")
        wtable.insert(0, "band_um", band)
        rows.append(wtable)
        tables[(band, "wang_snr3")] = wtable

    return tables, pd.concat(rows, ignore_index=True)


def make_plots(tables, external):
    bins_note = (
        f"Wang-matched pop-cosmos sample; area={COSMOS_AREA_DEG2} deg^2; "
        "y-range clipped to 0.03-30"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    variant_lines = [
        ("fsps", "FSPS", "#0072B2", "-", 2.2),
        ("aless_faint", "ALESS faint", "#E69F00", ":", 1.6),
        ("aless", "ALESS avg", "#E69F00", "--", 2.0),
        ("aless_bright", "ALESS bright", "#E69F00", "-.", 1.6),
        ("wang_snr3", "Wang SNR>=3", "#000000", "-", 1.8),
    ]
    for ax, band in zip(axes, BANDS_UM):
        plot_external_points(ax, external, band)
        plot_model_lines(ax, tables, band, variant_lines)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5, 1000)
        ax.set_ylim(0.03, 30)
        ax.set_title(f"{band} um")
        ax.set_xlabel("Flux density S [mJy]")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$S^{2.5}dN/dS$ [Jy$^{1.5}$ deg$^{-2}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=7.2, ncol=1)
    fig.suptitle("ALESS template variants vs corrected external SPIRE counts")
    fig.text(0.5, 0.01, bins_note, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    variant_path = OUT_DIR / "popcosmos_aless_variant_differential_counts.png"
    fig.savefig(variant_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    hybrid_lines = [
        ("fsps", "FSPS", "#0072B2", "-", 2.2),
        ("hybrid25", "25% ALESS", "#56B4E9", "--", 1.7),
        ("hybrid50", "50% ALESS", "#D55E00", "-", 2.1),
        ("hybrid75", "75% ALESS", "#E69F00", "--", 1.7),
        ("aless", "ALESS avg", "#E69F00", ":", 2.0),
        ("wang_snr3", "Wang SNR>=3", "#000000", "-", 1.8),
    ]
    for ax, band in zip(axes, BANDS_UM):
        plot_external_points(ax, external, band)
        plot_model_lines(ax, tables, band, hybrid_lines)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5, 1000)
        ax.set_ylim(0.03, 30)
        ax.set_title(f"{band} um")
        ax.set_xlabel("Flux density S [mJy]")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$S^{2.5}dN/dS$ [Jy$^{1.5}$ deg$^{-2}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=7.2, ncol=1)
    fig.suptitle("Hybrid FSPS/ALESS SED-shape experiment")
    fig.text(0.5, 0.01, bins_note, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    hybrid_path = OUT_DIR / "popcosmos_hybrid_sed_differential_counts.png"
    fig.savefig(hybrid_path, dpi=180)
    plt.close(fig)

    return variant_path, hybrid_path


def summarize_bright_counts(sample, wang):
    rows = []
    for band in BANDS_UM:
        for cut in [10, 20, 50, 100]:
            row = {"band_um": band, "flux_cut_mjy": cut}
            for key in ["fsps", "aless_faint", "aless", "aless_bright", "hybrid25", "hybrid50", "hybrid75"]:
                flux = sample[f"F{band}_{key}_mjy"].to_numpy(float)
                row[f"N_{key}_per_deg2"] = np.sum(np.isfinite(flux) & (flux >= cut)) / COSMOS_AREA_DEG2
            wflux = wang.loc[
                np.isfinite(wang[f"F{band}"])
                & np.isfinite(wang[f"SNR{band}"])
                & (wang[f"SNR{band}"] >= 3)
                & (wang[f"F{band}"] >= cut),
                f"F{band}",
            ]
            row["N_wang_snr3_per_deg2"] = len(wflux) / COSMOS_AREA_DEG2
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    pred = pd.read_pickle(PREDICTION_CACHE)
    pred = add_aless_variants_and_hybrids(pred)
    pred.to_pickle(PREDICTION_CACHE)

    wang = pc.load_wang_bands()
    sample = pred.merge(wang[["ID"]], on="ID", how="inner")
    external = pd.read_csv(EXTERNAL_DIFF_COUNTS)
    bins_mjy = np.logspace(np.log10(5), np.log10(1000), 16)

    tables, model_counts = make_count_tables(sample, wang, bins_mjy)
    model_counts.to_csv(OUT_DIR / "popcosmos_hybrid_sed_differential_counts.csv", index=False)

    bright_summary = summarize_bright_counts(sample, wang)
    bright_summary.to_csv(OUT_DIR / "popcosmos_hybrid_sed_bright_count_summary.csv", index=False)

    variant_path, hybrid_path = make_plots(tables, external)
    print(variant_path)
    print(hybrid_path)
    print(OUT_DIR / "popcosmos_hybrid_sed_differential_counts.csv")
    print(OUT_DIR / "popcosmos_hybrid_sed_bright_count_summary.csv")


if __name__ == "__main__":
    main()
