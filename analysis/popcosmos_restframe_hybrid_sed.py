"""Rest-frame FSPS/ALESS hybrid SED check.

The earlier hybrid check mixed the final observed fluxes. This script makes the
same idea more explicit in SED space:

1. Check that each pop-cosmos FSPS SED integrates to its stored L_IR.
2. Normalise the FSPS FIR SED to L_IR and the ALESS template to the same L_IR.
3. Mix the rest-frame shapes with a few alpha values.
4. Convert the mixed SEDs to observed SPIRE flux counts.

Because the flux calculation is linear, this should be almost identical to the
old flux-space hybrid if both endpoint SEDs are already normalised correctly.
The point is to make the method defensible and easier to extend to real dust
template grids later.
"""

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
NB_DIR = CODE_DIR.parent
ROOT = NB_DIR.parent
OUT_DIR = NB_DIR / "outputs"
EXTERNAL_DIFF_COUNTS = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
PREDICTION_CACHE = OUT_DIR / "popcosmos_full_sed_band_predictions.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
import popcosmos_full_sed_250_counts as pc  # noqa: E402
import popcosmos_aless_hybrid_counts as ah  # noqa: E402


BANDS_UM = [250, 350, 500]
ALPHAS = [(0.25, "resthybrid25"), (0.50, "resthybrid50"), (0.75, "resthybrid75")]
FSPS_RATIO_CACHE = OUT_DIR / "popcosmos_fsps_lir_integral_ratio.csv"
HYBRID_PREDICTIONS = OUT_DIR / "popcosmos_restframe_hybrid_predictions.pkl"


def integrate_lnu_over_lir_band_lsun(wave_um, lnu_lsun_hz):
    """Integral of Lnu dnu over 8-1000 um, in Lsun."""
    wave_um = np.asarray(wave_um, dtype=float)
    lnu_lsun_hz = np.asarray(lnu_lsun_hz, dtype=float)
    mask = (
        np.isfinite(wave_um)
        & np.isfinite(lnu_lsun_hz)
        & (wave_um >= 8.0)
        & (wave_um <= 1000.0)
        & (lnu_lsun_hz > 0)
    )
    if mask.sum() < 2:
        return np.nan
    nu_hz = pc.C_M_S / (wave_um[mask] * 1e-6)
    order = np.argsort(nu_hz)
    return float(np.trapz(lnu_lsun_hz[mask][order], nu_hz[order]))


def compute_or_load_fsps_lir_ratios(batch_size=8192):
    """Check the energy-balance normalisation of the full FSPS SED file."""
    if FSPS_RATIO_CACHE.exists():
        return pd.read_csv(FSPS_RATIO_CACHE)

    with h5py.File(pc.FULL_SED_H5, "r") as f:
        wave_um = f["wave_rest"][:] / 1e4
        idx = np.where((wave_um >= 8.0) & (wave_um <= 1000.0))[0]
        wave_lir = wave_um[idx]
        nu_hz = pc.C_M_S / (wave_lir * 1e-6)
        order = np.argsort(nu_hz)

        n = f["index"].shape[0]
        ids = f["index"][:].astype(np.int64)
        rows = f["row"][:].astype(np.int64)
        lir = f["L_IR"][:].astype(float)
        integral = np.full(n, np.nan, dtype=float)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            spec = f["spec_attenuated"][start:end, idx]
            integral[start:end] = np.trapz(spec[:, order], nu_hz[order], axis=1)

    ratio = integral / lir
    out = pd.DataFrame(
        {
            "row": rows,
            "ID": ids,
            "L_IR": lir,
            "fsps_lir_integral_lsun": integral,
            "fsps_lir_ratio": ratio,
        }
    )
    out.to_csv(FSPS_RATIO_CACHE, index=False)
    return out


def add_restframe_hybrid_fluxes(pred, ratios):
    """Add rest-frame-normalised hybrid flux columns to the prediction table."""
    out = pred.merge(
        ratios[["ID", "fsps_lir_integral_lsun", "fsps_lir_ratio"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    for band in BANDS_UM + [850]:
        fsps = out[f"F{band}_fsps_mjy"].to_numpy(float)
        aless = out[f"F{band}_aless_mjy"].to_numpy(float)
        ratio = out["fsps_lir_ratio"].to_numpy(float)

        # The full FSPS file already integrates to L_IR, but this makes the
        # intended rest-frame normalisation explicit.
        fsps_norm = fsps / ratio
        out[f"F{band}_fsps_lirnorm_mjy"] = fsps_norm

        for alpha, label in ALPHAS:
            out[f"F{band}_{label}_mjy"] = (1.0 - alpha) * fsps_norm + alpha * aless

    return out


def wang_differential_counts(wang, band, bins_mjy):
    """Raw Wang positive-flux catalogue counts, with no SNR>=3 cut."""
    flux = wang.loc[
        np.isfinite(wang[f"F{band}"])
        & np.isfinite(wang[f"s_F{band}"])
        & (wang[f"s_F{band}"] > 0)
        & (wang[f"F{band}"] > 0),
        f"F{band}",
    ]
    return ah.differential_counts(flux, bins_mjy)


def make_count_tables(sample, wang, bins_mjy):
    model_keys = ["fsps_lirnorm", "resthybrid25", "resthybrid50", "resthybrid75", "aless"]
    tables = {}
    rows = []

    for band in BANDS_UM:
        for key in model_keys:
            table = ah.differential_counts(sample[f"F{band}_{key}_mjy"], bins_mjy)
            table.insert(0, "model", key)
            table.insert(0, "band_um", band)
            tables[(band, key)] = table
            rows.append(table)

        wtable = wang_differential_counts(wang, band, bins_mjy)
        wtable.insert(0, "model", "wang_raw")
        wtable.insert(0, "band_um", band)
        tables[(band, "wang_raw")] = wtable
        rows.append(wtable)

    return tables, pd.concat(rows, ignore_index=True)


def plot_restframe_hybrid_counts(tables, external):
    lines = [
        ("fsps_lirnorm", "FSPS, LIR-normalised", "#0072B2", "-", 2.2),
        ("resthybrid25", "25% ALESS", "#56B4E9", "--", 1.7),
        ("resthybrid50", "50% ALESS", "#D55E00", "-", 2.1),
        ("resthybrid75", "75% ALESS", "#E69F00", "--", 1.7),
        ("aless", "ALESS avg", "#E69F00", ":", 2.0),
        ("wang_raw", "Wang raw", "#000000", "-", 1.8),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, band in zip(axes, BANDS_UM):
        ah.plot_external_points(ax, external, band)
        ah.plot_model_lines(ax, tables, band, lines)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5, 1000)
        ax.set_ylim(0.03, 30)
        ax.set_title(f"{band} um")
        ax.set_xlabel("Flux density S [mJy]")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"$S^{2.5}dN/dS$ [Jy$^{1.5}$ deg$^{-2}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=7.2)
    fig.suptitle("Rest-frame LIR-normalised FSPS/ALESS hybrid SED counts")
    fig.text(
        0.5,
        0.01,
        (
            f"Wang-matched pop-cosmos sample; area={ah.COSMOS_AREA_DEG2} deg^2; "
            "same objects and L_IR, only FIR SED shape changes"
        ),
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = OUT_DIR / "popcosmos_restframe_hybrid_sed_differential_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_bright_count_summary(sample, wang):
    rows = []
    for band in BANDS_UM:
        for cut in [10, 20, 50, 100]:
            row = {"band_um": band, "flux_cut_mjy": cut}
            for key in ["fsps_lirnorm", "resthybrid25", "resthybrid50", "resthybrid75", "aless"]:
                flux = sample[f"F{band}_{key}_mjy"].to_numpy(float)
                row[f"N_{key}_per_deg2"] = np.sum(np.isfinite(flux) & (flux >= cut)) / ah.COSMOS_AREA_DEG2
            wflux = wang.loc[
                np.isfinite(wang[f"F{band}"])
                & np.isfinite(wang[f"s_F{band}"])
                & (wang[f"s_F{band}"] > 0)
                & (wang[f"F{band}"] >= cut),
                f"F{band}",
            ]
            row["N_wang_raw_per_deg2"] = len(wflux) / ah.COSMOS_AREA_DEG2
            rows.append(row)
    return pd.DataFrame(rows)


def make_method_check(pred):
    rows = []
    for band in BANDS_UM:
        for alpha, label in ALPHAS:
            old_col = f"F{band}_hybrid{int(alpha * 100):02d}_mjy"
            new_col = f"F{band}_{label}_mjy"
            if old_col not in pred.columns:
                continue
            old = pred[old_col].to_numpy(float)
            new = pred[new_col].to_numpy(float)
            ok = np.isfinite(old) & np.isfinite(new) & (old > 0)
            frac = (new[ok] - old[ok]) / old[ok]
            rows.append(
                {
                    "band_um": band,
                    "alpha": alpha,
                    "old_flux_mix_col": old_col,
                    "new_restframe_col": new_col,
                    "N": int(ok.sum()),
                    "median_frac_change": np.nanmedian(frac),
                    "p16_frac_change": np.nanpercentile(frac, 16),
                    "p84_frac_change": np.nanpercentile(frac, 84),
                    "max_abs_frac_change": np.nanmax(np.abs(frac)),
                }
            )
    return pd.DataFrame(rows)


def plot_restframe_sed_examples(pred):
    base = pred[
        pred["done"]
        & np.isfinite(pred["SFR_pop"])
        & np.isfinite(pred["L_IR"])
        & (pred["SFR_pop"] > 0)
        & (pred["L_IR"] > 0)
    ].sort_values("SFR_pop")
    choices = [
        ("low SFR", base.iloc[int(0.05 * (len(base) - 1))]),
        ("median SFR", base.iloc[int(0.50 * (len(base) - 1))]),
        ("high SFR", base.iloc[int(0.995 * (len(base) - 1))]),
    ]

    template = pc.load_aless_template(pc.ALESS_PATH)
    lam_t = template["lambda_um"].to_numpy(float)
    fnu_t = template["fnu_average_mjy"].to_numpy(float)
    aless_integral = pc.template_integral_fnu_dnu(template)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    with h5py.File(pc.FULL_SED_H5, "r") as f:
        wave_um = f["wave_rest"][:] / 1e4
        nu_hz = pc.C_M_S / (wave_um * 1e-6)

        for ax, (label, row) in zip(axes, choices):
            sed = f["spec_attenuated"][int(row["row"]), :]
            lir = float(row["L_IR"])
            fsps_integral = integrate_lnu_over_lir_band_lsun(wave_um, sed)
            fsps_norm_lnu = sed * lir / fsps_integral

            aless_shape = pc.log_interp_positive(wave_um, lam_t, fnu_t)
            aless_lnu = aless_shape * lir / aless_integral
            hybrid_lnu = 0.5 * fsps_norm_lnu + 0.5 * aless_lnu

            m = (
                np.isfinite(wave_um)
                & (wave_um >= 5)
                & (wave_um <= 1000)
                & np.isfinite(fsps_norm_lnu)
                & np.isfinite(aless_lnu)
                & np.isfinite(hybrid_lnu)
                & (fsps_norm_lnu > 0)
                & (aless_lnu > 0)
                & (hybrid_lnu > 0)
            )
            ax.plot(wave_um[m], nu_hz[m] * fsps_norm_lnu[m] / lir, label="FSPS", color="#0072B2", lw=2)
            ax.plot(wave_um[m], nu_hz[m] * aless_lnu[m] / lir, label="ALESS", color="#E69F00", lw=2, ls="--")
            ax.plot(wave_um[m], nu_hz[m] * hybrid_lnu[m] / lir, label="50% hybrid", color="#D55E00", lw=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{label}\nSFR={row['SFR_pop']:.2g}, logLIR={row['log10LIR']:.2f}")
            ax.set_xlabel("rest wavelength (um)")
            ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"shape: $\nu L_\nu / L_{\rm IR}$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Rest-frame SED mixing example: FSPS + ALESS on the same grid")
    fig.tight_layout()
    path = OUT_DIR / "popcosmos_restframe_hybrid_sed_examples.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    pred = pd.read_pickle(PREDICTION_CACHE)
    ratios = compute_or_load_fsps_lir_ratios()
    pred = add_restframe_hybrid_fluxes(pred, ratios)
    pred.to_pickle(HYBRID_PREDICTIONS)

    wang = pc.load_wang_bands()
    sample = pred.merge(wang[["ID"]], on="ID", how="inner")
    external = pd.read_csv(EXTERNAL_DIFF_COUNTS)
    bins_mjy = np.logspace(np.log10(5), np.log10(1000), 16)

    tables, model_counts = make_count_tables(sample, wang, bins_mjy)
    model_counts_path = OUT_DIR / "popcosmos_restframe_hybrid_sed_differential_counts.csv"
    model_counts.to_csv(model_counts_path, index=False)

    bright = make_bright_count_summary(sample, wang)
    bright_path = OUT_DIR / "popcosmos_restframe_hybrid_bright_count_summary.csv"
    bright.to_csv(bright_path, index=False)

    method_check = make_method_check(pred)
    method_check_path = OUT_DIR / "popcosmos_restframe_hybrid_method_check.csv"
    method_check.to_csv(method_check_path, index=False)

    count_plot = plot_restframe_hybrid_counts(tables, external)
    sed_plot = plot_restframe_sed_examples(pred)

    print(count_plot)
    print(sed_plot)
    print(model_counts_path)
    print(bright_path)
    print(method_check_path)
    print(FSPS_RATIO_CACHE)

    print("\nFSPS integral / stored L_IR percentiles:")
    good = ratios["fsps_lir_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    print(good.quantile([0.01, 0.16, 0.50, 0.84, 0.99]).to_string())

    print("\nBright count summary at 20 mJy:")
    print(bright[bright["flux_cut_mjy"] == 20].to_string(index=False))

    print("\nRest-frame hybrid vs old flux-mix fractional change:")
    print(method_check.to_string(index=False))


if __name__ == "__main__":
    main()
