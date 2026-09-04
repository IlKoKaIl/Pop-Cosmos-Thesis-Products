"""Modified-blackbody dust-temperature grid for pop-cosmos FIR counts.

This is a first physically motivated step after the ALESS hybrid:

1. Keep each pop-cosmos galaxy and its stored L_IR fixed.
2. Replace the far-IR bump with a simple optically-thin modified blackbody:
   L_nu shape proportional to nu^beta B_nu(T_dust).
3. Normalise that shape so the 8-1000 um integral is the same L_IR.
4. Redshift to observed 250/350/500 um fluxes.
5. Compare differential counts to the same published SPIRE count table.

This is not meant to be the final dust model. It is a clean temperature-family
diagnostic: does the evaluator prefer colder FSPS-like shapes or warmer dust?
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9
import astropy.units as u


NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
OUT_DIR = NB_DIR / "outputs"
EXTERNAL_COUNTS = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
PREDICTION_CACHE = OUT_DIR / "popcosmos_restframe_hybrid_predictions.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(NB_DIR))
import popcosmos_full_sed_250_counts as pc  # noqa: E402
import popcosmos_aless_hybrid_counts as ah  # noqa: E402


H_J_S = 6.62607015e-34
K_B_J_K = 1.380649e-23
BANDS_UM = [250, 350, 500]
TDUST_GRID_K = [20, 25, 30, 35, 40, 45, 50]
BETA = 1.8
MODEL_COUNT_AREA_DEG2 = ah.COSMOS_AREA_DEG2
MIN_SCORE_FLUX_MJY = 10.0
MAX_SCORE_FLUX_MJY = 300.0
LOG10_ERROR_FLOOR_DEX = 0.08


def planck_bnu_shape(nu_hz, temperature_k):
    """Return B_nu shape in arbitrary units, stable enough for this grid."""
    nu_hz = np.asarray(nu_hz, dtype=float)
    x = H_J_S * nu_hz / (K_B_J_K * temperature_k)
    x = np.clip(x, 1e-8, 700.0)
    return (2.0 * H_J_S * nu_hz**3 / pc.C_M_S**2) / np.expm1(x)


def mbb_shape_lnu(wave_um, temperature_k, beta=BETA):
    """Optically-thin modified blackbody Lnu shape, arbitrary normalisation."""
    wave_um = np.asarray(wave_um, dtype=float)
    nu_hz = pc.C_M_S / (wave_um * 1e-6)
    return (nu_hz**beta) * planck_bnu_shape(nu_hz, temperature_k)


def mbb_lir_integral(temperature_k, beta=BETA):
    """Integral of the unnormalised MBB shape over 8-1000 um."""
    wave_um = np.logspace(np.log10(8.0), np.log10(1000.0), 4096)
    nu_hz = pc.C_M_S / (wave_um * 1e-6)
    shape = mbb_shape_lnu(wave_um, temperature_k, beta=beta)
    order = np.argsort(nu_hz)
    return float(np.trapz(shape[order], nu_hz[order]))


def predict_mbb_flux_mjy(z, lir_lsun, lambda_obs_um, temperature_k, beta=BETA):
    """Observed flux density for an L_IR-normalised MBB template."""
    z = np.asarray(z, dtype=float)
    lir_lsun = np.asarray(lir_lsun, dtype=float)
    lambda_rest_um = lambda_obs_um / (1.0 + z)

    shape = mbb_shape_lnu(lambda_rest_um, temperature_k, beta=beta)
    norm = mbb_lir_integral(temperature_k, beta=beta)
    lnu_w_hz = (lir_lsun * pc.L_SUN_W) * shape / norm

    dl_m = WMAP9.luminosity_distance(z).to_value(u.m)
    fnu_w_m2_hz = (1.0 + z) * lnu_w_hz / (4.0 * np.pi * dl_m**2)
    return fnu_w_m2_hz / pc.MJY_TO_W_M2_HZ


def add_mbb_predictions(pred):
    out = pred.copy()
    z = out["z_pop"].to_numpy(float)
    lir = out["L_IR"].to_numpy(float)
    for temp in TDUST_GRID_K:
        key = f"mbb_T{temp:02d}"
        for band in BANDS_UM:
            out[f"F{band}_{key}_mjy"] = predict_mbb_flux_mjy(
                z,
                lir,
                lambda_obs_um=band,
                temperature_k=temp,
            )
    return out


def make_count_tables(sample, bins_mjy):
    rows = []
    tables = {}
    model_keys = [f"mbb_T{temp:02d}" for temp in TDUST_GRID_K]
    for band in BANDS_UM:
        for model in model_keys:
            table = ah.differential_counts(sample[f"F{band}_{model}_mjy"], bins_mjy)
            table.insert(0, "model", model)
            table.insert(0, "band_um", band)
            rows.append(table)
            tables[(band, model)] = table
    return tables, pd.concat(rows, ignore_index=True)


def log_interp(x, xp, fp):
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    good = np.isfinite(xp) & np.isfinite(fp) & (xp > 0) & (fp > 0)
    xp = xp[good]
    fp = fp[good]
    out = np.full_like(x, np.nan, dtype=float)
    if len(xp) < 2:
        return out
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    ok = np.isfinite(x) & (x > 0) & (x >= xp.min()) & (x <= xp.max())
    out[ok] = 10 ** np.interp(np.log10(x[ok]), np.log10(xp), np.log10(fp))
    return out


def prepare_external_counts():
    external = pd.read_csv(EXTERNAL_COUNTS)
    for col in ["flux_mjy", "euclidean_best_jy15_deg2", "euclidean_err_jy15_deg2"]:
        external[col] = pd.to_numeric(external[col], errors="coerce")
    external = external[
        np.isfinite(external["flux_mjy"])
        & np.isfinite(external["euclidean_best_jy15_deg2"])
        & np.isfinite(external["euclidean_err_jy15_deg2"])
        & (external["euclidean_best_jy15_deg2"] > 0)
        & (external["euclidean_err_jy15_deg2"] > 0)
        & (external["flux_mjy"] >= MIN_SCORE_FLUX_MJY)
        & (external["flux_mjy"] <= MAX_SCORE_FLUX_MJY)
    ].copy()
    external["source_key"] = external["paper"] + " / " + external["method_or_table"]
    external["count_regime"] = np.where(
        external["method_or_table"].str.contains(r"P\(D\)", regex=True),
        "P(D) statistical",
        "resolved/prior",
    )
    return external


def compare_model_to_external(model_counts, external_group, band, model):
    m = model_counts[
        (model_counts["band_um"] == band)
        & (model_counts["model"] == model)
        & (model_counts["N_bin"] > 0)
        & (model_counts["euclidean_jy15_deg2"] > 0)
    ].copy()
    obs = external_group[external_group["band_um"] == band].copy()
    if len(m) < 2 or obs.empty:
        return None

    pred = log_interp(
        obs["flux_mjy"].to_numpy(float),
        m["flux_mjy"].to_numpy(float),
        m["euclidean_jy15_deg2"].to_numpy(float),
    )
    pred_err = log_interp(
        obs["flux_mjy"].to_numpy(float),
        m["flux_mjy"].to_numpy(float),
        m["euclidean_err_jy15_deg2"].to_numpy(float),
    )
    obs_y = obs["euclidean_best_jy15_deg2"].to_numpy(float)
    obs_err = obs["euclidean_err_jy15_deg2"].to_numpy(float)
    ok = np.isfinite(pred) & np.isfinite(obs_y) & np.isfinite(obs_err) & (pred > 0) & (obs_y > 0)
    if ok.sum() < 2:
        return None

    obs_y = obs_y[ok]
    obs_err = obs_err[ok]
    pred = pred[ok]
    pred_err = pred_err[ok]
    obs_log = np.log10(obs_y)
    pred_log = np.log10(pred)
    obs_log_err = obs_err / (obs_y * np.log(10))
    pred_log_err = np.zeros_like(obs_log_err)
    good_pred_err = np.isfinite(pred_err) & (pred_err > 0)
    pred_log_err[good_pred_err] = pred_err[good_pred_err] / (pred[good_pred_err] * np.log(10))
    sigma_log = np.sqrt(obs_log_err**2 + pred_log_err**2 + LOG10_ERROR_FLOOR_DEX**2)
    delta_log = pred_log - obs_log
    chi2 = float(np.sum((delta_log / sigma_log) ** 2))

    return {
        "band_um": band,
        "model": model,
        "T_dust_K": int(model.split("T")[-1]),
        "N_points": int(ok.sum()),
        "chi2_log": chi2,
        "reduced_chi2_log": chi2 / int(ok.sum()),
        "median_log10_model_over_obs": float(np.nanmedian(delta_log)),
    }


def make_scorecard(model_counts):
    external = prepare_external_counts()
    model_keys = [f"mbb_T{temp:02d}" for temp in TDUST_GRID_K]
    rows = []
    for source_key, group in external.groupby("source_key"):
        for model in model_keys:
            band_rows = []
            for band in BANDS_UM:
                result = compare_model_to_external(model_counts, group, band, model)
                if result is None:
                    continue
                result["external_source"] = source_key
                rows.append(result)
                band_rows.append(result)
            if band_rows:
                rows.append(
                    {
                        "external_source": source_key,
                        "band_um": "all",
                        "model": model,
                        "T_dust_K": int(model.split("T")[-1]),
                        "N_points": sum(r["N_points"] for r in band_rows),
                        "chi2_log": sum(r["chi2_log"] for r in band_rows),
                        "reduced_chi2_log": sum(r["chi2_log"] for r in band_rows)
                        / sum(r["N_points"] for r in band_rows),
                        "median_log10_model_over_obs": float(
                            np.nanmedian([r["median_log10_model_over_obs"] for r in band_rows])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def make_pooled_summary(scorecard):
    sub = scorecard[scorecard["band_um"].astype(str) != "all"].copy()
    pooled = sub.groupby(["model", "T_dust_K"], as_index=False).agg(
        N_points=("N_points", "sum"),
        chi2_log=("chi2_log", "sum"),
        median_log10_model_over_obs=("median_log10_model_over_obs", "median"),
    )
    pooled["reduced_chi2_log"] = pooled["chi2_log"] / pooled["N_points"]
    return pooled.sort_values("reduced_chi2_log")


def make_regime_summary(scorecard):
    sub = scorecard[scorecard["band_um"].astype(str) != "all"].copy()
    sub["count_regime"] = np.where(
        sub["external_source"].str.contains(r"P\(D\)", regex=True),
        "P(D) statistical",
        "resolved/prior",
    )
    out = sub.groupby(["count_regime", "model", "T_dust_K"], as_index=False).agg(
        N_points=("N_points", "sum"),
        chi2_log=("chi2_log", "sum"),
        median_log10_model_over_obs=("median_log10_model_over_obs", "median"),
    )
    out["reduced_chi2_log"] = out["chi2_log"] / out["N_points"]
    return out.sort_values(["count_regime", "reduced_chi2_log"])


def make_bright_count_summary(sample):
    rows = []
    for band in BANDS_UM:
        for cut in [10, 20, 50, 100]:
            row = {"band_um": band, "flux_cut_mjy": cut}
            for temp in TDUST_GRID_K:
                key = f"mbb_T{temp:02d}"
                flux = sample[f"F{band}_{key}_mjy"].to_numpy(float)
                row[f"N_{key}_per_deg2"] = np.sum(np.isfinite(flux) & (flux >= cut)) / MODEL_COUNT_AREA_DEG2
            rows.append(row)
    return pd.DataFrame(rows)


def plot_mbb_shapes():
    wave_um = np.logspace(np.log10(8.0), np.log10(1000.0), 512)
    nu_hz = pc.C_M_S / (wave_um * 1e-6)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for temp in TDUST_GRID_K:
        shape = mbb_shape_lnu(wave_um, temp)
        integral = mbb_lir_integral(temp)
        nu_lnu_over_lir = nu_hz * shape / integral
        ax.plot(wave_um, nu_lnu_over_lir, label=f"{temp} K")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 3)
    ax.set_xlabel("rest wavelength (um)")
    ax.set_ylabel(r"shape: $\nu L_\nu / L_{\rm IR}$")
    ax.set_title(r"Simple modified-blackbody shapes, $\beta=1.8$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "popcosmos_mbb_temperature_grid_shapes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_count_grid(tables, external):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(TDUST_GRID_K)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, band in zip(axes, BANDS_UM):
        ah.plot_external_points(ax, external, band)
        for temp, color in zip(TDUST_GRID_K, colors):
            key = f"mbb_T{temp:02d}"
            tab = tables[(band, key)]
            ax.plot(
                tab["flux_mjy"],
                tab["euclidean_jy15_deg2"],
                color=color,
                lw=1.8 if temp in {25, 35, 45} else 1.1,
                label=f"T={temp}K",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5, 1000)
        ax.set_ylim(0.03, 30)
        ax.set_title(f"{band} um")
        ax.set_xlabel("Flux density S [mJy]")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$S^{2.5}dN/dS$ [Jy$^{1.5}$ deg$^{-2}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[0].legend(dedup.values(), dedup.keys(), fontsize=6.6, ncol=1)
    fig.suptitle(r"Modified-blackbody temperature grid, same pop-cosmos $L_{\rm IR}$")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "popcosmos_mbb_temperature_grid_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_score_summary(pooled, regime):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    pooled_plot = pooled.sort_values("T_dust_K")
    axes[0].plot(
        pooled_plot["T_dust_K"],
        pooled_plot["reduced_chi2_log"],
        marker="o",
        color="#0072B2",
    )
    axes[0].set_title("All external count sources")
    axes[0].set_xlabel("T_dust (K)")
    axes[0].set_ylabel(r"rough reduced $\chi^2$")
    axes[0].grid(True, alpha=0.25)

    for label, group in regime.groupby("count_regime"):
        group = group.sort_values("T_dust_K")
        axes[1].plot(group["T_dust_K"], group["reduced_chi2_log"], marker="o", label=label)
    axes[1].set_title("Split by count type")
    axes[1].set_xlabel("T_dust (K)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_yscale("log")
    fig.suptitle("Evaluator score for simple modified-blackbody temperature grid")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / "popcosmos_mbb_temperature_grid_score_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    pred = pd.read_pickle(PREDICTION_CACHE)
    pred = add_mbb_predictions(pred)
    wang = pc.load_wang_bands()
    sample = pred.merge(wang[["ID"]], on="ID", how="inner")
    external = pd.read_csv(EXTERNAL_COUNTS)
    bins_mjy = np.logspace(np.log10(5), np.log10(1000), 16)

    tables, model_counts = make_count_tables(sample, bins_mjy)
    scorecard = make_scorecard(model_counts)
    pooled = make_pooled_summary(scorecard)
    regime = make_regime_summary(scorecard)
    bright = make_bright_count_summary(sample)

    model_counts_path = OUT_DIR / "popcosmos_mbb_temperature_grid_differential_counts.csv"
    scorecard_path = OUT_DIR / "popcosmos_mbb_temperature_grid_scorecard.csv"
    pooled_path = OUT_DIR / "popcosmos_mbb_temperature_grid_pooled_summary.csv"
    regime_path = OUT_DIR / "popcosmos_mbb_temperature_grid_regime_summary.csv"
    bright_path = OUT_DIR / "popcosmos_mbb_temperature_grid_bright_count_summary.csv"

    model_counts.to_csv(model_counts_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    regime.to_csv(regime_path, index=False)
    bright.to_csv(bright_path, index=False)

    shapes_plot = plot_mbb_shapes()
    counts_plot = plot_count_grid(tables, external)
    score_plot = plot_score_summary(pooled, regime)

    print(model_counts_path)
    print(scorecard_path)
    print(pooled_path)
    print(regime_path)
    print(bright_path)
    print(shapes_plot)
    print(counts_plot)
    print(score_plot)
    print("\nPooled summary:")
    print(pooled.to_string(index=False))
    print("\nRegime summary:")
    print(regime.to_string(index=False))
    print("\nBright count summary at 20 mJy:")
    print(bright[bright["flux_cut_mjy"] == 20].to_string(index=False))


if __name__ == "__main__":
    main()
