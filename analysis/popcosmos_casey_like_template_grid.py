"""Simple Casey-like FIR SED grid for pop-cosmos counts.

Casey (2012) motivates a compact FIR SED model: a modified blackbody for the
cold dust peak joined to a mid-IR power law for warmer dust/AGN/clumpy regions.

This script implements a deliberately simple diagnostic version:

- optically-thin modified blackbody on the cold side
- power law on the Wien/mid-IR side
- smooth join where the MBB log-slope equals the chosen power-law slope
- every template is renormalised so its 8-1000 um integral matches each
  pop-cosmos galaxy's L_IR

The purpose is not final SED fitting. It tests whether adding a mid-IR tail to
the MBB temperature family changes the number-count evaluator result.
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
import popcosmos_mbb_temperature_grid as mbb  # noqa: E402


BANDS_UM = [250, 350, 500]
TDUST_GRID_K = [25, 30, 35, 40, 45]
ALPHA_GRID = [1.5, 2.0, 2.5, 3.0]
BETA = 1.8
MIN_SCORE_FLUX_MJY = 10.0
MAX_SCORE_FLUX_MJY = 300.0
LOG10_ERROR_FLOOR_DEX = 0.08


def model_key(temp_k, alpha):
    return f"casey_T{temp_k:02d}_a{int(alpha * 10):02d}"


def model_label(temp_k, alpha):
    return f"Casey T{temp_k}K a={alpha:.1f}"


def casey_join_um(temp_k, alpha, beta=BETA):
    """Find wavelength where MBB dlog(Lnu)/dlog(lambda) matches alpha."""
    wave = np.logspace(np.log10(8.0), np.log10(1000.0), 4096)
    shape = mbb.mbb_shape_lnu(wave, temp_k, beta=beta)
    slope = np.gradient(np.log(shape), np.log(wave))

    # On the Wien side the slope is large and positive; near the peak it drops.
    candidates = np.where((slope[:-1] >= alpha) & (slope[1:] < alpha))[0]
    if len(candidates) == 0:
        return float(wave[np.argmax(shape)])

    i = candidates[0]
    x0, x1 = slope[i], slope[i + 1]
    w0, w1 = wave[i], wave[i + 1]
    if x0 == x1:
        return float(w0)
    frac = (alpha - x0) / (x1 - x0)
    return float(np.exp(np.log(w0) + frac * (np.log(w1) - np.log(w0))))


def casey_like_shape_lnu(wave_um, temp_k, alpha, beta=BETA):
    """Continuous MBB + mid-IR power-law shape in Lnu units."""
    wave_um = np.asarray(wave_um, dtype=float)
    join = casey_join_um(temp_k, alpha, beta=beta)
    mbb_shape = mbb.mbb_shape_lnu(wave_um, temp_k, beta=beta)
    join_shape = mbb.mbb_shape_lnu(np.array([join]), temp_k, beta=beta)[0]
    powerlaw_shape = join_shape * (wave_um / join) ** alpha
    return np.where(wave_um < join, powerlaw_shape, mbb_shape)


def casey_lir_integral(temp_k, alpha, beta=BETA):
    wave = np.logspace(np.log10(8.0), np.log10(1000.0), 4096)
    nu_hz = pc.C_M_S / (wave * 1e-6)
    shape = casey_like_shape_lnu(wave, temp_k, alpha, beta=beta)
    order = np.argsort(nu_hz)
    return float(np.trapz(shape[order], nu_hz[order]))


def predict_casey_flux_mjy(z, lir_lsun, lambda_obs_um, temp_k, alpha, beta=BETA):
    z = np.asarray(z, dtype=float)
    lir_lsun = np.asarray(lir_lsun, dtype=float)
    lambda_rest_um = lambda_obs_um / (1.0 + z)

    shape = casey_like_shape_lnu(lambda_rest_um, temp_k, alpha, beta=beta)
    norm = casey_lir_integral(temp_k, alpha, beta=beta)
    lnu_w_hz = (lir_lsun * pc.L_SUN_W) * shape / norm

    dl_m = WMAP9.luminosity_distance(z).to_value(u.m)
    fnu_w_m2_hz = (1.0 + z) * lnu_w_hz / (4.0 * np.pi * dl_m**2)
    return fnu_w_m2_hz / pc.MJY_TO_W_M2_HZ


def add_casey_predictions(pred):
    out = pred.copy()
    z = out["z_pop"].to_numpy(float)
    lir = out["L_IR"].to_numpy(float)
    for temp in TDUST_GRID_K:
        for alpha in ALPHA_GRID:
            key = model_key(temp, alpha)
            for band in BANDS_UM:
                out[f"F{band}_{key}_mjy"] = predict_casey_flux_mjy(
                    z,
                    lir,
                    lambda_obs_um=band,
                    temp_k=temp,
                    alpha=alpha,
                )
    return out


def make_count_tables(sample, bins_mjy):
    rows = []
    tables = {}
    for temp in TDUST_GRID_K:
        for alpha in ALPHA_GRID:
            key = model_key(temp, alpha)
            for band in BANDS_UM:
                table = ah.differential_counts(sample[f"F{band}_{key}_mjy"], bins_mjy)
                table.insert(0, "alpha_mir", alpha)
                table.insert(0, "T_dust_K", temp)
                table.insert(0, "model_label", model_label(temp, alpha))
                table.insert(0, "model", key)
                table.insert(0, "band_um", band)
                rows.append(table)
                tables[(band, key)] = table
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
    return external


def compare_model_to_external(model_counts, external_group, band, key):
    m = model_counts[
        (model_counts["band_um"] == band)
        & (model_counts["model"] == key)
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
    base = m.iloc[0]

    return {
        "band_um": band,
        "model": key,
        "model_label": base["model_label"],
        "T_dust_K": int(base["T_dust_K"]),
        "alpha_mir": float(base["alpha_mir"]),
        "N_points": int(ok.sum()),
        "chi2_log": chi2,
        "reduced_chi2_log": chi2 / int(ok.sum()),
        "median_log10_model_over_obs": float(np.nanmedian(delta_log)),
    }


def make_scorecard(model_counts):
    external = prepare_external_counts()
    rows = []
    keys = sorted(model_counts["model"].unique())
    for source_key, group in external.groupby("source_key"):
        for key in keys:
            band_rows = []
            for band in BANDS_UM:
                result = compare_model_to_external(model_counts, group, band, key)
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
                        "model": key,
                        "model_label": band_rows[0]["model_label"],
                        "T_dust_K": band_rows[0]["T_dust_K"],
                        "alpha_mir": band_rows[0]["alpha_mir"],
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
    pooled = sub.groupby(["model", "model_label", "T_dust_K", "alpha_mir"], as_index=False).agg(
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
    out = sub.groupby(
        ["count_regime", "model", "model_label", "T_dust_K", "alpha_mir"],
        as_index=False,
    ).agg(
        N_points=("N_points", "sum"),
        chi2_log=("chi2_log", "sum"),
        median_log10_model_over_obs=("median_log10_model_over_obs", "median"),
    )
    out["reduced_chi2_log"] = out["chi2_log"] / out["N_points"]
    return out.sort_values(["count_regime", "reduced_chi2_log"])


def make_template_shape_plot():
    wave = np.logspace(np.log10(8), np.log10(1000), 600)
    nu_hz = pc.C_M_S / (wave * 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    for temp in [30, 35, 40]:
        alpha = 2.0
        shape = casey_like_shape_lnu(wave, temp, alpha)
        norm = casey_lir_integral(temp, alpha)
        axes[0].plot(wave, nu_hz * shape / norm, label=f"T={temp}K, a=2.0")

    for alpha in ALPHA_GRID:
        temp = 35
        shape = casey_like_shape_lnu(wave, temp, alpha)
        norm = casey_lir_integral(temp, alpha)
        axes[1].plot(wave, nu_hz * shape / norm, label=f"T=35K, a={alpha:.1f}")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 3)
        ax.set_xlabel("rest wavelength (um)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"shape: $\nu L_\nu / L_{\rm IR}$")
    axes[0].set_title("Changing dust temperature")
    axes[1].set_title("Changing mid-IR slope")
    fig.suptitle(r"Simplified Casey-like templates, $\beta=1.8$")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / "popcosmos_casey_like_template_grid_shapes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_score_heatmap(pooled):
    pivot = pooled.pivot(index="alpha_mir", columns="T_dust_K", values="reduced_chi2_log")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(np.log10(pivot.to_numpy(float)), origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("T_dust (K)")
    ax.set_ylabel("mid-IR slope alpha")
    ax.set_title(r"Casey-like grid score: $\log_{10}$ rough reduced $\chi^2$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\chi^2_\nu)$")
    for i, alpha in enumerate(pivot.index):
        for j, temp in enumerate(pivot.columns):
            val = pivot.loc[alpha, temp]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    path = OUT_DIR / "popcosmos_casey_like_template_grid_score_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_count_plot(tables, external, pooled):
    top = pooled.head(4)
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, band in zip(axes, BANDS_UM):
        ah.plot_external_points(ax, external, band)
        for (_, row), color in zip(top.iterrows(), colors):
            key = row["model"]
            tab = tables[(band, key)]
            ax.plot(
                tab["flux_mjy"],
                tab["euclidean_jy15_deg2"],
                color=color,
                lw=2,
                label=row["model_label"],
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
    axes[0].legend(dedup.values(), dedup.keys(), fontsize=6.6)
    fig.suptitle("Best simplified Casey-like templates vs external SPIRE counts")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "popcosmos_casey_like_template_grid_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    pred = pd.read_pickle(PREDICTION_CACHE)
    pred = add_casey_predictions(pred)
    wang = pc.load_wang_bands()
    sample = pred.merge(wang[["ID"]], on="ID", how="inner")
    external = pd.read_csv(EXTERNAL_COUNTS)
    bins_mjy = np.logspace(np.log10(5), np.log10(1000), 16)

    tables, model_counts = make_count_tables(sample, bins_mjy)
    scorecard = make_scorecard(model_counts)
    pooled = make_pooled_summary(scorecard)
    regime = make_regime_summary(scorecard)

    model_counts_path = OUT_DIR / "popcosmos_casey_like_template_grid_differential_counts.csv"
    scorecard_path = OUT_DIR / "popcosmos_casey_like_template_grid_scorecard.csv"
    pooled_path = OUT_DIR / "popcosmos_casey_like_template_grid_pooled_summary.csv"
    regime_path = OUT_DIR / "popcosmos_casey_like_template_grid_regime_summary.csv"
    model_counts.to_csv(model_counts_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    regime.to_csv(regime_path, index=False)

    shape_plot = make_template_shape_plot()
    heatmap = make_score_heatmap(pooled)
    counts_plot = make_count_plot(tables, external, pooled)

    print(model_counts_path)
    print(scorecard_path)
    print(pooled_path)
    print(regime_path)
    print(shape_plot)
    print(heatmap)
    print(counts_plot)
    print("\nPooled summary:")
    print(pooled.to_string(index=False))
    print("\nBest by regime:")
    print(regime.groupby("count_regime").head(3).to_string(index=False))


if __name__ == "__main__":
    main()
