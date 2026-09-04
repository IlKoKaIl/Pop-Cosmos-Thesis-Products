"""First ALESS-template check for pop-cosmos FIR fluxes.

 simple bridge test:

1. Treat the ALESS average SED as a FIR SED shape.
2. Scale that shape to each pop-cosmos galaxy's model L_IR.
3. Predict observed-frame fluxes at 24/100/160/250/350/500/850 um.
4. Summarize high-SFR subsets and compare the template-predicted fluxes
   to Wang observed fluxes where available.

This does not replace a true pop-cosmos FIR SED calculation. It is a first
sanity check for the supervisor suggestion: "if high-SFR pop-cosmos galaxies
look like ALESS SMGs, what fluxes would we expect?"
"""

from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9
import astropy.units as u
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALESS_PATH = ROOT / "aless_average_seds.dat.txt"
POP_H5 = ROOT / "catalog data/real pop-cosmos data/mcmc_summaries.h5"
LIR_H5 = ROOT / "fsps_lir_scalars.h5"
WANG_MASTER = ROOT / "catalog data/wang/master.dat.gz"
WANG_README = ROOT / "catalog data/wang/ReadMe.txt"

BANDS_UM = [24, 100, 160, 250, 350, 500, 850]
SFR_THRESHOLDS = [10, 30, 100]
L_SUN_W = 3.828e26
C_M_S = 299792458.0
MJY_TO_W_M2_HZ = 1e-29


def load_aless_template(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path, comments="#")
    return pd.DataFrame(
        {
            "lambda_um": data[:, 0],
            "fnu_average_mjy": data[:, 1],
            "fnu_bright_mjy": data[:, 2],
            "fnu_faint_mjy": data[:, 3],
        }
    )


def log_interp_positive(x, xp, fp):
    """Log-log interpolation for positive template fluxes."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    good_template = np.isfinite(xp) & np.isfinite(fp) & (xp > 0) & (fp > 0)
    xp_good = np.asarray(xp[good_template], dtype=float)
    fp_good = np.asarray(fp[good_template], dtype=float)
    good_x = (
        np.isfinite(x) & (x > 0) & (x >= np.nanmin(xp_good)) & (x <= np.nanmax(xp_good))
    )
    out[good_x] = 10 ** np.interp(
        np.log10(x[good_x]),
        np.log10(xp_good),
        np.log10(fp_good),
    )
    return out


def template_integral_fnu_dnu(
    template: pd.DataFrame, fcol: str = "fnu_average_mjy"
) -> float:
    """Integrate arbitrary F_nu template shape over rest-frame 8-1000 um."""
    lam = template["lambda_um"].to_numpy(float)
    fnu = template[fcol].to_numpy(float)
    mask = (
        np.isfinite(lam) & np.isfinite(fnu) & (lam >= 8.0) & (lam <= 1000.0) & (fnu > 0)
    )
    lam_m = lam[mask] * 1e-6
    nu_hz = C_M_S / lam_m
    order = np.argsort(nu_hz)
    return float(np.trapz(fnu[mask][order], nu_hz[order]))


def scaled_template_nu_lnu(
    template: pd.DataFrame, lir_lsun: float, fcol: str = "fnu_average_mjy"
):
    """Return rest lambda and nu L_nu / Lsun for the template scaled to L_IR."""
    lam = template["lambda_um"].to_numpy(float)
    fnu_shape = template[fcol].to_numpy(float)
    integral = template_integral_fnu_dnu(template, fcol)
    scale = lir_lsun * L_SUN_W / integral
    lnu_w_hz = scale * fnu_shape
    nu_hz = C_M_S / (lam * 1e-6)
    nu_lnu_lsun = nu_hz * lnu_w_hz / L_SUN_W
    return lam, nu_lnu_lsun


def predict_flux_mjy(
    template: pd.DataFrame, z, lir_lsun, lambda_obs_um, fcol: str = "fnu_average_mjy"
):
    """Predict observed flux density for a L_IR-scaled ALESS SED shape."""
    z = np.asarray(z, dtype=float)
    lir_lsun = np.asarray(lir_lsun, dtype=float)
    lambda_rest_um = lambda_obs_um / (1.0 + z)

    lam_grid = template["lambda_um"].to_numpy(float)
    fnu_grid = template[fcol].to_numpy(float)
    shape_fnu = log_interp_positive(lambda_rest_um, lam_grid, fnu_grid)

    integral = template_integral_fnu_dnu(template, fcol)
    scale = lir_lsun * L_SUN_W / integral
    lnu_w_hz = scale * shape_fnu

    dl_m = WMAP9.luminosity_distance(z).to_value(u.m)
    fnu_w_m2_hz = (1.0 + z) * lnu_w_hz / (4.0 * np.pi * dl_m**2)
    return fnu_w_m2_hz / MJY_TO_W_M2_HZ


def load_pop_with_lir() -> pd.DataFrame:
    with h5py.File(POP_H5, "r") as f:
        pop = pd.DataFrame(
            {
                "ID": f["metadata/index_farmer"][:].astype(np.int64),
                "z_pop": f["pop-cosmos/z"][:, 2],
                "log10M_pop": f["pop-cosmos/log10M_remain"][:, 2],
                "log10SFR_pop": f["pop-cosmos/log10SFR"][:, 2],
                "log10sSFR_pop": f["pop-cosmos/log10sSFR"][:, 2],
                "dust2_pop": f["pop-cosmos/dust2"][:, 2],
                "lnfAGN_pop": f["pop-cosmos/lnfAGN"][:, 2],
            }
        )

    with h5py.File(LIR_H5, "r") as f:
        lir = pd.DataFrame(
            {
                "ID": f["index"][:].astype(np.int64),
                "L_IR": f["L_IR"][:],
                "done_lir": f["done"][:].astype(bool),
            }
        )

    df = pop.merge(lir, on="ID", how="inner", validate="one_to_one")
    df = df[(df["ID"] > 0) & df["done_lir"]].copy()
    df["SFR_pop"] = 10 ** df["log10SFR_pop"]
    df["log10LIR"] = np.log10(df["L_IR"])
    return df


def load_wang() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = Table.read(WANG_MASTER, format="ascii.cds", readme=WANG_README)

    df = tab.to_pandas()
    keep = ["ID"]
    for band in BANDS_UM:
        keep.extend([f"F{band}", f"s_F{band}"])
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[df["ID"] > 0].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for band in BANDS_UM:
        fcol = f"F{band}"
        scol = f"s_F{band}"
        if fcol in df.columns and scol in df.columns:
            df[f"SNR{band}"] = df[fcol] / df[scol].replace(0, np.nan)

    return df


def add_aless_fluxes(pop: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    out = pop.copy()
    z = out["z_pop"].to_numpy(float)
    lir = out["L_IR"].to_numpy(float)
    for band in BANDS_UM:
        out[f"F{band}_aless_mjy"] = predict_flux_mjy(template, z, lir, band)
    return out


def summarize_predicted_fluxes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sfr_min in SFR_THRESHOLDS:
        m = (
            np.isfinite(df["SFR_pop"])
            & np.isfinite(df["z_pop"])
            & np.isfinite(df["L_IR"])
            & (df["SFR_pop"] >= sfr_min)
            & (df["z_pop"] > 0)
            & (df["z_pop"] < 6)
        )
        sub = df.loc[m].copy()
        row = {
            "sfr_min_msun_per_yr": sfr_min,
            "N": int(len(sub)),
            "median_z": float(np.nanmedian(sub["z_pop"])) if len(sub) else np.nan,
            "median_log10SFR_pop": (
                float(np.nanmedian(sub["log10SFR_pop"])) if len(sub) else np.nan
            ),
            "median_log10LIR": (
                float(np.nanmedian(sub["log10LIR"])) if len(sub) else np.nan
            ),
        }
        for band in [24, 250, 350, 500, 850]:
            col = f"F{band}_aless_mjy"
            arr = sub[col].to_numpy(float)
            row[f"median_F{band}_aless_mjy"] = (
                float(np.nanmedian(arr)) if len(sub) else np.nan
            )
            row[f"p16_F{band}_aless_mjy"] = (
                float(np.nanpercentile(arr, 16)) if len(sub) else np.nan
            )
            row[f"p84_F{band}_aless_mjy"] = (
                float(np.nanpercentile(arr, 84)) if len(sub) else np.nan
            )
            row[f"N_F{band}_gt_1mJy"] = int(np.nansum(arr > 1.0)) if len(sub) else 0
            row[f"N_F{band}_gt_3mJy"] = int(np.nansum(arr > 3.0)) if len(sub) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_wang_comparison(pred: pd.DataFrame, wang: pd.DataFrame) -> pd.DataFrame:
    matched = pred.merge(wang, on="ID", how="inner")
    rows = []
    for sfr_min in SFR_THRESHOLDS:
        for band in [24, 250, 350, 500, 850]:
            obs_col = f"F{band}"
            pred_col = f"F{band}_aless_mjy"
            snr_col = f"SNR{band}"
            if obs_col not in matched.columns or snr_col not in matched.columns:
                continue
            mask = (
                (matched["SFR_pop"] >= sfr_min)
                & np.isfinite(matched[obs_col])
                & np.isfinite(matched[pred_col])
                & np.isfinite(matched[snr_col])
                & (matched[snr_col] >= 3)
                & (matched[obs_col] > 0)
                & (matched[pred_col] > 0)
            )
            sub = matched.loc[
                mask, [obs_col, pred_col, "z_pop", "log10SFR_pop", "log10LIR"]
            ].copy()
            if len(sub) == 0:
                rows.append(
                    {
                        "sfr_min_msun_per_yr": sfr_min,
                        "band_um": band,
                        "N": 0,
                        "median_z": np.nan,
                        "median_observed_mjy": np.nan,
                        "median_aless_pred_mjy": np.nan,
                        "median_log10_pred_over_obs": np.nan,
                        "spearman_logpred_vs_logobs": np.nan,
                    }
                )
                continue

            log_obs = np.log10(sub[obs_col])
            log_pred = np.log10(sub[pred_col])
            rows.append(
                {
                    "sfr_min_msun_per_yr": sfr_min,
                    "band_um": band,
                    "N": int(len(sub)),
                    "median_z": float(np.nanmedian(sub["z_pop"])),
                    "median_log10SFR_pop": float(np.nanmedian(sub["log10SFR_pop"])),
                    "median_log10LIR": float(np.nanmedian(sub["log10LIR"])),
                    "median_observed_mjy": float(np.nanmedian(sub[obs_col])),
                    "median_aless_pred_mjy": float(np.nanmedian(sub[pred_col])),
                    "median_log10_pred_over_obs": float(
                        np.nanmedian(log_pred - log_obs)
                    ),
                    "spearman_logpred_vs_logobs": float(
                        log_pred.corr(log_obs, method="spearman")
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_template(template: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for col, label, color in [
        ("fnu_average_mjy", "ALESS average", "black"),
        ("fnu_bright_mjy", "optically bright", "#1f77b4"),
        ("fnu_faint_mjy", "optically faint", "#d62728"),
    ]:
        m = template[col] > 0
        ax.loglog(
            template.loc[m, "lambda_um"],
            template.loc[m, col],
            lw=1.6,
            label=label,
            color=color,
        )

    for x in [8, 24, 100, 250, 350, 500, 850, 1000]:
        ax.axvline(x, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("rest wavelength [um]")
    ax.set_ylabel("template Fnu [mJy]")
    ax.set_title("ALESS average SED template shape")
    ax.legend(fontsize=8)
    ax.set_xlim(1, 2000)
    ax.set_ylim(1e-4, 30)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_templates(df: pd.DataFrame, template: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    for sfr_min, color in zip(SFR_THRESHOLDS, colors):
        sub = df.loc[df["SFR_pop"] >= sfr_min]
        if len(sub) == 0:
            continue
        med_lir = float(np.nanmedian(sub["L_IR"]))
        med_z = float(np.nanmedian(sub["z_pop"]))
        lam, nu_lnu = scaled_template_nu_lnu(template, med_lir)
        m = (lam >= 1) & (lam <= 2000) & np.isfinite(nu_lnu) & (nu_lnu > 0)
        ax.loglog(
            lam[m],
            nu_lnu[m],
            color=color,
            lw=1.8,
            label=f"SFR >= {sfr_min}, med z={med_z:.2f}",
        )
        ax.axvline(250 / (1 + med_z), color=color, ls="--", lw=1.0, alpha=0.65)
    ax.set_xlabel("rest wavelength [um]")
    ax.set_ylabel(r"scaled template $\nu L_\nu$ [$L_\odot$]")
    ax.set_title("ALESS template scaled to pop-cosmos median L_IR")
    ax.legend(fontsize=8)
    ax.set_xlim(1, 2000)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_predicted_flux_distributions(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, band in zip(axes, [250, 850]):
        col = f"F{band}_aless_mjy"
        for sfr_min, color in zip(SFR_THRESHOLDS, ["#1f77b4", "#ff7f0e", "#d62728"]):
            arr = df.loc[
                (df["SFR_pop"] >= sfr_min) & np.isfinite(df[col]) & (df[col] > 0), col
            ]
            if len(arr) == 0:
                continue
            bins = np.logspace(
                np.log10(max(1e-4, arr.quantile(0.005))),
                np.log10(arr.quantile(0.995)),
                55,
            )
            ax.hist(
                arr,
                bins=bins,
                histtype="step",
                lw=1.8,
                density=True,
                label=f"SFR >= {sfr_min}",
                color=color,
            )
        ax.axvline(1, color="0.4", ls="--", lw=1)
        ax.axvline(3, color="0.2", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(f"ALESS-predicted observed F{band} [mJy]")
        ax.set_ylabel("density")
        ax.set_title(f"Predicted {band} um flux distribution")
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_wang_comparison(pred: pd.DataFrame, wang: pd.DataFrame, out_path: Path):
    matched = pred.merge(wang, on="ID", how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, band in zip(axes, [250, 850]):
        obs_col = f"F{band}"
        pred_col = f"F{band}_aless_mjy"
        snr_col = f"SNR{band}"
        mask = (
            (matched["SFR_pop"] >= 10)
            & np.isfinite(matched[obs_col])
            & np.isfinite(matched[pred_col])
            & np.isfinite(matched[snr_col])
            & (matched[snr_col] >= 3)
            & (matched[obs_col] > 0)
            & (matched[pred_col] > 0)
        )
        sub = matched.loc[mask, [obs_col, pred_col]]
        if len(sub) == 0:
            ax.set_title(f"{band} um: no SNR>=3 matches")
            continue
        hb = ax.hexbin(
            sub[obs_col],
            sub[pred_col],
            xscale="log",
            yscale="log",
            gridsize=35,
            mincnt=1,
            cmap="magma",
        )
        lo = np.nanmin([sub[obs_col].quantile(0.01), sub[pred_col].quantile(0.01)])
        hi = np.nanmax([sub[obs_col].quantile(0.99), sub[pred_col].quantile(0.99)])
        ax.plot([lo, hi], [lo, hi], color="cyan", ls="--", lw=1.2, label="1:1")
        med_ratio = np.nanmedian(np.log10(sub[pred_col]) - np.log10(sub[obs_col]))
        ax.set_xlabel(f"Wang observed F{band} [mJy]")
        ax.set_ylabel(f"ALESS-predicted F{band} [mJy]")
        ax.set_title(
            f"SFR >= 10, Wang SNR>=3, N={len(sub):,}\nmedian log(pred/obs)={med_ratio:+.2f}"
        )
        ax.legend(fontsize=8)
    fig.colorbar(hb, ax=axes, shrink=0.88, pad=0.02, label="counts per hexbin")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    template = load_aless_template(ALESS_PATH)
    pop = load_pop_with_lir()
    pred = add_aless_fluxes(pop, template)

    template_summary = {
        "n_rows": len(template),
        "lambda_min_um_positive": float(
            template.loc[template["fnu_average_mjy"] > 0, "lambda_um"].min()
        ),
        "lambda_max_um_positive": float(
            template.loc[template["fnu_average_mjy"] > 0, "lambda_um"].max()
        ),
        "fnu_average_24um_mjy": float(
            log_interp_positive(
                np.array([24.0]),
                template["lambda_um"].to_numpy(),
                template["fnu_average_mjy"].to_numpy(),
            )[0]
        ),
        "fnu_average_250um_mjy": float(
            log_interp_positive(
                np.array([250.0]),
                template["lambda_um"].to_numpy(),
                template["fnu_average_mjy"].to_numpy(),
            )[0]
        ),
        "fnu_average_850um_mjy": float(
            log_interp_positive(
                np.array([850.0]),
                template["lambda_um"].to_numpy(),
                template["fnu_average_mjy"].to_numpy(),
            )[0]
        ),
    }

    template_summary_path = OUT_DIR / "popcosmos_aless_template_summary.csv"
    pd.DataFrame([template_summary]).to_csv(template_summary_path, index=False)

    pred_summary = summarize_predicted_fluxes(pred)
    pred_summary_path = OUT_DIR / "popcosmos_aless_predicted_flux_summary.csv"
    pred_summary.to_csv(pred_summary_path, index=False)

    wang = load_wang()
    wang_summary = summarize_wang_comparison(pred, wang)
    wang_summary_path = OUT_DIR / "popcosmos_aless_wang_comparison_summary.csv"
    wang_summary.to_csv(wang_summary_path, index=False)

    plot_template(template, OUT_DIR / "popcosmos_aless_template_shape.png")
    plot_scaled_templates(
        pred, template, OUT_DIR / "popcosmos_aless_scaled_seds_by_sfr.png"
    )
    plot_predicted_flux_distributions(
        pred, OUT_DIR / "popcosmos_aless_predicted_flux_distributions.png"
    )
    plot_wang_comparison(
        pred, wang, OUT_DIR / "popcosmos_aless_wang_observed_vs_template_flux.png"
    )

    print("Saved:", template_summary_path)
    print("Saved:", pred_summary_path)
    print("Saved:", wang_summary_path)
    print()
    print("Predicted flux summary:")
    print(pred_summary.to_string(index=False))
    print()
    print("Wang comparison summary:")
    print(wang_summary.to_string(index=False))


if __name__ == "__main__":
    main()
