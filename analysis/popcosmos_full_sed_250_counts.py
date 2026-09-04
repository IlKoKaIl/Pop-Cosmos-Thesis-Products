"""Use Boris's full FSPS/pop-cosmos SEDs for FIR/sub-mm count checks.

This is the next step after the ALESS-template bridge test:

1. Pick the five highest-SFR pop-cosmos galaxies and plot their actual
   rest-frame FSPS SEDs.
2. Overlay ALESS scaled to the same L_IR, so we can see whether the real
   pop-cosmos high-SFR SEDs look SMG-like.
3. Convert the full pop-cosmos SED to observed 250/350/500/850um fluxes.
4. Compare those predicted fluxes to Wang observed fluxes and
   make simple cumulative number-count plots.

Notes:
- fsps_map_median_full.h5 stores wave_rest in Angstrom and spec_attenuated
  in Lsun/Hz.
- The band prediction uses Fnu_obs = (1 + z) Lnu_rest / (4 pi D_L^2).
"""

from pathlib import Path
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9
import astropy.units as u
from astropy.table import Table


CODE_DIR = Path(__file__).resolve().parent
NB_DIR = CODE_DIR.parent
ROOT = NB_DIR.parent
OUT_DIR = NB_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_SED_H5 = ROOT / "Boris work/fsps_map_median_full.h5"
POP_H5 = ROOT / "catalog data/real pop-cosmos data/mcmc_summaries.h5"
ALESS_PATH = ROOT / "aless_average_seds.dat.txt"
WANG_MASTER = ROOT / "catalog data/wang/master.dat.gz"
WANG_README = ROOT / "catalog data/wang/ReadMe.txt"
EXTERNAL_COUNTS_CSV = ROOT / "catalog data/external_number_counts/external_spire_number_counts_starter.csv"

L_SUN_W = 3.828e26
C_M_S = 299792458.0
MJY_TO_W_M2_HZ = 1e-29
OBS_BANDS_UM = [250, 350, 500, 850]
BRIGHT_FLUX_CUTS_MJY = [5, 10, 20, 50, 100]
WANG_BRIGHT_MIN_MJY = 5.0
SFR_CUTS = [10, 30, 100]
COSMOS_AREA_DEG2 = 2.0
SCATTER_LIMS_MJY = (1e-3, 1e3)
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


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
    lam = template["lambda_um"].to_numpy(float)
    fnu = template[fcol].to_numpy(float)
    mask = (
        np.isfinite(lam) & np.isfinite(fnu) & (lam >= 8.0) & (lam <= 1000.0) & (fnu > 0)
    )
    lam_m = lam[mask] * 1e-6
    nu_hz = C_M_S / lam_m
    order = np.argsort(nu_hz)
    return float(np.trapz(fnu[mask][order], nu_hz[order]))


def aless_nu_lnu_scaled(
    template: pd.DataFrame, lir_lsun: float, fcol: str = "fnu_average_mjy"
):
    lam_um = template["lambda_um"].to_numpy(float)
    fnu_shape = template[fcol].to_numpy(float)
    integral = template_integral_fnu_dnu(template, fcol)
    scale = lir_lsun * L_SUN_W / integral
    lnu_w_hz = scale * fnu_shape
    nu_hz = C_M_S / (lam_um * 1e-6)
    return lam_um, nu_hz * lnu_w_hz / L_SUN_W


def predict_aless_flux_mjy(
    template: pd.DataFrame,
    z,
    lir_lsun,
    lambda_obs_um,
    fcol: str = "fnu_average_mjy",
):
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


def flux_from_lnu_mjy(z, lnu_lsun_hz):
    dl_m = WMAP9.luminosity_distance(z).to_value(u.m)
    fnu_w_m2_hz = (1.0 + z) * lnu_lsun_hz * L_SUN_W / (4.0 * np.pi * dl_m**2)
    return fnu_w_m2_hz / MJY_TO_W_M2_HZ


def interpolate_lnu_at_rest_um(spec_batch, wave_um, lambda_rest_um):
    """Log-log interpolate row-wise Lnu at each row's rest wavelength."""
    lambda_rest_um = np.asarray(lambda_rest_um, dtype=float)
    idx = np.searchsorted(wave_um, lambda_rest_um, side="right") - 1
    idx = np.clip(idx, 0, len(wave_um) - 2)

    rows = np.arange(spec_batch.shape[0])
    x0 = wave_um[idx]
    x1 = wave_um[idx + 1]
    y0 = spec_batch[rows, idx]
    y1 = spec_batch[rows, idx + 1]

    good = (
        np.isfinite(lambda_rest_um)
        & np.isfinite(y0)
        & np.isfinite(y1)
        & (lambda_rest_um > 0)
        & (y0 > 0)
        & (y1 > 0)
        & (x0 > 0)
        & (x1 > x0)
    )

    out = np.full(spec_batch.shape[0], np.nan, dtype=float)
    frac = (np.log(lambda_rest_um[good]) - np.log(x0[good])) / (
        np.log(x1[good]) - np.log(x0[good])
    )
    out[good] = np.exp(np.log(y0[good]) + frac * (np.log(y1[good]) - np.log(y0[good])))
    return out


def load_wang_bands() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = Table.read(WANG_MASTER, format="ascii.cds", readme=WANG_README)

    cols = ["ID"]
    for band in OBS_BANDS_UM:
        cols.extend([f"F{band}", f"s_F{band}"])
    df = tab.to_pandas()[cols].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["ID"] > 0].copy()
    for band in OBS_BANDS_UM:
        df[f"SNR{band}"] = df[f"F{band}"] / df[f"s_F{band}"].replace(0, np.nan)
    return df


def load_pop_metadata() -> pd.DataFrame:
    with h5py.File(POP_H5, "r") as f:
        out = pd.DataFrame(
            {
                "ID": f["metadata/index_farmer"][:].astype(np.int64),
                "log10SFR_pop": f["pop-cosmos/log10SFR"][:, 2],
                "log10M_pop": f["pop-cosmos/log10M_remain"][:, 2],
                "dust2_pop": f["pop-cosmos/dust2"][:, 2],
                "lnfAGN_pop": f["pop-cosmos/lnfAGN"][:, 2],
                "lntauAGN_pop": f["pop-cosmos/lntauAGN"][:, 2],
                "dust_index_pop": f["pop-cosmos/dust_index"][:, 2],
            }
        )
    out["SFR_pop"] = 10 ** out["log10SFR_pop"]
    out["fAGN_pop"] = np.exp(out["lnfAGN_pop"])
    return out


def compute_band_predictions(batch_size=2048) -> pd.DataFrame:
    """Process the 19 GB SED file in row batches and cache compact outputs."""
    pop = load_pop_metadata()

    with h5py.File(FULL_SED_H5, "r") as f:
        n = f["index"].shape[0]
        wave_um = f["wave_rest"][:] / 1e4
        ids = f["index"][:].astype(np.int64)
        rows = f["row"][:].astype(np.int64)
        z = f["z"][:]
        lir = f["L_IR"][:]
        done = f["done"][:].astype(bool)

        fluxes = {band: np.full(n, np.nan, dtype=float) for band in OBS_BANDS_UM}
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            spec = f["spec_attenuated"][start:end, :]
            z_batch = z[start:end]
            valid = done[start:end] & np.isfinite(z_batch) & (z_batch > 0)
            for band in OBS_BANDS_UM:
                rest_um = band / (1.0 + z_batch)
                lnu = interpolate_lnu_at_rest_um(spec, wave_um, rest_um)
                fluxes[band][start:end][valid] = flux_from_lnu_mjy(
                    z_batch[valid], lnu[valid]
                )

    out = pd.DataFrame(
        {
            "row": rows,
            "ID": ids,
            "z_pop": z,
            "L_IR": lir,
            "done": done,
        }
    )
    for band in OBS_BANDS_UM:
        out[f"F{band}_fsps_mjy"] = fluxes[band]
    out = out.merge(pop, on="ID", how="left", validate="one_to_one")
    out["log10LIR"] = np.where(out["L_IR"] > 0, np.log10(out["L_IR"]), np.nan)

    aless = load_aless_template(ALESS_PATH)
    for band in OBS_BANDS_UM:
        out[f"F{band}_aless_mjy"] = predict_aless_flux_mjy(
            aless,
            out["z_pop"].to_numpy(float),
            out["L_IR"].to_numpy(float),
            lambda_obs_um=band,
        )
    return out


def cumulative_counts(flux, grid):
    flux = np.asarray(flux, dtype=float)
    flux = flux[np.isfinite(flux) & (flux > 0)]
    return np.array([np.sum(flux >= g) for g in grid], dtype=int)


def spearman_simple(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return float(x[ok].rank().corr(y[ok].rank()))


def read_seds_for_rows(rows):
    rows = np.asarray(rows, dtype=int)
    with h5py.File(FULL_SED_H5, "r") as f:
        wave_um = f["wave_rest"][:] / 1e4
        sort_order = np.argsort(rows)
        spec_sorted = f["spec_attenuated"][rows[sort_order], :]
        spec = spec_sorted[np.argsort(sort_order)]
    return wave_um, spec


def pick_sfr_rank_samples(pred: pd.DataFrame, n_each=10) -> pd.DataFrame:
    base = pred[
        pred["done"]
        & np.isfinite(pred["SFR_pop"])
        & np.isfinite(pred["L_IR"])
        & (pred["SFR_pop"] > 0)
        & (pred["L_IR"] > 0)
    ].copy()
    base = base.sort_values("SFR_pop").reset_index(drop=True)

    low = base.head(n_each).copy()
    mid_start = max((len(base) // 2) - (n_each // 2), 0)
    median = base.iloc[mid_start : mid_start + n_each].copy()
    high = base.tail(n_each).iloc[::-1].copy()

    low["sample"] = "low SFR"
    median["sample"] = "median SFR"
    high["sample"] = "top SFR"
    return pd.concat([high, median, low], ignore_index=True)


def plot_sfr_rank_sed_samples(pred: pd.DataFrame, template: pd.DataFrame):
    samples = pick_sfr_rank_samples(pred, n_each=10)
    wave_um, spec = read_seds_for_rows(samples["row"].to_numpy(int))
    nu_hz = C_M_S / (wave_um * 1e-6)
    nu_lnu = spec * nu_hz[None, :]

    rows = []
    for (_, row), sed_nu_lnu in zip(samples.iterrows(), nu_lnu):
        peak_mask = (
            np.isfinite(wave_um)
            & np.isfinite(sed_nu_lnu)
            & (wave_um >= 30)
            & (wave_um <= 300)
            & (sed_nu_lnu > 0)
        )
        peak_um = np.nan
        if np.any(peak_mask):
            peak_um = wave_um[peak_mask][np.nanargmax(sed_nu_lnu[peak_mask])]
        rows.append(
            {
                "sample": row["sample"],
                "ID": int(row["ID"]),
                "z_pop": row["z_pop"],
                "SFR_pop": row["SFR_pop"],
                "log10LIR": row["log10LIR"],
                "peak_um_30_300": peak_um,
                "fAGN_pop": row["fAGN_pop"],
            }
        )
    sample_table = pd.DataFrame(rows)
    sample_table.to_csv(OUT_DIR / "popcosmos_full_sed_sfr_rank_sample_table.csv", index=False)

    colors = {"top SFR": "tab:red", "median SFR": "tab:green", "low SFR": "tab:blue"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=False)
    for ax, sample_name in zip(axes, ["top SFR", "median SFR", "low SFR"]):
        idx = samples.index[samples["sample"] == sample_name].to_numpy()
        for j in idx:
            sed_nu_lnu = nu_lnu[j]
            m = (
                np.isfinite(wave_um)
                & np.isfinite(sed_nu_lnu)
                & (wave_um >= 5)
                & (wave_um <= 1000)
                & (sed_nu_lnu > 0)
            )
            label = f"ID {int(samples.loc[j, 'ID'])}" if sample_name == "top SFR" else None
            ax.plot(wave_um[m], sed_nu_lnu[m], color=colors[sample_name], alpha=0.65, lw=1.4, label=label)

        sub = sample_table[sample_table["sample"] == sample_name]
        med_lir = 10 ** np.nanmedian(sub["log10LIR"])
        a_lam, a_nu_lnu = aless_nu_lnu_scaled(template, med_lir)
        am = (
            np.isfinite(a_lam)
            & np.isfinite(a_nu_lnu)
            & (a_lam >= 5)
            & (a_lam <= 1000)
            & (a_nu_lnu > 0)
        )
        ax.plot(a_lam[am], a_nu_lnu[am], "k--", lw=1.8, label="ALESS at median L_IR")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{sample_name}\nmedian SFR={np.nanmedian(sub['SFR_pop']):.2g}, "
            f"peak={np.nanmedian(sub['peak_um_30_300']):.0f}um"
        )
        ax.set_xlabel("rest wavelength (um)")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel(r"$\nu L_\nu$ ($L_\odot$)")
    axes[0].legend(fontsize=7)
    fig.suptitle("FSPS SEDs by SFR rank: top, median, and low-SFR samples")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_sfr_rank_samples.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, sample_name in zip(axes, ["top SFR", "median SFR", "low SFR"]):
        idx = samples.index[samples["sample"] == sample_name].to_numpy()
        for j in idx:
            sed_nu_lnu = nu_lnu[j]
            m = (
                np.isfinite(wave_um)
                & np.isfinite(sed_nu_lnu)
                & (wave_um >= 8)
                & (wave_um <= 1000)
                & (sed_nu_lnu > 0)
            )
            norm = np.nanmax(sed_nu_lnu[m]) if np.any(m) else np.nan
            if np.isfinite(norm) and norm > 0:
                ax.plot(wave_um[m], sed_nu_lnu[m] / norm, color=colors[sample_name], alpha=0.65, lw=1.4)
        a_lam, a_nu_lnu = aless_nu_lnu_scaled(template, 1e12)
        am = (
            np.isfinite(a_lam)
            & np.isfinite(a_nu_lnu)
            & (a_lam >= 8)
            & (a_lam <= 1000)
            & (a_nu_lnu > 0)
        )
        ax.plot(a_lam[am], a_nu_lnu[am] / np.nanmax(a_nu_lnu[am]), "k--", lw=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(sample_name)
        ax.set_xlabel("rest wavelength (um)")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("shape only, normalized to peak")
    fig.suptitle("Shape-only SED comparison by SFR rank")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_sfr_rank_samples_normalized.png", dpi=180)
    plt.close(fig)

    return sample_table


def deterministic_sample(df: pd.DataFrame, n=1000) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=42).copy()


def sed_percentiles_for_sample(sample: pd.DataFrame):
    wave_um, spec = read_seds_for_rows(sample["row"].to_numpy(int))
    nu_hz = C_M_S / (wave_um * 1e-6)
    nu_lnu = spec * nu_hz[None, :]
    lir = sample["L_IR"].to_numpy(float)
    normed = nu_lnu / lir[:, None]
    normed[~np.isfinite(normed)] = np.nan
    normed[normed <= 0] = np.nan
    return wave_um, np.nanpercentile(normed, [16, 50, 84], axis=0)


def summarize_sed_sample(name: str, sample: pd.DataFrame, wave_um, pct) -> dict:
    med = pct[1]
    peak_mask = (
        np.isfinite(wave_um)
        & np.isfinite(med)
        & (wave_um >= 30)
        & (wave_um <= 300)
        & (med > 0)
    )
    peak_um = np.nan
    if np.any(peak_mask):
        peak_um = wave_um[peak_mask][np.nanargmax(med[peak_mask])]

    def interp_med(lam_um):
        return log_interp_positive(np.array([lam_um]), wave_um, med)[0]

    v15 = interp_med(15.0)
    v100 = interp_med(100.0)
    hot_ratio = np.nan
    if np.isfinite(v15) and np.isfinite(v100) and v15 > 0 and v100 > 0:
        hot_ratio = np.log10(v15 / v100)

    return {
        "sample": name,
        "N": len(sample),
        "median_SFR_pop": np.nanmedian(sample["SFR_pop"]),
        "median_log10LIR": np.nanmedian(sample["log10LIR"]),
        "median_fAGN_pop": np.nanmedian(sample["fAGN_pop"]),
        "peak_um_30_300": peak_um,
        "log10_nuLnu15_over_100": hot_ratio,
    }


def plot_population_median_seds(pred: pd.DataFrame):
    """Median SED shapes for broad SFR and AGN-parameter slices."""
    base = pred[
        pred["done"]
        & np.isfinite(pred["row"])
        & np.isfinite(pred["SFR_pop"])
        & np.isfinite(pred["L_IR"])
        & np.isfinite(pred["fAGN_pop"])
        & (pred["SFR_pop"] > 0)
        & (pred["L_IR"] > 0)
    ].copy()

    sfr_samples = [
        ("low SFR\n0.1-1", (base["SFR_pop"] >= 0.1) & (base["SFR_pop"] < 1.0), OKABE_ITO["blue"]),
        ("normal SFR\n1-10", (base["SFR_pop"] >= 1.0) & (base["SFR_pop"] < 10.0), OKABE_ITO["green"]),
        ("high SFR\n30-300", (base["SFR_pop"] >= 30.0) & (base["SFR_pop"] < 300.0), OKABE_ITO["orange"]),
        ("extreme SFR\n>1000", base["SFR_pop"] >= 1000.0, OKABE_ITO["vermillion"]),
    ]

    summary_rows = []
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for name, mask, color in sfr_samples:
        sample = deterministic_sample(base.loc[mask], n=1200)
        if len(sample) < 5:
            continue
        wave_um, pct = sed_percentiles_for_sample(sample)
        summary_rows.append(summarize_sed_sample(name.replace("\n", " "), sample, wave_um, pct))
        m = (
            np.isfinite(wave_um)
            & np.isfinite(pct[1])
            & (wave_um >= 5)
            & (wave_um <= 1000)
            & (pct[1] > 0)
        )
        ax.plot(wave_um[m], pct[1][m], color=color, lw=2.2, label=f"{name}, N={len(sample)}")
        ax.fill_between(wave_um[m], pct[0][m], pct[2][m], color=color, alpha=0.16, lw=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rest wavelength (um)")
    ax.set_ylabel(r"shape: $\nu L_\nu / L_{\rm IR}$")
    ax.set_title("Median pop-cosmos SED shape by SFR slice")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_median_sfr_seds.png", dpi=180)
    plt.close(fig)

    active = base[base["SFR_pop"] >= 10].copy()
    q20, q80 = np.nanpercentile(active["fAGN_pop"], [20, 80])
    agn_samples = [
        (f"low fAGN\n< {q20:.3g}", active["fAGN_pop"] <= q20, OKABE_ITO["blue"]),
        (f"high fAGN\n> {q80:.3g}", active["fAGN_pop"] >= q80, OKABE_ITO["vermillion"]),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for name, mask, color in agn_samples:
        sample = deterministic_sample(active.loc[mask], n=1200)
        if len(sample) < 5:
            continue
        wave_um, pct = sed_percentiles_for_sample(sample)
        summary_rows.append(summarize_sed_sample(name.replace("\n", " "), sample, wave_um, pct))
        m = (
            np.isfinite(wave_um)
            & np.isfinite(pct[1])
            & (wave_um >= 3)
            & (wave_um <= 1000)
            & (pct[1] > 0)
        )
        ax.plot(wave_um[m], pct[1][m], color=color, lw=2.2, label=f"{name}, N={len(sample)}")
        ax.fill_between(wave_um[m], pct[0][m], pct[2][m], color=color, alpha=0.16, lw=0)

    ax.axvspan(3, 30, color="0.85", alpha=0.25, label="mid-IR / hot dust region")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rest wavelength (um)")
    ax.set_ylabel(r"shape: $\nu L_\nu / L_{\rm IR}$")
    ax.set_title("Median SED shape for high vs low model AGN-like parameter, SFR > 10")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_agn_parameter_median_seds.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "popcosmos_full_sed_median_sed_summary.csv", index=False)
    return summary


def plot_top5_seds(pred: pd.DataFrame, template: pd.DataFrame):
    top = pred[np.isfinite(pred["SFR_pop"]) & pred["done"]].nlargest(5, "SFR_pop")

    with h5py.File(FULL_SED_H5, "r") as f:
        wave_um = f["wave_rest"][:] / 1e4
        top_rows = top["row"].to_numpy(int)
        sort_order = np.argsort(top_rows)
        spec_sorted = f["spec_attenuated"][top_rows[sort_order], :]
        spec = spec_sorted[np.argsort(sort_order)]

    nu_hz = C_M_S / (wave_um * 1e-6)
    nu_lnu = spec * nu_hz[None, :]

    rows = []
    for i, ((_, row), sed_nu_lnu) in enumerate(zip(top.iterrows(), nu_lnu)):
        peak_mask = (
            np.isfinite(wave_um)
            & np.isfinite(sed_nu_lnu)
            & (wave_um >= 30)
            & (wave_um <= 300)
            & (sed_nu_lnu > 0)
        )
        peak_um = np.nan
        if np.any(peak_mask):
            peak_um = wave_um[peak_mask][np.nanargmax(sed_nu_lnu[peak_mask])]

        out_row = {
            "rank": i + 1,
            "row": int(row["row"]),
            "ID": int(row["ID"]),
            "z_pop": row["z_pop"],
            "SFR_pop": row["SFR_pop"],
            "log10SFR_pop": row["log10SFR_pop"],
            "L_IR": row["L_IR"],
            "log10LIR": row["log10LIR"],
            "peak_um_30_300": peak_um,
            "dust2_pop": row["dust2_pop"],
            "lnfAGN_pop": row["lnfAGN_pop"],
            "fAGN_pop": row["fAGN_pop"],
            "lntauAGN_pop": row["lntauAGN_pop"],
        }
        for band in OBS_BANDS_UM:
            out_row[f"F{band}_fsps_mjy"] = row[f"F{band}_fsps_mjy"]
            out_row[f"F{band}_aless_mjy"] = row[f"F{band}_aless_mjy"]
            out_row[f"fsps_over_aless_F{band}"] = (
                row[f"F{band}_fsps_mjy"] / row[f"F{band}_aless_mjy"]
            )
        rows.append(out_row)
    top_table = pd.DataFrame(rows)
    top_table.to_csv(OUT_DIR / "popcosmos_full_sed_top5_sfr_table.csv", index=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for color, ((_, row), sed_nu_lnu) in zip(colors, zip(top.iterrows(), nu_lnu)):
        label = f"ID {int(row['ID'])}, SFR={row['SFR_pop']:.0f}"
        m = (
            np.isfinite(wave_um)
            & np.isfinite(sed_nu_lnu)
            & (wave_um >= 5)
            & (wave_um <= 1000)
            & (sed_nu_lnu > 0)
        )
        ax.plot(wave_um[m], sed_nu_lnu[m], color=color, lw=2, label=label)

        a_lam, a_nu_lnu = aless_nu_lnu_scaled(template, row["L_IR"])
        am = (
            np.isfinite(a_lam)
            & np.isfinite(a_nu_lnu)
            & (a_lam >= 5)
            & (a_lam <= 1000)
            & (a_nu_lnu > 0)
        )
        ax.plot(a_lam[am], a_nu_lnu[am], color=color, lw=1.4, ls="--", alpha=0.75)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rest wavelength (um)")
    ax.set_ylabel(r"$\nu L_\nu$ ($L_\odot$)")
    ax.set_title("Top 5 pop-cosmos SFR objects: full FSPS SED vs ALESS scaled to same L_IR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_top5_vs_aless.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for color, ((_, row), sed_nu_lnu) in zip(colors, zip(top.iterrows(), nu_lnu)):
        m = (
            np.isfinite(wave_um)
            & np.isfinite(sed_nu_lnu)
            & (wave_um >= 8)
            & (wave_um <= 1000)
            & (sed_nu_lnu > 0)
        )
        norm = np.nanmax(sed_nu_lnu[m])
        ax.plot(
            wave_um[m],
            sed_nu_lnu[m] / norm,
            color=color,
            lw=2,
            label=f"FSPS ID {int(row['ID'])}",
        )

    a_lam, a_nu_lnu = aless_nu_lnu_scaled(template, 1e12)
    am = (
        np.isfinite(a_lam)
        & np.isfinite(a_nu_lnu)
        & (a_lam >= 8)
        & (a_lam <= 1000)
        & (a_nu_lnu > 0)
    )
    ax.plot(a_lam[am], a_nu_lnu[am] / np.nanmax(a_nu_lnu[am]), "k--", lw=2.4, label="ALESS shape")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rest wavelength (um)")
    ax.set_ylabel("shape only, normalized to peak")
    ax.set_title("SED shape check: do the highest-SFR FSPS SEDs look ALESS-like?")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_top5_shape_normalized.png", dpi=180)
    plt.close(fig)

    return top_table


def binned_log_ratio(
    df: pd.DataFrame,
    x_col: str,
    num_col: str,
    den_col: str,
    bins,
    label: str,
    band: int,
) -> list[dict]:
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (
            np.isfinite(df[x_col])
            & np.isfinite(df[num_col])
            & np.isfinite(df[den_col])
            & (df[x_col] >= lo)
            & (df[x_col] < hi)
            & (df[num_col] > 0)
            & (df[den_col] > 0)
        )
        sub = df.loc[m]
        if len(sub) < 10:
            continue
        ratio = np.log10(sub[num_col].to_numpy(float) / sub[den_col].to_numpy(float))
        rows.append(
            {
                "band_um": band,
                "model": label,
                "x_low_mjy": lo,
                "x_high_mjy": hi,
                "x_median_mjy": np.nanmedian(sub[x_col]),
                "N": len(sub),
                "median_log10_ratio": np.nanmedian(ratio),
                "p16_log10_ratio": np.nanpercentile(ratio, 16),
                "p84_log10_ratio": np.nanpercentile(ratio, 84),
            }
        )
    return rows


def make_wang_brightcut_bias_outputs(detected_by_band: dict[int, pd.DataFrame]):
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=True)
    for ax, band in zip(axes.ravel(), OBS_BANDS_UM):
        obs_col = f"F{band}"
        fsps_col = f"F{band}_fsps_mjy"
        aless_col = f"F{band}_aless_mjy"
        det = detected_by_band[band]
        det = det[det[obs_col] >= WANG_BRIGHT_MIN_MJY].copy()
        if len(det) == 0:
            ax.set_title(f"{band}um: no bright Wang detections")
            continue

        upper = max(6.0, np.nanpercentile(det[obs_col], 99.5))
        bins = np.logspace(np.log10(WANG_BRIGHT_MIN_MJY), np.log10(upper), 8)
        rows.extend(binned_log_ratio(det, obs_col, fsps_col, obs_col, bins, "FSPS", band))
        rows.extend(binned_log_ratio(det, obs_col, aless_col, obs_col, bins, "ALESS", band))

        for model, color in [("FSPS", OKABE_ITO["blue"]), ("ALESS", OKABE_ITO["orange"])]:
            model_rows = [r for r in rows if r["band_um"] == band and r["model"] == model]
            if not model_rows:
                continue
            d = pd.DataFrame(model_rows)
            ax.plot(
                d["x_median_mjy"],
                d["median_log10_ratio"],
                marker="o",
                lw=2,
                color=color,
                label=model,
            )
            ax.fill_between(
                d["x_median_mjy"],
                d["p16_log10_ratio"],
                d["p84_log10_ratio"],
                color=color,
                alpha=0.18,
                lw=0,
            )

        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(-2.2, 1.6)
        ax.set_title(f"{band}um, Wang F >= {WANG_BRIGHT_MIN_MJY:g} mJy")
        ax.set_xlabel(f"Wang observed F{band} (mJy)")
        ax.grid(alpha=0.25, which="both")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}(\mathrm{model}/\mathrm{Wang})$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bright Wang detections: median model residual in observed-flux slices")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_wang_brightcut_flux_bias.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "popcosmos_full_sed_wang_brightcut_flux_bias.csv", index=False)
    return summary


def make_model_model_binned_bias_outputs(matched: pd.DataFrame):
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=True)
    for ax, band in zip(axes.ravel(), OBS_BANDS_UM):
        fsps_col = f"F{band}_fsps_mjy"
        aless_col = f"F{band}_aless_mjy"
        m = (
            np.isfinite(matched[fsps_col])
            & np.isfinite(matched[aless_col])
            & (matched[fsps_col] > 0)
            & (matched[aless_col] > 0)
            & (matched[aless_col] >= 1e-3)
        )
        sub = matched.loc[m].copy()
        bins = np.logspace(-3, 3, 16)
        rows.extend(binned_log_ratio(sub, aless_col, fsps_col, aless_col, bins, "FSPS/ALESS", band))
        band_rows = [r for r in rows if r["band_um"] == band]
        if band_rows:
            d = pd.DataFrame(band_rows)
            ax.plot(
                d["x_median_mjy"],
                d["median_log10_ratio"],
                marker="o",
                lw=2,
                color=OKABE_ITO["purple"],
            )
            ax.fill_between(
                d["x_median_mjy"],
                d["p16_log10_ratio"],
                d["p84_log10_ratio"],
                color=OKABE_ITO["purple"],
                alpha=0.18,
                lw=0,
            )
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(-1.2, 1.8)
        ax.set_title(f"{band}um")
        ax.set_xlabel(f"ALESS-predicted F{band} (mJy)")
        ax.grid(alpha=0.25, which="both")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}(\mathrm{FSPS}/\mathrm{ALESS})$")
    fig.suptitle("Model-to-model sliced bias: FSPS vs ALESS in predicted-flux bins")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_fsps_vs_aless_binned_bias.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "popcosmos_full_sed_fsps_vs_aless_binned_bias.csv", index=False)
    return summary


def make_external_counts_overlay(matched: pd.DataFrame, wang: pd.DataFrame):
    if not EXTERNAL_COUNTS_CSV.exists():
        return pd.DataFrame()

    external = pd.read_csv(EXTERNAL_COUNTS_CSV)
    external = external[
        np.isfinite(external["flux_mjy"])
        & np.isfinite(external["integral_N_gt_S_per_deg2"])
        & (external["integral_N_gt_S_per_deg2"] > 0)
    ].copy()
    external.to_csv(OUT_DIR / "popcosmos_full_sed_external_counts_used.csv", index=False)

    grid = np.logspace(np.log10(WANG_BRIGHT_MIN_MJY), np.log10(300), 160)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, band in zip(axes, [250, 350, 500]):
        obs_col = f"F{band}"
        snr_col = f"SNR{band}"
        fsps_col = f"F{band}_fsps_mjy"
        aless_col = f"F{band}_aless_mjy"
        wang_det_flux = wang.loc[
            np.isfinite(wang[obs_col])
            & np.isfinite(wang[snr_col])
            & (wang[snr_col] >= 3)
            & (wang[obs_col] > 0),
            obs_col,
        ]

        ax.plot(
            grid,
            cumulative_counts(matched[fsps_col], grid) / COSMOS_AREA_DEG2,
            label="pop-cosmos FSPS",
            color=OKABE_ITO["blue"],
            lw=2,
        )
        ax.plot(
            grid,
            cumulative_counts(matched[aless_col], grid) / COSMOS_AREA_DEG2,
            label="pop-cosmos + ALESS",
            color=OKABE_ITO["orange"],
            lw=2,
            ls="--",
        )
        ax.plot(
            grid,
            cumulative_counts(wang_det_flux, grid) / COSMOS_AREA_DEG2,
            label="Wang SNR>=3",
            color=OKABE_ITO["black"],
            lw=2,
        )

        for paper, marker, color in [
            ("Clements et al.", "s", OKABE_ITO["green"]),
            ("Oliver et al.", "D", OKABE_ITO["vermillion"]),
            ("Pearson et al.", "^", OKABE_ITO["purple"]),
        ]:
            sub = external[(external["band_um"] == band) & (external["paper"] == paper)]
            if len(sub) == 0:
                continue
            yerr = sub["integral_err_per_deg2"].to_numpy(float)
            yerr[~np.isfinite(yerr)] = 0
            ax.errorbar(
                sub["flux_mjy"],
                sub["integral_N_gt_S_per_deg2"],
                yerr=yerr,
                fmt=marker,
                ms=5,
                color=color,
                mec="white",
                mew=0.5,
                lw=1,
                alpha=0.9,
                label=paper,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{band}um")
        ax.set_xlabel("observed flux cut (mJy)")
        ax.grid(alpha=0.25, which="both")

    axes[0].set_ylabel(r"cumulative counts $N(>S)$ per deg$^2$")
    axes[0].legend(fontsize=8)
    fig.suptitle("pop-cosmos/Wang counts compared with starter external SPIRE counts")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_external_counts_overlay.png", dpi=180)
    plt.close(fig)

    return external


def make_counts_and_wang_outputs(pred: pd.DataFrame):
    wang = load_wang_bands()
    matched = pred.merge(wang, on="ID", how="inner")

    count_rows = []
    summary_rows = []
    detected_by_band = {}

    for band in OBS_BANDS_UM:
        obs_col = f"F{band}"
        snr_col = f"SNR{band}"
        fsps_col = f"F{band}_fsps_mjy"
        aless_col = f"F{band}_aless_mjy"

        detected = matched[
            np.isfinite(matched[obs_col])
            & np.isfinite(matched[snr_col])
            & (matched[snr_col] >= 3)
            & (matched[obs_col] > 0)
            & np.isfinite(matched[fsps_col])
            & (matched[fsps_col] > 0)
            & np.isfinite(matched[aless_col])
            & (matched[aless_col] > 0)
        ].copy()
        detected_by_band[band] = detected

        summary_rows.append(
            {
                "band_um": band,
                "N_wang_matched_total": len(matched),
                "N_wang_snr3_with_fsps": len(detected),
                "median_wang_mjy": np.nanmedian(detected[obs_col]),
                "median_fsps_mjy": np.nanmedian(detected[fsps_col]),
                "median_aless_mjy": np.nanmedian(detected[aless_col]),
                "median_log10_fsps_over_wang": np.nanmedian(
                    np.log10(detected[fsps_col] / detected[obs_col])
                ),
                "median_log10_aless_over_wang": np.nanmedian(
                    np.log10(detected[aless_col] / detected[obs_col])
                ),
                "mad_log10_fsps_over_wang": np.nanmedian(
                    np.abs(
                        np.log10(detected[fsps_col] / detected[obs_col])
                        - np.nanmedian(np.log10(detected[fsps_col] / detected[obs_col]))
                    )
                ),
                "spearman_wang_vs_fsps": spearman_simple(detected[obs_col], detected[fsps_col]),
                "spearman_wang_vs_aless": spearman_simple(detected[obs_col], detected[aless_col]),
            }
        )

        for cut in BRIGHT_FLUX_CUTS_MJY:
            row = {
                "band_um": band,
                "flux_cut_mjy": cut,
                "N_pop_fsps_all": int(np.sum(pred[fsps_col] >= cut)),
                "N_pop_aless_all": int(np.sum(pred[aless_col] >= cut)),
                "N_pop_fsps_wangmatched": int(np.sum(matched[fsps_col] >= cut)),
                "N_pop_aless_wangmatched": int(np.sum(matched[aless_col] >= cut)),
                "N_wang_observed_finite": int(
                    np.sum(np.isfinite(wang[obs_col]) & (wang[obs_col] >= cut))
                ),
                "N_wang_snr3_observed": int(
                    np.sum(
                        np.isfinite(wang[obs_col])
                        & np.isfinite(wang[snr_col])
                        & (wang[snr_col] >= 3)
                        & (wang[obs_col] >= cut)
                    )
                ),
            }
            row["surface_density_pop_fsps_wangmatched_per_deg2"] = (
                row["N_pop_fsps_wangmatched"] / COSMOS_AREA_DEG2
            )
            row["surface_density_pop_aless_wangmatched_per_deg2"] = (
                row["N_pop_aless_wangmatched"] / COSMOS_AREA_DEG2
            )
            row["surface_density_wang_snr3_per_deg2"] = (
                row["N_wang_snr3_observed"] / COSMOS_AREA_DEG2
            )
            for sfr_cut in SFR_CUTS:
                m = pred["SFR_pop"] >= sfr_cut
                row[f"N_pop_fsps_SFRgt{sfr_cut}"] = int(np.sum(m & (pred[fsps_col] >= cut)))
            count_rows.append(row)

    count_summary = pd.DataFrame(count_rows)
    count_summary.to_csv(OUT_DIR / "popcosmos_full_sed_multiband_counts_summary.csv", index=False)
    count_summary[count_summary["band_um"] == 250].to_csv(
        OUT_DIR / "popcosmos_full_sed_250_counts_summary.csv", index=False
    )

    wang_summary = pd.DataFrame(summary_rows)
    wang_summary.to_csv(OUT_DIR / "popcosmos_full_sed_wang_multiband_summary.csv", index=False)
    wang_summary[wang_summary["band_um"] == 250].to_csv(
        OUT_DIR / "popcosmos_full_sed_wang_250_summary.csv", index=False
    )

    brightcut_bias = make_wang_brightcut_bias_outputs(detected_by_band)

    grid = np.logspace(np.log10(WANG_BRIGHT_MIN_MJY), 2.5, 150)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, band in zip(axes.ravel(), OBS_BANDS_UM):
        obs_col = f"F{band}"
        snr_col = f"SNR{band}"
        fsps_col = f"F{band}_fsps_mjy"
        aless_col = f"F{band}_aless_mjy"
        wang_det_flux = wang.loc[
            np.isfinite(wang[obs_col])
            & np.isfinite(wang[snr_col])
            & (wang[snr_col] >= 3),
            obs_col,
        ]
        wang_finite_flux = wang.loc[np.isfinite(wang[obs_col]) & (wang[obs_col] > 0), obs_col]

        ax.plot(
            grid,
            cumulative_counts(matched[fsps_col], grid),
            label="FSPS, Wang-matched IDs",
            lw=2,
        )
        ax.plot(
            grid,
            cumulative_counts(matched[aless_col], grid),
            label="ALESS, Wang-matched IDs",
            lw=2,
            ls="--",
        )
        ax.plot(
            grid,
            cumulative_counts(wang_det_flux, grid),
            label="Wang observed, SNR>=3",
            lw=2,
            color="black",
        )
        ax.plot(
            grid,
            cumulative_counts(wang_finite_flux, grid),
            label="Wang finite flux, no SNR cut",
            lw=1.5,
            color="0.5",
            ls=":",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{band}um")
        ax.grid(alpha=0.25, which="both")
    for ax in axes[-1, :]:
        ax.set_xlabel("observed flux cut (mJy)")
    for ax in axes[:, 0]:
        ax.set_ylabel("raw cumulative count N(>S)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bright FIR/sub-mm number counts, Wang-matched comparison")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_multiband_counts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(OBS_BANDS_UM), figsize=(17, 8), sharex=False, sharey=False)
    for col, band in enumerate(OBS_BANDS_UM):
        detected = detected_by_band[band]
        obs_col = f"F{band}"
        for row_i, (pred_col, name) in enumerate(
            [(f"F{band}_fsps_mjy", "FSPS"), (f"F{band}_aless_mjy", "ALESS")]
        ):
            ax = axes[row_i, col]
            ax.scatter(
                detected[obs_col],
                detected[pred_col],
                s=8,
                alpha=0.25,
                edgecolors="none",
            )
            finite = (
                np.isfinite(detected[obs_col])
                & np.isfinite(detected[pred_col])
                & (detected[obs_col] > 0)
                & (detected[pred_col] > 0)
            )
            if np.any(finite):
                ax.plot(SCATTER_LIMS_MJY, SCATTER_LIMS_MJY, "k--", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(SCATTER_LIMS_MJY)
            ax.set_ylim(SCATTER_LIMS_MJY)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{name} {band}um")
            ax.set_xlabel(f"Wang F{band} (mJy)")
            if col == 0:
                ax.set_ylabel("predicted flux (mJy)")
            ax.grid(alpha=0.25, which="both")
    fig.suptitle("Wang detections vs pop-cosmos predictions, per band")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_wang_multiband_compare.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), sharex=True, sharey=True)
    for ax, band in zip(axes.ravel(), OBS_BANDS_UM):
        fsps_col = f"F{band}_fsps_mjy"
        aless_col = f"F{band}_aless_mjy"
        m = (
            np.isfinite(matched[fsps_col])
            & np.isfinite(matched[aless_col])
            & (matched[fsps_col] > 0)
            & (matched[aless_col] > 0)
        )
        ax.hexbin(
            matched.loc[m, aless_col],
            matched.loc[m, fsps_col],
            xscale="log",
            yscale="log",
            gridsize=55,
            mincnt=1,
            cmap="cividis",
            bins="log",
        )
        ax.plot(SCATTER_LIMS_MJY, SCATTER_LIMS_MJY, "w--", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(SCATTER_LIMS_MJY)
        ax.set_ylim(SCATTER_LIMS_MJY)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{band}um")
        ax.grid(alpha=0.2, which="both")
    for ax in axes[-1, :]:
        ax.set_xlabel("ALESS-predicted flux (mJy)")
    for ax in axes[:, 0]:
        ax.set_ylabel("FSPS-predicted flux (mJy)")
    fig.suptitle("Direct model comparison: FSPS SED prediction vs ALESS template prediction")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "popcosmos_full_sed_fsps_vs_aless_predictions.png", dpi=180)
    plt.close(fig)

    model_model_bias = make_model_model_binned_bias_outputs(matched)
    external_counts = make_external_counts_overlay(matched, wang)

    return count_summary, wang_summary, brightcut_bias, model_model_bias, external_counts


def make_extreme_sfr_outputs(pred: pd.DataFrame):
    rows = []
    for sfr_cut in [300, 500, 1000]:
        sub = pred[
            pred["done"]
            & np.isfinite(pred["SFR_pop"])
            & (pred["SFR_pop"] >= sfr_cut)
            & np.isfinite(pred["z_pop"])
        ].copy()
        row = {
            "SFR_cut": sfr_cut,
            "N": len(sub),
            "median_z": np.nanmedian(sub["z_pop"]),
            "median_log10LIR": np.nanmedian(sub["log10LIR"]),
            "median_dust2_pop": np.nanmedian(sub["dust2_pop"]),
            "median_lnfAGN_pop": np.nanmedian(sub["lnfAGN_pop"]),
            "median_fAGN_pop": np.nanmedian(sub["fAGN_pop"]),
            "N_fAGN_gt_0p1": int(np.sum(sub["fAGN_pop"] > 0.1)),
            "N_fAGN_gt_0p3": int(np.sum(sub["fAGN_pop"] > 0.3)),
        }
        for band in OBS_BANDS_UM:
            row[f"median_F{band}_fsps_mjy"] = np.nanmedian(sub[f"F{band}_fsps_mjy"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "popcosmos_full_sed_extreme_sfr_summary.csv", index=False)

    cols = [
        "ID",
        "z_pop",
        "SFR_pop",
        "log10LIR",
        "dust2_pop",
        "lnfAGN_pop",
        "fAGN_pop",
        "lntauAGN_pop",
    ]
    for band in OBS_BANDS_UM:
        cols.append(f"F{band}_fsps_mjy")
    top = pred[np.isfinite(pred["SFR_pop"])].nlargest(25, "SFR_pop")[cols].copy()
    top.to_csv(OUT_DIR / "popcosmos_full_sed_extreme_sfr_top25.csv", index=False)

    return summary, top


def main():
    cache_path = OUT_DIR / "popcosmos_full_sed_band_predictions.pkl"
    if cache_path.exists():
        print(f"Loading cached multiband predictions from {cache_path}...")
        pred = pd.read_pickle(cache_path)
    else:
        print("Loading full FSPS SED and computing observed 250/350/500/850um fluxes...")
        pred = compute_band_predictions()
        pred.to_pickle(cache_path)

    template = load_aless_template(ALESS_PATH)
    top_table = plot_top5_seds(pred, template)
    sfr_sample_table = plot_sfr_rank_sed_samples(pred, template)
    median_sed_summary = plot_population_median_seds(pred)
    (
        count_summary,
        wang_summary,
        brightcut_bias,
        model_model_bias,
        external_counts,
    ) = make_counts_and_wang_outputs(pred)
    extreme_summary, extreme_top = make_extreme_sfr_outputs(pred)

    print("\nTop 5 SFR objects:")
    print(top_table.to_string(index=False))
    print("\nSFR-rank SED samples:")
    print(sfr_sample_table.to_string(index=False))
    print("\nCounts:")
    print(count_summary.to_string(index=False))
    print("\nWang comparison:")
    print(wang_summary.to_string(index=False))
    print("\nMedian SED summary:")
    print(median_sed_summary.to_string(index=False))
    print("\nBright-cut Wang residual slices:")
    print(brightcut_bias.to_string(index=False))
    print("\nFSPS/ALESS model-model residual slices:")
    print(model_model_bias.to_string(index=False))
    print("\nExternal count rows used:")
    print(external_counts.to_string(index=False))
    print("\nExtreme SFR summary:")
    print(extreme_summary.to_string(index=False))
    print("\nTop 25 extreme SFR objects:")
    print(extreme_top.to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
