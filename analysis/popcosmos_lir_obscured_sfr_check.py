"""First L_IR-based obscured-SFR check for the real pop-cosmos catalog.

This script joins the new FSPS-based total infrared luminosities onto the
existing pop-cosmos median posterior summaries, then makes a small set of
meeting-ready summaries and plots:

- How strongly model L_IR tracks pop-cosmos SFR
- How the model L_IR compares to a simple Kennicutt (1998) IR-SFR reference
- Whether Wang long-wavelength detections sit at higher model L_IR, as expected
"""

from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def robust_sigma(x: np.ndarray) -> float:
    """Return a robust scatter estimate using 1.4826 * MAD."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def load_pop_cosmos_with_lir(summary_path: Path, lir_path: Path) -> pd.DataFrame:
    """Load the pop-cosmos median catalog and merge the new L_IR scalars.

    Inputs
    ------
    summary_path : Path
        Path to `mcmc_summaries.h5`.
    lir_path : Path
        Path to `fsps_lir_scalars.h5`.

    Returns
    -------
    pd.DataFrame
        One row per galaxy with IDs, median pop-cosmos properties, and new
        infrared-luminosity quantities.
    """
    with h5py.File(summary_path, "r") as f:
        pop = pd.DataFrame(
            {
                "ID": f["metadata/index_farmer"][:].astype(np.int64),
                "log10M_pop": f["pop-cosmos/log10M_remain"][:, 2],
                "log10SFR_pop": f["pop-cosmos/log10SFR"][:, 2],
                "log10sSFR_pop": f["pop-cosmos/log10sSFR"][:, 2],
                "z_pop": f["pop-cosmos/z"][:, 2],
                "lnfAGN_pop": f["pop-cosmos/lnfAGN"][:, 2],
                "ra_pop": f["metadata/ra"][:],
                "dec_pop": f["metadata/dec"][:],
            }
        )

    with h5py.File(lir_path, "r") as f:
        lir = pd.DataFrame(
            {
                "ID": f["index"][:].astype(np.int64),
                "row_lir": f["row"][:].astype(np.int64),
                "z_lir": f["z"][:],
                "L_IR": f["L_IR"][:],
                "L_dust_balance": f["L_dust_balance"][:],
                "mfrac": f["mfrac"][:],
                "done_lir": f["done"][:].astype(bool),
            }
        )

    df = pop.merge(lir, on="ID", how="inner", validate="one_to_one")
    df = df[df["ID"] > 0].copy()
    df["log10LIR"] = np.log10(df["L_IR"])
    df["log10Ldust_balance"] = np.log10(df["L_dust_balance"])

    # Quick Kennicutt 1998 reference line in its original Salpeter-style form:
    # L_IR [Lsun] ~ 5.8e9 * SFR [Msun/yr]
    df["log10LIR_k98_sal"] = df["log10SFR_pop"] + np.log10(5.8e9)
    df["dlogLIR_model_minus_k98_sal"] = df["log10LIR"] - df["log10LIR_k98_sal"]

    return df


def base_sf_mask(df: pd.DataFrame) -> pd.Series:
    """Return the same broad star-forming-like cut used in earlier notes."""
    return (
        np.isfinite(df["log10M_pop"])
        & np.isfinite(df["log10SFR_pop"])
        & np.isfinite(df["log10sSFR_pop"])
        & np.isfinite(df["z_pop"])
        & np.isfinite(df["log10LIR"])
        & np.isfinite(df["dlogLIR_model_minus_k98_sal"])
        & (df["log10M_pop"] >= 8.5)
        & (df["log10M_pop"] <= 11.5)
        & (df["z_pop"] >= 0.0)
        & (df["z_pop"] < 4.0)
        & (df["log10sSFR_pop"] > -11.0)
        & df["done_lir"]
    )


def summarize_lir_subset(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    """Summarize the L_IR-SFR relation for one subset."""
    m = mask.fillna(False)
    if int(m.sum()) == 0:
        return {
            "subset": label,
            "N": 0,
            "median_z": np.nan,
            "median_log10M_pop": np.nan,
            "median_log10SFR_pop": np.nan,
            "median_log10LIR": np.nan,
            "median_dlogLIR_model_minus_k98_sal": np.nan,
            "sigma_mad_dlogLIR_model_minus_k98_sal": np.nan,
            "spearman_rho_logSFR_vs_logLIR": np.nan,
        }

    sub = df.loc[m, [
        "z_pop",
        "log10M_pop",
        "log10SFR_pop",
        "log10LIR",
        "dlogLIR_model_minus_k98_sal",
    ]].copy()

    return {
        "subset": label,
        "N": int(len(sub)),
        "median_z": float(np.nanmedian(sub["z_pop"])),
        "median_log10M_pop": float(np.nanmedian(sub["log10M_pop"])),
        "median_log10SFR_pop": float(np.nanmedian(sub["log10SFR_pop"])),
        "median_log10LIR": float(np.nanmedian(sub["log10LIR"])),
        "median_dlogLIR_model_minus_k98_sal": float(
            np.nanmedian(sub["dlogLIR_model_minus_k98_sal"])
        ),
        "sigma_mad_dlogLIR_model_minus_k98_sal": robust_sigma(
            sub["dlogLIR_model_minus_k98_sal"].to_numpy()
        ),
        "spearman_rho_logSFR_vs_logLIR": float(
            sub["log10SFR_pop"].corr(sub["log10LIR"], method="spearman")
        ),
    }


def make_lir_sfr_plot(df: pd.DataFrame, all_mask: pd.Series, bin_a_mask: pd.Series, out_path: Path):
    """Make simple SFR-L_IR hexbins with a Kennicutt reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)

    panels = [
        (axes[0], all_mask, "All SF-like sample"),
        (axes[1], bin_a_mask, "Bin A: 1 <= z < 2, 9 <= logM <= 11.5"),
    ]

    hb = None
    for ax, mask, title in panels:
        d = df.loc[mask, ["log10SFR_pop", "log10LIR"]].dropna().copy()
        hb = ax.hexbin(
            d["log10SFR_pop"],
            d["log10LIR"],
            gridsize=65,
            mincnt=1,
            cmap="viridis",
        )
        x_line = np.linspace(d["log10SFR_pop"].quantile(0.01), d["log10SFR_pop"].quantile(0.99), 200)
        y_k98 = x_line + np.log10(5.8e9)
        ax.plot(x_line, y_k98, "r--", lw=1.6, label="Kennicutt 1998 ref.")
        ax.set_xlabel(r"$\log_{10}(\mathrm{SFR}_{pop}/M_\odot\,\mathrm{yr}^{-1})$")
        ax.set_ylabel(r"$\log_{10}(L_{\mathrm{IR}}/L_\odot)$")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    cbar = fig.colorbar(hb, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("counts per hexbin")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_lir_offset_redshift_plot(df: pd.DataFrame, mask: pd.Series, out_path: Path):
    """Plot median L_IR offset from the simple Kennicutt reference vs redshift."""
    z_edges = np.arange(0.0, 4.0 + 0.5, 0.5)
    rows = []
    for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
        m = mask & (df["z_pop"] >= z0) & (df["z_pop"] < z1)
        if int(m.sum()) == 0:
            continue
        arr = df.loc[m, "dlogLIR_model_minus_k98_sal"].to_numpy()
        rows.append(
            {
                "z_mid": 0.5 * (z0 + z1),
                "N": int(m.sum()),
                "median": float(np.nanmedian(arr)),
                "p16": float(np.nanpercentile(arr, 16)),
                "p84": float(np.nanpercentile(arr, 84)),
            }
        )

    binned = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(binned["z_mid"], binned["median"], marker="o", color="#1f77b4", lw=1.8)
    ax1.fill_between(binned["z_mid"], binned["p16"], binned["p84"], color="#1f77b4", alpha=0.18)
    ax1.axhline(0.0, color="k", ls="--", lw=1)
    ax1.set_xlabel("redshift z")
    ax1.set_ylabel(r"$\Delta \log_{10}(L_{\mathrm{IR}})$ model $-$ Kennicutt ref.")
    ax1.set_title("How the model $L_{IR}$ offset changes with redshift")

    ax2 = ax1.twinx()
    ax2.bar(binned["z_mid"], binned["N"], width=0.38, alpha=0.18, color="gray")
    ax2.set_ylabel("counts per redshift bin")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return binned


def load_wang_catalog(master_path: Path, readme_path: Path) -> pd.DataFrame:
    """Load the Wang catalog columns needed for the long-wave split."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = Table.read(master_path, format="ascii.cds", readme=readme_path)

    df = tab.to_pandas()
    keep = ["ID", "F250", "s_F250", "F350", "s_F350", "F500", "s_F500", "F850", "s_F850"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[df["ID"] > 0].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for lam in [250, 350, 500, 850]:
        df[f"SNR{lam}"] = df[f"F{lam}"] / df[f"s_F{lam}"].replace(0, np.nan)

    df["long_detect"] = (
        (df["SNR250"] >= 3)
        | (df["SNR350"] >= 3)
        | (df["SNR500"] >= 3)
        | (df["SNR850"] >= 3)
    )
    return df


def summarize_wang_groups(df: pd.DataFrame, base_mask: pd.Series) -> pd.DataFrame:
    """Summarize model L_IR for all matched / long-detected / not-long-detected groups."""
    rows = []
    groups = {
        "all_matched": base_mask,
        "long_detect": base_mask & df["long_detect"],
        "not_long_detect": base_mask & ~df["long_detect"],
    }

    for label, mask in groups.items():
        m = mask.fillna(False)
        if int(m.sum()) == 0:
            rows.append(
                {
                    "group": label,
                    "N": 0,
                    "median_z": np.nan,
                    "median_log10M_pop": np.nan,
                    "median_log10SFR_pop": np.nan,
                    "median_log10LIR": np.nan,
                    "median_dlogLIR_model_minus_k98_sal": np.nan,
                }
            )
            continue

        sub = df.loc[m]
        rows.append(
            {
                "group": label,
                "N": int(len(sub)),
                "median_z": float(np.nanmedian(sub["z_pop"])),
                "median_log10M_pop": float(np.nanmedian(sub["log10M_pop"])),
                "median_log10SFR_pop": float(np.nanmedian(sub["log10SFR_pop"])),
                "median_log10LIR": float(np.nanmedian(sub["log10LIR"])),
                "median_dlogLIR_model_minus_k98_sal": float(
                    np.nanmedian(sub["dlogLIR_model_minus_k98_sal"])
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_wang_bands(df: pd.DataFrame, base_mask: pd.Series) -> pd.DataFrame:
    """Summarize the direct link between model L_IR and Wang long-wave fluxes."""
    rows = []
    for band in [250, 350, 500, 850]:
        fcol = f"F{band}"
        snr_col = f"SNR{band}"
        m = (
            base_mask
            & np.isfinite(df[fcol])
            & np.isfinite(df[snr_col])
            & (df[snr_col] >= 3)
            & (df[fcol] > 0)
        )
        if int(m.sum()) == 0:
            rows.append(
                {
                    "band_um": band,
                    "N_detect": 0,
                    "median_log10LIR": np.nan,
                    "median_log10SFR_pop": np.nan,
                    "spearman_rho_logflux_vs_logLIR": np.nan,
                }
            )
            continue

        sub = df.loc[m, [fcol, "log10LIR", "log10SFR_pop"]].copy()
        sub[f"log10F{band}"] = np.log10(sub[fcol])
        rows.append(
            {
                "band_um": band,
                "N_detect": int(len(sub)),
                "median_log10LIR": float(np.nanmedian(sub["log10LIR"])),
                "median_log10SFR_pop": float(np.nanmedian(sub["log10SFR_pop"])),
                "spearman_rho_logflux_vs_logLIR": float(
                    sub[f"log10F{band}"].corr(sub["log10LIR"], method="spearman")
                ),
            }
        )

    return pd.DataFrame(rows)


def make_wang_lir_plot(df: pd.DataFrame, bin_a_mask: pd.Series, out_path: Path):
    """Show how Wang long-wave detections line up with model L_IR."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    groups = [
        ("all matched", bin_a_mask, "#7f7f7f"),
        ("long-detected", bin_a_mask & df["long_detect"], "#d62728"),
        ("not long-detected", bin_a_mask & ~df["long_detect"], "#1f77b4"),
    ]
    for label, mask, color in groups:
        arr = df.loc[mask, "log10LIR"].dropna()
        axes[0].hist(arr, bins=60, density=True, histtype="step", lw=1.8, label=label, color=color)
    axes[0].set_xlabel(r"$\log_{10}(L_{\mathrm{IR}}/L_\odot)$")
    axes[0].set_ylabel("density")
    axes[0].set_title("Bin A model $L_{IR}$ distributions")
    axes[0].legend(fontsize=8)

    hb = None
    for ax, band, title, cmap in [
        (axes[1], 250, "Wang 250 um vs model $L_{IR}$", "magma"),
        (axes[2], 850, "Wang 850 um vs model $L_{IR}$", "cividis"),
    ]:
        fcol = f"F{band}"
        snr_col = f"SNR{band}"
        m = (
            bin_a_mask
            & np.isfinite(df[fcol])
            & np.isfinite(df[snr_col])
            & (df[snr_col] >= 3)
            & (df[fcol] > 0)
        )
        d = df.loc[m, ["log10LIR", fcol]].copy()
        d[f"log10F{band}"] = np.log10(d[fcol])
        hb = ax.hexbin(d["log10LIR"], d[f"log10F{band}"], gridsize=45, mincnt=1, cmap=cmap)
        ax.set_xlabel(r"$\log_{10}(L_{\mathrm{IR}}/L_\odot)$")
        ax.set_ylabel(rf"$\log_{{10}}(F_{{{band}}})$")
        ax.set_title(title)

    cbar = fig.colorbar(hb, ax=axes[1:], shrink=0.88, pad=0.03)
    cbar.set_label("counts per hexbin")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    summary_path = ROOT / "catalog data/real pop-cosmos data/mcmc_summaries.h5"
    lir_path = ROOT / "fsps_lir_scalars.h5"
    wang_master = ROOT / "catalog data/wang/master.dat.gz"
    wang_readme = ROOT / "catalog data/wang/ReadMe.txt"

    df = load_pop_cosmos_with_lir(summary_path, lir_path)

    # Sanity check: the new scalar file lines up exactly with the main catalog.
    z_diff = np.nanmax(np.abs(df["z_pop"] - df["z_lir"]))
    print(f"Max |z_pop - z_lir| after merge: {z_diff:.3e}")

    base = base_sf_mask(df)
    subsets = {
        "all_sf_like": base,
        "binA_1<=z<2_9<=logM<=11.5": base & (df["z_pop"] >= 1.0) & (df["z_pop"] < 2.0) & (df["log10M_pop"] >= 9.0),
        "binB_1.5<=z<2.5_logM>=10": base & (df["z_pop"] >= 1.5) & (df["z_pop"] < 2.5) & (df["log10M_pop"] >= 10.0),
        "narrow_1.0<=z<1.5": base & (df["z_pop"] >= 1.0) & (df["z_pop"] < 1.5) & (df["log10M_pop"] >= 9.0),
        "narrow_1.5<=z<2.0": base & (df["z_pop"] >= 1.5) & (df["z_pop"] < 2.0) & (df["log10M_pop"] >= 9.0),
    }

    lir_summary = pd.DataFrame(
        [summarize_lir_subset(df, mask, label) for label, mask in subsets.items()]
    )
    lir_summary_path = OUT_DIR / "popcosmos_lir_summary.csv"
    lir_summary.to_csv(lir_summary_path, index=False)
    print("Saved:", lir_summary_path)

    make_lir_sfr_plot(
        df,
        subsets["all_sf_like"],
        subsets["binA_1<=z<2_9<=logM<=11.5"],
        OUT_DIR / "popcosmos_lir_vs_sfr.png",
    )
    redshift_offset_summary = make_lir_offset_redshift_plot(
        df,
        subsets["all_sf_like"],
        OUT_DIR / "popcosmos_lir_offset_by_redshift.png",
    )
    redshift_offset_path = OUT_DIR / "popcosmos_lir_offset_redshift_bins.csv"
    redshift_offset_summary.to_csv(redshift_offset_path, index=False)
    print("Saved:", redshift_offset_path)

    wang = load_wang_catalog(wang_master, wang_readme)
    wang_df = df.merge(wang, on="ID", how="inner")
    wang_bin_a = (
        base_sf_mask(wang_df)
        & (wang_df["z_pop"] >= 1.0)
        & (wang_df["z_pop"] < 2.0)
        & (wang_df["log10M_pop"] >= 9.0)
    )

    wang_group_summary = summarize_wang_groups(wang_df, wang_bin_a)
    wang_band_summary = summarize_wang_bands(wang_df, wang_bin_a)
    # Save the two clean tables separately too.
    wang_group_path = OUT_DIR / "popcosmos_wang_lir_group_summary.csv"
    wang_band_path = OUT_DIR / "popcosmos_wang_lir_band_summary.csv"
    wang_group_summary.to_csv(wang_group_path, index=False)
    wang_band_summary.to_csv(wang_band_path, index=False)
    print("Saved:", wang_group_path)
    print("Saved:", wang_band_path)

    make_wang_lir_plot(
        wang_df,
        wang_bin_a,
        OUT_DIR / "popcosmos_wang_lir_vs_flux.png",
    )


if __name__ == "__main__":
    main()
