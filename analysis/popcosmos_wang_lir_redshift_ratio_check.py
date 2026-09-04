"""Redshift diagnostic for Wang long-wave fluxes vs model L_IR.

This follows Dave's suggestion directly: check whether the ratio between the
model total infrared luminosity and the observed Wang single-band fluxes
changes with redshift. If it does, that supports the idea that part of the
scatter comes from observed-frame vs rest-frame effects.
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


def load_pop_lir(summary_path: Path, lir_path: Path) -> pd.DataFrame:
    """Load the pop-cosmos median catalog plus Boris's L_IR scalars."""
    with h5py.File(summary_path, "r") as f:
        pop = pd.DataFrame(
            {
                "ID": f["metadata/index_farmer"][:].astype(np.int64),
                "log10M_pop": f["pop-cosmos/log10M_remain"][:, 2],
                "log10SFR_pop": f["pop-cosmos/log10SFR"][:, 2],
                "log10sSFR_pop": f["pop-cosmos/log10sSFR"][:, 2],
                "z_pop": f["pop-cosmos/z"][:, 2],
            }
        )

    with h5py.File(lir_path, "r") as f:
        lir = pd.DataFrame(
            {
                "ID": f["index"][:].astype(np.int64),
                "L_IR": f["L_IR"][:],
            }
        )

    df = pop.merge(lir, on="ID", how="inner", validate="one_to_one")
    df["log10LIR"] = np.log10(df["L_IR"])
    return df


def base_sf_mask(df: pd.DataFrame) -> pd.Series:
    """Broad SF-like cut, same style as earlier notes."""
    return (
        np.isfinite(df["log10M_pop"])
        & np.isfinite(df["log10SFR_pop"])
        & np.isfinite(df["log10sSFR_pop"])
        & np.isfinite(df["z_pop"])
        & np.isfinite(df["log10LIR"])
        & (df["log10M_pop"] >= 8.5)
        & (df["log10M_pop"] <= 11.5)
        & (df["z_pop"] >= 0.0)
        & (df["z_pop"] < 4.0)
        & (df["log10sSFR_pop"] > -11.0)
    )


def load_wang(master_path: Path, readme_path: Path) -> pd.DataFrame:
    """Load Wang flux columns and build simple SNR-based detection flags."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = Table.read(master_path, format="ascii.cds", readme=readme_path)

    df = tab.to_pandas()
    keep = ["ID", "F250", "s_F250", "F350", "s_F350", "F500", "s_F500", "F850", "s_F850"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[df["ID"] > 0].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for band in [250, 350, 500, 850]:
        df[f"SNR{band}"] = df[f"F{band}"] / df[f"s_F{band}"].replace(0, np.nan)

    return df


def robust_sigma(x: np.ndarray) -> float:
    """Return 1.4826 * MAD as a robust spread estimate."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def band_ratio_summary(df: pd.DataFrame, base_mask: pd.Series, band: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the redshift dependence of log10(L_IR / F_band).

    Returns
    -------
    summary_one_row, binned_rows
        Overall one-row summary for the band plus the per-redshift-bin table.
    """
    fcol = f"F{band}"
    snr_col = f"SNR{band}"
    mask = (
        base_mask
        & np.isfinite(df[fcol])
        & np.isfinite(df[snr_col])
        & (df[snr_col] >= 3)
        & (df[fcol] > 0)
    )

    sub = df.loc[mask, ["z_pop", "log10LIR", fcol, "log10SFR_pop"]].copy()
    sub[f"log10F{band}"] = np.log10(sub[fcol])
    sub[f"log10LIR_over_F{band}"] = sub["log10LIR"] - sub[f"log10F{band}"]

    one_row = pd.DataFrame(
        [
            {
                "band_um": band,
                "N_detect": int(len(sub)),
                "median_z": float(np.nanmedian(sub["z_pop"])),
                "median_log10LIR": float(np.nanmedian(sub["log10LIR"])),
                "median_log10SFR_pop": float(np.nanmedian(sub["log10SFR_pop"])),
                "median_log10LIR_over_log10F": float(np.nanmedian(sub[f"log10LIR_over_F{band}"])),
                "sigma_mad_log10LIR_over_log10F": robust_sigma(sub[f"log10LIR_over_F{band}"].to_numpy()),
                "spearman_rho_ratio_vs_z": float(sub["z_pop"].corr(sub[f"log10LIR_over_F{band}"], method="spearman")),
            }
        ]
    )

    z_edges = np.arange(0.0, 4.0 + 0.5, 0.5)
    rows = []
    for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
        m = (sub["z_pop"] >= z0) & (sub["z_pop"] < z1)
        if int(m.sum()) == 0:
            continue
        arr = sub.loc[m, f"log10LIR_over_F{band}"].to_numpy()
        rows.append(
            {
                "band_um": band,
                "z_mid": 0.5 * (z0 + z1),
                "N": int(m.sum()),
                "median_ratio_log10": float(np.nanmedian(arr)),
                "p16_ratio_log10": float(np.nanpercentile(arr, 16)),
                "p84_ratio_log10": float(np.nanpercentile(arr, 84)),
            }
        )

    return one_row, pd.DataFrame(rows)


def make_ratio_vs_redshift_plot(df: pd.DataFrame, base_mask: pd.Series, out_path: Path):
    """Plot log10(L_IR/F_band) vs redshift for 250 and 850 um."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharey=True)

    for ax, band, cmap in [
        (axes[0], 250, "magma"),
        (axes[1], 850, "cividis"),
    ]:
        fcol = f"F{band}"
        snr_col = f"SNR{band}"
        mask = (
            base_mask
            & np.isfinite(df[fcol])
            & np.isfinite(df[snr_col])
            & (df[snr_col] >= 3)
            & (df[fcol] > 0)
        )
        sub = df.loc[mask, ["z_pop", "log10LIR", fcol]].copy()
        sub[f"log10F{band}"] = np.log10(sub[fcol])
        sub[f"log10LIR_over_F{band}"] = sub["log10LIR"] - sub[f"log10F{band}"]

        hb = ax.hexbin(
            sub["z_pop"],
            sub[f"log10LIR_over_F{band}"],
            gridsize=45,
            mincnt=1,
            cmap=cmap,
        )

        # Median trend by redshift bin.
        mids, meds = [], []
        p16s, p84s = [], []
        for z0 in np.arange(0.0, 4.0, 0.5):
            z1 = z0 + 0.5
            m = (sub["z_pop"] >= z0) & (sub["z_pop"] < z1)
            if int(m.sum()) == 0:
                continue
            arr = sub.loc[m, f"log10LIR_over_F{band}"].to_numpy()
            mids.append(0.5 * (z0 + z1))
            meds.append(np.nanmedian(arr))
            p16s.append(np.nanpercentile(arr, 16))
            p84s.append(np.nanpercentile(arr, 84))

        ax.plot(mids, meds, color="deepskyblue", marker="o", lw=1.8)
        ax.fill_between(mids, p16s, p84s, color="deepskyblue", alpha=0.18)
        ax.set_xlabel("redshift z")
        ax.set_title(rf"Wang {band} um: $\log_{{10}}(L_{{IR}}) - \log_{{10}}(F_{{{band}}})$")
        ax.set_ylabel(r"$\log_{10}(L_{IR}) - \log_{10}(F_{\lambda})$")

    cbar = fig.colorbar(hb, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("counts per hexbin")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    pop = load_pop_lir(
        ROOT / "catalog data/real pop-cosmos data/mcmc_summaries.h5",
        ROOT / "fsps_lir_scalars.h5",
    )
    wang = load_wang(
        ROOT / "catalog data/wang/master.dat.gz",
        ROOT / "catalog data/wang/ReadMe.txt",
    )
    matched = pop.merge(wang, on="ID", how="inner")
    base = base_sf_mask(matched)

    overall_rows = []
    binned_rows = []
    for band in [250, 350, 500, 850]:
        one, binned = band_ratio_summary(matched, base, band)
        overall_rows.append(one)
        binned_rows.append(binned)

    overall = pd.concat(overall_rows, ignore_index=True)
    binned = pd.concat(binned_rows, ignore_index=True)

    overall_path = OUT_DIR / "popcosmos_wang_lir_fluxratio_summary.csv"
    binned_path = OUT_DIR / "popcosmos_wang_lir_fluxratio_redshift_bins.csv"
    overall.to_csv(overall_path, index=False)
    binned.to_csv(binned_path, index=False)
    print("Saved:", overall_path)
    print("Saved:", binned_path)

    make_ratio_vs_redshift_plot(
        matched,
        base,
        OUT_DIR / "popcosmos_wang_lir_fluxratio_vs_redshift.png",
    )


if __name__ == "__main__":
    main()
