"""First-pass pop-cosmos vs Wang (2024) long-wavelength comparison.

This script joins pop-cosmos posterior summaries to the Wang far-IR/sub-mm
catalog by COSMOS ID, computes main-sequence offsets (Delta_MS), and reports
starburst fractions in the same z/mass bins used in the notebook.
"""

import warnings
from pathlib import Path
import gzip
import shutil

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from astropy.table import Table


def speagle_log10sfr(log10m, z):
    """Compute Speagle+2014 expected log10(SFR) at given mass and redshift.

    Parameters
    ----------
    log10m : array-like
        log10 stellar mass in solar masses.
    z : array-like
        Redshift.

    Returns
    -------
    np.ndarray
        Expected log10(SFR) from Speagle relation.
    """
    t_gyr = Planck18.age(np.asarray(z)).value
    return (0.84 - 0.026 * t_gyr) * np.asarray(log10m) - (6.51 - 0.11 * t_gyr)


def robust_sigma(x):
    """Robust scatter estimate using 1.4826*MAD around the median.

    Parameters
    ----------
    x : array-like
        Values to summarize.

    Returns
    -------
    float
        Robust sigma estimate.
    """
    x = np.asarray(x)
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def ensure_uncompressed_h5(h5_gz_path: Path, h5_path: Path):
    """Ensure an uncompressed .h5 exists from a .h5.gz source.

    Parameters
    ----------
    h5_gz_path : Path
        Path to compressed source file.
    h5_path : Path
        Destination path for uncompressed file.
    """
    if h5_path.exists():
        return
    if not h5_gz_path.exists():
        raise FileNotFoundError(f"Missing compressed file: {h5_gz_path}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(h5_gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024 * 8)


def load_pop_cosmos_summary(h5_path: Path) -> pd.DataFrame:
    """Load pop-cosmos median posterior summaries needed for this analysis.

    Parameters
    ----------
    h5_path : Path
        Path to `mcmc_summaries.h5`.

    Returns
    -------
    pd.DataFrame
        Columns: ID, magcut_Ch1, log10M, log10SFR, log10sSFR, z.
    """
    with h5py.File(h5_path, "r") as f:
        idx_farmer = f["metadata/index_farmer"][:].astype(np.int64)
        magcut_ch1 = f["metadata/magcut_Ch1"][:].astype(bool)

        # percentile axis is [2.5, 16, 50, 84, 97.5], index 2 = median
        log10m = f["pop-cosmos/log10M_remain"][:, 2]
        log10sfr = f["pop-cosmos/log10SFR"][:, 2]
        log10ssfr = f["pop-cosmos/log10sSFR"][:, 2]
        z = f["pop-cosmos/z"][:, 2]

    df = pd.DataFrame(
        {
            "ID": idx_farmer,
            "magcut_Ch1": magcut_ch1,
            "log10M": log10m,
            "log10SFR": log10sfr,
            "log10sSFR": log10ssfr,
            "z": z,
        }
    )
    # keep valid farmer IDs only
    df = df[df["ID"] > 0].copy()
    return df


def load_wang_master(master_path: Path, readme_path: Path) -> pd.DataFrame:
    """Load Wang+2024 master catalog with SNR-based long-wavelength flags.

    Parameters
    ----------
    master_path : Path
        Path to `master.dat.gz`.
    readme_path : Path
        Path to CDS-style `ReadMe.txt` for schema parsing.

    Returns
    -------
    pd.DataFrame
        Wang catalog subset with ID, coordinates, SNR columns, and boolean
        long-wavelength detection flags.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Table.read(master_path, format="ascii.cds", readme=readme_path)

    data = {}
    for col in t.colnames:
        c = t[col]
        if hasattr(c, "filled"):
            arr = c.filled(np.nan)
        else:
            arr = np.asarray(c)
        data[col] = arr

    df = pd.DataFrame(data)

    numeric_cols = [
        "F24",
        "s_F24",
        "F100",
        "s_F100",
        "F160",
        "s_F160",
        "F250",
        "s_F250",
        "F350",
        "s_F350",
        "F500",
        "s_F500",
        "F850",
        "s_F850",
    ]
    for c in numeric_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build per-band SNR columns from flux / uncertainty in Wang catalog.
    for lam in [24, 100, 160, 250, 350, 500, 850]:
        fcol = f"F{lam}"
        scol = f"s_F{lam}"
        snr_col = f"SNR{lam}"
        if fcol in df and scol in df:
            den = df[scol].replace(0, np.nan)
            df[snr_col] = df[fcol] / den

    spire_scuba_snr_cols = [
        c for c in ["SNR250", "SNR350", "SNR500", "SNR850"] if c in df
    ]
    farir_snr_cols = [
        c
        for c in ["SNR100", "SNR160", "SNR250", "SNR350", "SNR500", "SNR850"]
        if c in df
    ]

    # Long-wavelength detection proxies used for split comparisons.
    df["long_detect_spire_scuba_snr3"] = (df[spire_scuba_snr_cols] >= 3).any(axis=1)
    df["long_detect_firplus_snr3"] = (df[farir_snr_cols] >= 3).any(axis=1)

    keep_cols = (
        [
            "ID",
            "RAdeg",
            "DEdeg",
            "long_detect_spire_scuba_snr3",
            "long_detect_firplus_snr3",
        ]
        + spire_scuba_snr_cols
        + farir_snr_cols
    )

    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    # Keep COSMOS2020-linked sources only (Wang uses negative IDs for radio-only priors).
    out = out[out["ID"] > 0].copy()
    return out


def summarize_group(df: pd.DataFrame, base_mask: pd.Series, label: str):
    """Summarize starburst and Delta_MS statistics for a selected subgroup.

    Parameters
    ----------
    df : pd.DataFrame
        Merged analysis table with `delta_ms` and `log10SFR`.
    base_mask : pd.Series
        Boolean mask selecting the subgroup.
    label : str
        Group label to store in output.

    Returns
    -------
    dict
        One-row summary with counts, fractions, and robust Delta_MS stats.
    """
    m = base_mask.fillna(False)
    n = int(m.sum())
    if n == 0:
        return {
            "group": label,
            "N": 0,
            "N_starburst": 0,
            "starburst_frac": np.nan,
            "starburst_sfr_share": np.nan,
            "delta_ms_median": np.nan,
            "delta_ms_sigma": np.nan,
            "z_median": np.nan,
            "log10M_median": np.nan,
        }

    sb = m & (df["delta_ms"] >= 0.6)
    n_sb = int(sb.sum())

    sfr_total = np.sum(10 ** df.loc[m, "log10SFR"].to_numpy())
    sfr_sb = np.sum(10 ** df.loc[sb, "log10SFR"].to_numpy())

    d = df.loc[m, "delta_ms"].to_numpy()

    return {
        "group": label,
        "N": n,
        "N_starburst": n_sb,
        "starburst_frac": n_sb / n,
        "starburst_sfr_share": (sfr_sb / sfr_total) if sfr_total > 0 else np.nan,
        "delta_ms_median": float(np.nanmedian(d)),
        "delta_ms_sigma": float(robust_sigma(d)),
        "z_median": float(np.nanmedian(df.loc[m, "z"].to_numpy())),
        "log10M_median": float(np.nanmedian(df.loc[m, "log10M"].to_numpy())),
    }


def main():
    """Run the complete pop-cosmos vs Wang first-pass comparison."""
    # .../Research project/Project coding/pop_cosmos_notebook/<this_file>
    # parents[1] => .../Research project/Project coding
    root = Path(__file__).resolve().parents[1]

    pop_h5_gz = root / "catalog data/real pop-cosmos data/mcmc_summaries.h5.gz"
    pop_h5 = root / "catalog data/real pop-cosmos data/mcmc_summaries.h5"
    wang_master = root / "catalog data/wang/master.dat.gz"
    wang_readme = root / "catalog data/wang/ReadMe.txt"

    ensure_uncompressed_h5(pop_h5_gz, pop_h5)
    if not wang_master.exists() or not wang_readme.exists():
        raise FileNotFoundError("Missing Wang master/readme files.")

    pop_df = load_pop_cosmos_summary(pop_h5)
    wang_df = load_wang_master(wang_master, wang_readme)

    # Join the two catalogs on shared COSMOS/Farmer ID.
    merged = pop_df.merge(wang_df, how="inner", on="ID", validate="one_to_one")

    merged["log10SFR_speagle"] = speagle_log10sfr(
        merged["log10M"].to_numpy(), merged["z"].to_numpy()
    )
    merged["delta_ms"] = merged["log10SFR"] - merged["log10SFR_speagle"]

    # Baseline star-forming sample quality cuts (same logic as notebook task).
    sf_base = (
        np.isfinite(merged["log10M"])
        & np.isfinite(merged["log10SFR"])
        & np.isfinite(merged["log10sSFR"])
        & np.isfinite(merged["z"])
        & (merged["z"] >= 0.0)
        & (merged["z"] < 4.0)
        & (merged["log10M"] >= 8.5)
        & (merged["log10M"] <= 11.5)
        & (merged["log10sSFR"] > -11.0)
    )

    # Bin A: same as task 2 in notebook
    bin_a = (
        sf_base
        & (merged["z"] >= 1.0)
        & (merged["z"] < 2.0)
        & (merged["log10M"] >= 9.0)
        & (merged["log10M"] <= 11.5)
    )

    # Bin B: literature-style z~2 high-mass slice
    bin_b = (
        sf_base
        & (merged["z"] >= 1.5)
        & (merged["z"] < 2.5)
        & (merged["log10M"] >= 10.0)
    )

    rows = []
    for bin_label, bin_mask in [
        ("binA_1<=z<2_9<=logM<=11.5", bin_a),
        ("binB_1.5<=z<2.5_logM>=10", bin_b),
    ]:
        rows.append(summarize_group(merged, bin_mask, f"{bin_label}__all_matched"))
        rows.append(
            summarize_group(
                merged,
                bin_mask & merged["long_detect_spire_scuba_snr3"],
                f"{bin_label}__long_detect_spire_scuba_snr3",
            )
        )
        rows.append(
            summarize_group(
                merged,
                bin_mask & (~merged["long_detect_spire_scuba_snr3"]),
                f"{bin_label}__not_long_detect_spire_scuba_snr3",
            )
        )

    out_df = pd.DataFrame(rows)

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    out_csv = out_dir / "popcosmos_wang_comparison_summary.csv"
    out_df.to_csv(out_csv, index=False)

    print("=== Data linkage summary ===")
    print(f"pop-cosmos rows (ID>0): {len(pop_df):,}")
    print(f"Wang rows: {len(wang_df):,}")
    print(f"Matched rows (ID inner join): {len(merged):,}")
    print(
        "Matched long-detected fraction (SPIRE/SCUBA SNR>=3): "
        f"{100.0*merged['long_detect_spire_scuba_snr3'].mean():.2f}%"
    )

    print("\n=== Starburst comparison summary ===")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 220
    ):
        print(out_df)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
