from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BORIS_DIR = ROOT / "Boris work"
CAT_DIR = ROOT / "catalog data" / "Main pop cosmos"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

MIR_PATH = BORIS_DIR / "cosmos2020_mir_photometry.h5"
FARMER_PATH = CAT_DIR / "farmer.dat.gz"


def load_farmer_irac_columns(path: Path) -> pd.DataFrame:
    """Load only the COSMOS2020 Farmer IRAC columns needed for Ch1/Ch2."""
    colspecs = [
        (0, 6),        # ID
        (1814, 1823),  # IRAC_CH1_FLUX
        (1876, 1877),  # IRAC_CH1_VALID
        (1878, 1887),  # IRAC_CH2_FLUX
        (1939, 1940),  # IRAC_CH2_VALID
    ]
    names = [
        "ID",
        "IRAC_CH1_FLUX",
        "IRAC_CH1_VALID",
        "IRAC_CH2_FLUX",
        "IRAC_CH2_VALID",
    ]
    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        compression="gzip",
        dtype=str,
        na_values=["---", "--", "-"],
    )
    for col in names:
        if col == "ID":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_mir_validation(path: Path) -> dict:
    """Load the MIR validation arrays needed for the histogram diagnostics."""
    with h5py.File(path, "r") as f:
        return {
            "index_farmer": f["metadata/index_farmer"][:],
            "z": f["metadata/z_median"][:],
            "stored_mag_Ch1": f["validation/stored_mag_Ch1"][:],
            "stored_mag_Ch2": f["validation/stored_mag_Ch2"][:],
        }


def observed_abmag_from_flux(flux_ujy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Convert valid microJy fluxes to AB magnitudes."""
    return np.where((flux_ujy > 0) & valid, -2.5 * np.log10(flux_ujy) + 23.9, np.nan)


def build_hist_summary(z: np.ndarray, bins: np.ndarray, label: str) -> pd.DataFrame:
    """Tabulate histogram counts by redshift bin."""
    counts, edges = np.histogram(z, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame(
        {
            "sample": label,
            "z_lo": edges[:-1],
            "z_hi": edges[1:],
            "z_mid": centers,
            "count": counts,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    farmer = load_farmer_irac_columns(FARMER_PATH)
    mir = load_mir_validation(MIR_PATH)
    idx = mir["index_farmer"] - 1
    z = mir["z"]

    obs_ch1 = observed_abmag_from_flux(
        farmer.iloc[idx]["IRAC_CH1_FLUX"].to_numpy(float),
        farmer.iloc[idx]["IRAC_CH1_VALID"].fillna(0).astype(int).to_numpy().astype(bool),
    )
    obs_ch2 = observed_abmag_from_flux(
        farmer.iloc[idx]["IRAC_CH2_FLUX"].to_numpy(float),
        farmer.iloc[idx]["IRAC_CH2_VALID"].fillna(0).astype(int).to_numpy().astype(bool),
    )
    model_ch1 = mir["stored_mag_Ch1"]
    model_ch2 = mir["stored_mag_Ch2"]

    resid_ch1 = model_ch1 - obs_ch1
    resid_ch2 = model_ch2 - obs_ch2

    mask1 = np.isfinite(z) & np.isfinite(resid_ch1)
    mask2 = np.isfinite(z) & np.isfinite(resid_ch2)

    abs_r1 = np.abs(resid_ch1[mask1])
    abs_r2 = np.abs(resid_ch2[mask2])

    thr1 = float(np.percentile(abs_r1, 95))
    thr2 = float(np.percentile(abs_r2, 95))

    worst1 = mask1.copy()
    worst1[mask1] = abs_r1 >= thr1
    worst2 = mask2.copy()
    worst2[mask2] = abs_r2 >= thr2

    bins = np.arange(0.0, 6.6, 0.1)

    # Plot 1: full matched-sample redshift histograms
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.5), sharex=True, constrained_layout=True)
    for ax, zvals, title, color, n in [
        (axes[0], z[mask1], "Matched sample redshift histogram: IRAC Ch1", "#1f77b4", int(mask1.sum())),
        (axes[1], z[mask2], "Matched sample redshift histogram: IRAC Ch2", "#ff7f0e", int(mask2.sum())),
    ]:
        ax.hist(zvals, bins=bins, color=color, alpha=0.85)
        ax.axvline(3.5, color="red", ls="--", lw=1.5, label="z = 3.5")
        ax.set_yscale("log")
        ax.set_ylabel("count")
        ax.set_title(f"{title} (N = {n:,})")
        ax.legend(loc="upper right")
    axes[1].set_xlabel("Redshift z")
    fig.savefig(OUT_DIR / "popcosmos_irac_redshift_histograms.png", dpi=180)
    plt.close(fig)

    # Plot 2: worst-residual redshift histograms
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.8), sharex=True, constrained_layout=True)
    for ax, zvals, zbad, title, color, thr in [
        (axes[0], z[mask1], z[worst1], "Worst residual redshift histogram: IRAC Ch1", "#1f77b4", thr1),
        (axes[1], z[mask2], z[worst2], "Worst residual redshift histogram: IRAC Ch2", "#ff7f0e", thr2),
    ]:
        ax.hist(zvals, bins=bins, density=True, color="lightgray", alpha=0.75, label="all matched")
        ax.hist(zbad, bins=bins, density=True, color=color, alpha=0.8, label="worst 5% by |residual|")
        ax.axvline(3.5, color="red", ls="--", lw=1.5, label="z = 3.5")
        ax.set_ylabel("normalized count")
        ax.set_title(f"{title} (threshold |residual| >= {thr:.3f} mag)")
        ax.legend(loc="upper right")
    axes[1].set_xlabel("Redshift z")
    fig.savefig(OUT_DIR / "popcosmos_irac_worst_residual_redshift_histograms.png", dpi=180)
    plt.close(fig)

    # Summary tables
    rows = [
        {"sample": "IRAC Ch1 matched", "N": int(mask1.sum()), "worst_fraction": 0.05, "worst_abs_resid_threshold_mag": thr1},
        {"sample": "IRAC Ch2 matched", "N": int(mask2.sum()), "worst_fraction": 0.05, "worst_abs_resid_threshold_mag": thr2},
        {"sample": "IRAC Ch1 worst", "N": int(worst1.sum()), "worst_fraction": np.nan, "worst_abs_resid_threshold_mag": thr1},
        {"sample": "IRAC Ch2 worst", "N": int(worst2.sum()), "worst_fraction": np.nan, "worst_abs_resid_threshold_mag": thr2},
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "popcosmos_irac_redshift_diagnostics_summary.csv", index=False)

    hist_table = pd.concat(
        [
            build_hist_summary(z[mask1], bins, "IRAC Ch1 matched"),
            build_hist_summary(z[mask2], bins, "IRAC Ch2 matched"),
            build_hist_summary(z[worst1], bins, "IRAC Ch1 worst"),
            build_hist_summary(z[worst2], bins, "IRAC Ch2 worst"),
        ],
        ignore_index=True,
    )
    hist_table.to_csv(OUT_DIR / "popcosmos_irac_redshift_histogram_bins.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
