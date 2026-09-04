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
    """Load only the COSMOS2020 Farmer IRAC columns needed here.

    Returns a DataFrame with ID, flux, and valid-flag columns for IRAC Ch1/Ch2.
    """
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
    """Load the stored pop-cosmos IRAC model mags and matching metadata."""
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


def mad(values: np.ndarray) -> float:
    """Return the median absolute deviation."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def binned_stats(z: np.ndarray, resid: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    """Compute simple redshift-binned residual summaries."""
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        m = (z >= left) & (z < right) & np.isfinite(resid)
        if not np.any(m):
            continue
        x = resid[m]
        rows.append(
            {
                "z_lo": left,
                "z_hi": right,
                "N": int(m.sum()),
                "median_resid_mag": float(np.median(x)),
                "p16_resid_mag": float(np.percentile(x, 16)),
                "p84_resid_mag": float(np.percentile(x, 84)),
            }
        )
    return pd.DataFrame(rows)


def make_observed_vs_model_plot(
    obs_mag: np.ndarray,
    model_mag: np.ndarray,
    out_path: Path,
    title: str,
    channel_label: str,
) -> dict:
    """Make a dense observed-vs-model IRAC plot and return summary stats."""
    mask = np.isfinite(obs_mag) & np.isfinite(model_mag)
    x = obs_mag[mask]
    y = model_mag[mask]
    resid = y - x

    lo = float(np.nanpercentile(np.concatenate([x, y]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([x, y]), 99.5))

    fig, ax = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    hb = ax.hexbin(x, y, gridsize=95, mincnt=1, cmap="viridis")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1 line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Observed {channel_label} AB mag (COSMOS2020 Farmer)")
    ax.set_ylabel(f"Model {channel_label} AB mag (pop-cosmos stored)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True)
    cbar = fig.colorbar(hb, ax=ax, location="right", shrink=0.92, pad=0.02)
    cbar.set_label("counts per hexbin")

    med = float(np.median(resid))
    spread = mad(resid)
    ax.text(
        0.04,
        0.96,
        f"N = {mask.sum():,}\nmedian(model-observed) = {med:+.3f} mag\nMAD = {spread:.3f} mag",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "channel": channel_label,
        "N": int(mask.sum()),
        "median_resid_mag": med,
        "mad_resid_mag": spread,
    }


def make_residual_vs_z_plot(
    z1: np.ndarray,
    resid1: np.ndarray,
    z2: np.ndarray,
    resid2: np.ndarray,
    out_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot residuals vs redshift with binned median and scatter bands."""
    bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5])
    s1 = binned_stats(z1, resid1, bins)
    s2 = binned_stats(z2, resid2, bins)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.5), sharex=True, constrained_layout=True)

    for ax, z, resid, stats, label, color in [
        (axes[0], z1, resid1, s1, "Ch1", "#1f77b4"),
        (axes[1], z2, resid2, s2, "Ch2", "#ff7f0e"),
    ]:
        m = np.isfinite(z) & np.isfinite(resid)
        ax.scatter(z[m], resid[m], s=2, alpha=0.08, color=color, rasterized=True)
        centers = 0.5 * (stats["z_lo"].to_numpy() + stats["z_hi"].to_numpy())
        ax.plot(centers, stats["median_resid_mag"], color="red", lw=2, label="binned median")
        ax.fill_between(
            centers,
            stats["p16_resid_mag"],
            stats["p84_resid_mag"],
            color="red",
            alpha=0.18,
            label="16-84 percentile",
        )
        ax.axhline(0.0, color="black", lw=1, ls="--")
        ax.set_ylabel(f"{label}: model - observed (mag)")
        ax.legend(loc="upper right", frameon=True)
        ax.set_ylim(-0.45, 0.45)

    axes[0].set_title("IRAC residuals vs redshift")
    axes[1].set_xlabel("Redshift z")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    s1.insert(0, "channel", "Ch1")
    s2.insert(0, "channel", "Ch2")
    return s1, s2


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    farmer = load_farmer_irac_columns(FARMER_PATH)
    mir = load_mir_validation(MIR_PATH)

    idx = mir["index_farmer"] - 1

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

    summary_rows = []
    summary_rows.append(
        make_observed_vs_model_plot(
            obs_ch1,
            model_ch1,
            OUT_DIR / "popcosmos_irac_ch1_observed_vs_model.png",
            "Observed vs model IRAC Ch1",
            "IRAC Ch1",
        )
    )
    summary_rows.append(
        make_observed_vs_model_plot(
            obs_ch2,
            model_ch2,
            OUT_DIR / "popcosmos_irac_ch2_observed_vs_model.png",
            "Observed vs model IRAC Ch2",
            "IRAC Ch2",
        )
    )

    resid_ch1 = model_ch1 - obs_ch1
    resid_ch2 = model_ch2 - obs_ch2
    zstats_ch1, zstats_ch2 = make_residual_vs_z_plot(
        mir["z"],
        resid_ch1,
        mir["z"],
        resid_ch2,
        OUT_DIR / "popcosmos_irac_residual_vs_redshift.png",
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "popcosmos_irac_validation_summary.csv", index=False)
    pd.concat([zstats_ch1, zstats_ch2], ignore_index=True).to_csv(
        OUT_DIR / "popcosmos_irac_residual_vs_redshift_summary.csv", index=False
    )

    print(summary.to_string(index=False))
    print()
    print("Saved:")
    for name in [
        "popcosmos_irac_ch1_observed_vs_model.png",
        "popcosmos_irac_ch2_observed_vs_model.png",
        "popcosmos_irac_residual_vs_redshift.png",
        "popcosmos_irac_validation_summary.csv",
        "popcosmos_irac_residual_vs_redshift_summary.csv",
    ]:
        print(" -", OUT_DIR / name)


if __name__ == "__main__":
    main()
