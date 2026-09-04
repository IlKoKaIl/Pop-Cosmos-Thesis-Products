"""Quick Wang/Jin/pop-cosmos FSPS FIR flux comparison.

Purpose:
- Wang IDs are COSMOS2020 Farmer IDs.
- Jin IDs are old COSMOS2015-like IDs for the normal optical/NIR prior sources.
- COSMOS2020 Farmer has an ID_COSMOS2015 bridge column, so match through that
  bridge first instead of treating the IDs as directly comparable.
- Compare matched-object fluxes in the same observed SPIRE bands.
- This is a per-object diagnostic, so sky area is not used here.
"""

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u


CODE_DIR = Path(__file__).resolve().parent
NB_DIR = CODE_DIR.parent
PROJECT_CODING = CODE_DIR.parents[1]
OUT_DIR = NB_DIR / "outputs"

PREDICTION_CACHE = OUT_DIR / "popcosmos_full_sed_band_predictions.pkl"
WANG_MASTER = PROJECT_CODING / "catalog data/wang/master.dat.gz"
WANG_README = PROJECT_CODING / "catalog data/wang/ReadMe.txt"
JIN_FITS = PROJECT_CODING / "catalog data/Jin-et-all_files/COSMOS_Super_Deblended_FIRmm_Catalog_20180719.fits"
COSMOS2020_FARMER = PROJECT_CODING / "catalog data/Main pop cosmos/farmer.dat.gz"

BANDS = [250, 350, 500]
MATCH_RADIUS_ARCSEC = 1.0

OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE_DIR))


def load_wang_full():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = Table.read(WANG_MASTER, format="ascii.cds", readme=WANG_README)
    cols = ["ID", "RAdeg", "DEdeg"]
    for band in BANDS:
        cols.extend([f"F{band}", f"s_F{band}"])
    df = tab.to_pandas()[cols].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["ID"] > 0].copy()
    return df


def load_jin():
    with fits.open(JIN_FITS, memmap=True) as hdul:
        data = hdul[1].data
        cols = {
            "jin_ID": data["ID"].astype(np.int64),
            "RA_jin": data["RA"].astype(float),
            "DEC_jin": data["DEC"].astype(float),
            "goodArea_jin": data["goodArea"].astype(int),
            "SNR_IR_jin": data["SNR_IR"].astype(float),
        }
        for band in BANDS:
            cols[f"F{band}_jin_mjy"] = data[f"F{band}"].astype(float)
            cols[f"eF{band}_jin_mjy"] = data[f"DF{band}"].astype(float)
        return pd.DataFrame(cols)


def load_cosmos2020_farmer_bridge():
    """Read only the fixed-width ID bridge columns from the large Farmer file."""
    df = pd.read_fwf(
        COSMOS2020_FARMER,
        compression="gzip",
        colspecs=[
            (0, 6),       # Farmer COSMOS2020 ID
            (69, 81),     # detection RA
            (82, 92),     # detection Dec
            (2139, 2146), # ID_COSMOS2015
            (2401, 2408), # ID_CLASSIC, useful for sanity checks only
        ],
        names=["ID", "RAdeg_farmer", "DEdeg_farmer", "COSMOS2015", "Classic"],
        na_values=["---"],
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coordinate_match(base, jin):
    base_coord = SkyCoord(base["RAdeg"].to_numpy(float) * u.deg, base["DEdeg"].to_numpy(float) * u.deg)
    jin_coord = SkyCoord(jin["RA_jin"].to_numpy(float) * u.deg, jin["DEC_jin"].to_numpy(float) * u.deg)
    idx, sep2d, _ = base_coord.match_to_catalog_sky(jin_coord)
    keep = sep2d <= MATCH_RADIUS_ARCSEC * u.arcsec
    return idx, sep2d, keep


def build_matched_table():
    pred = pd.read_pickle(PREDICTION_CACHE)
    pred = pred[["ID"] + [f"F{band}_fsps_mjy" for band in BANDS]].copy()

    wang = load_wang_full()
    base = pred.merge(wang, on="ID", how="inner", validate="one_to_one")
    base = base[
        np.isfinite(base["RAdeg"])
        & np.isfinite(base["DEdeg"])
        & (base["RAdeg"] > 0)
        & (base["DEdeg"] > -90)
    ].copy()

    jin = load_jin()
    jin = jin[np.isfinite(jin["RA_jin"]) & np.isfinite(jin["DEC_jin"])].copy()

    bridge = load_cosmos2020_farmer_bridge()
    base = base.merge(
        bridge[["ID", "RAdeg_farmer", "DEdeg_farmer", "COSMOS2015", "Classic"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    bridged_base = base[base["COSMOS2015"].notna()].copy()
    bridged_base["COSMOS2015"] = bridged_base["COSMOS2015"].astype(np.int64)
    matched = bridged_base.merge(
        jin,
        left_on="COSMOS2015",
        right_on="jin_ID",
        how="inner",
        validate="many_to_one",
    )

    matched_coord = SkyCoord(matched["RAdeg"].to_numpy(float) * u.deg, matched["DEdeg"].to_numpy(float) * u.deg)
    jin_coord = SkyCoord(matched["RA_jin"].to_numpy(float) * u.deg, matched["DEC_jin"].to_numpy(float) * u.deg)
    matched["jin_match_sep_arcsec"] = matched_coord.separation(jin_coord).arcsec
    matched["match_method"] = "farmer_to_cosmos2015_id"

    for band in BANDS:
        matched[f"SNR{band}_wang"] = matched[f"F{band}"] / matched[f"s_F{band}"].replace(0, np.nan)
        matched[f"SNR{band}_jin"] = matched[f"F{band}_jin_mjy"] / matched[f"eF{band}_jin_mjy"].replace(0, np.nan)

    _, coord_sep2d, coord_keep = coordinate_match(base, jin)
    coord_matched_ids = set(base.loc[coord_keep, "ID"].astype(int))
    id_matched_ids = set(matched["ID"].astype(int))
    audit = pd.DataFrame(
        [
            {"check": "popcosmos_wang_rows", "value": len(base)},
            {"check": "rows_with_cosmos2015_bridge", "value": int(base["COSMOS2015"].notna().sum())},
            {"check": "id_bridge_matched_to_jin", "value": len(matched)},
            {
                "check": f"coordinate_matched_to_jin_lt_{MATCH_RADIUS_ARCSEC:.1f}_arcsec",
                "value": int(coord_keep.sum()),
            },
            {"check": "matched_by_both_id_bridge_and_coordinate", "value": len(id_matched_ids & coord_matched_ids)},
            {"check": "coordinate_only_not_id_bridge", "value": len(coord_matched_ids - id_matched_ids)},
            {"check": "id_bridge_only_not_coordinate", "value": len(id_matched_ids - coord_matched_ids)},
            {"check": "id_bridge_median_sep_arcsec", "value": float(np.nanmedian(matched["jin_match_sep_arcsec"]))},
            {
                "check": f"coordinate_match_median_sep_arcsec_lt_{MATCH_RADIUS_ARCSEC:.1f}",
                "value": float(np.nanmedian(coord_sep2d[coord_keep].arcsec)),
            },
        ]
    )

    return matched, audit


def plot_scatter(matched):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharex=False, sharey=False)
    rng = np.random.default_rng(7)

    for ax, band in zip(axes, BANDS):
        sub = matched[
            (matched[f"F{band}_jin_mjy"] > 0)
            & (matched[f"SNR{band}_jin"] >= 3)
            & (matched[f"F{band}"] > 0)
            & (matched[f"F{band}_fsps_mjy"] > 0)
            & np.isfinite(matched[f"F{band}_jin_mjy"])
            & np.isfinite(matched[f"F{band}"])
            & np.isfinite(matched[f"F{band}_fsps_mjy"])
        ].copy()
        if len(sub) > 20000:
            sub = sub.iloc[rng.choice(len(sub), 20000, replace=False)]

        ax.scatter(
            sub[f"F{band}_jin_mjy"],
            sub[f"F{band}"],
            s=3,
            alpha=0.18,
            color="black",
            label="Wang vs Jin",
        )
        ax.scatter(
            sub[f"F{band}_jin_mjy"],
            sub[f"F{band}_fsps_mjy"],
            s=3,
            alpha=0.18,
            color="#0072B2",
            label="FSPS vs Jin",
        )

        lo = max(0.03, np.nanpercentile(sub[f"F{band}_jin_mjy"], 0.5))
        hi = np.nanpercentile(
            np.concatenate(
                [
                    sub[f"F{band}_jin_mjy"].to_numpy(float),
                    sub[f"F{band}"].to_numpy(float),
                    sub[f"F{band}_fsps_mjy"].to_numpy(float),
                ]
            ),
            99.5,
        )
        grid = np.logspace(np.log10(lo), np.log10(hi), 100)
        ax.plot(grid, grid, color="0.5", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{band} um")
        ax.set_xlabel("Jin super-deblended flux [mJy]")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Wang / pop-cosmos FSPS flux [mJy]")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Wang and pop-cosmos FSPS compared to Jin by COSMOS2020 Farmer -> COSMOS2015 ID bridge "
        "(Jin SNR>=3)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = OUT_DIR / "popcosmos_wang_jin_fsps_flux_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_ratio_summary(matched):
    rows = []
    for band in BANDS:
        base = matched[
            (matched[f"F{band}_jin_mjy"] > 0)
            & (matched[f"eF{band}_jin_mjy"] > 0)
            & (matched[f"F{band}"] > 0)
            & (matched[f"s_F{band}"] > 0)
            & (matched[f"F{band}_fsps_mjy"] > 0)
        ].copy()
        regimes = {
            "all_positive": base,
            "jin_snr3": base[base[f"SNR{band}_jin"] >= 3],
            "jin_and_wang_snr3": base[(base[f"SNR{band}_jin"] >= 3) & (base[f"SNR{band}_wang"] >= 3)],
        }
        for regime, sub in regimes.items():
            for label, num, den in [
                ("Wang/Jin", f"F{band}", f"F{band}_jin_mjy"),
                ("FSPS/Jin", f"F{band}_fsps_mjy", f"F{band}_jin_mjy"),
                ("FSPS/Wang", f"F{band}_fsps_mjy", f"F{band}"),
            ]:
                ratio = np.log10(sub[num].to_numpy(float) / sub[den].to_numpy(float))
                ratio = ratio[np.isfinite(ratio)]
                if len(ratio) == 0:
                    continue
                rows.append(
                    {
                        "band_um": band,
                        "regime": regime,
                        "ratio": label,
                        "N": int(len(ratio)),
                        "median_log10_ratio": float(np.nanmedian(ratio)),
                        "p16_log10_ratio": float(np.nanpercentile(ratio, 16)),
                        "p84_log10_ratio": float(np.nanpercentile(ratio, 84)),
                        "median_linear_ratio": float(10 ** np.nanmedian(ratio)),
                    }
                )
    return pd.DataFrame(rows)


def plot_ratio_summary(summary):
    summary = summary[summary["regime"] == "jin_snr3"].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xlabels = []
    x = []
    y = []
    yerr_low = []
    yerr_high = []
    colors = {"Wang/Jin": "black", "FSPS/Jin": "#0072B2", "FSPS/Wang": "#D55E00"}

    i = 0
    for band in BANDS:
        for ratio in ["Wang/Jin", "FSPS/Jin", "FSPS/Wang"]:
            row = summary[(summary["band_um"] == band) & (summary["ratio"] == ratio)].iloc[0]
            x.append(i)
            y.append(row["median_log10_ratio"])
            yerr_low.append(row["median_log10_ratio"] - row["p16_log10_ratio"])
            yerr_high.append(row["p84_log10_ratio"] - row["median_log10_ratio"])
            xlabels.append(f"{band}\n{ratio}")
            ax.errorbar(
                i,
                row["median_log10_ratio"],
                yerr=[[yerr_low[-1]], [yerr_high[-1]]],
                fmt="o",
                color=colors[ratio],
                capsize=3,
            )
            i += 1
        i += 0.7

    ax.axhline(0, color="0.45", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("median log10 flux ratio")
    ax.set_title("Matched-object flux ratios for Jin SNR>=3: 0 means perfect agreement")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / "popcosmos_wang_jin_fsps_ratio_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    matched, audit = build_matched_table()
    matched_path = OUT_DIR / "popcosmos_wang_jin_fsps_matched_fluxes.csv"
    matched.to_csv(matched_path, index=False)
    audit_path = OUT_DIR / "popcosmos_wang_jin_fsps_match_audit.csv"
    audit.to_csv(audit_path, index=False)

    summary = make_ratio_summary(matched)
    summary_path = OUT_DIR / "popcosmos_wang_jin_fsps_ratio_summary.csv"
    summary.to_csv(summary_path, index=False)

    scatter = plot_scatter(matched)
    ratio_plot = plot_ratio_summary(summary)

    print(f"Matched rows: {len(matched)}")
    print(matched["jin_match_sep_arcsec"].describe(percentiles=[0.5, 0.9, 0.99]).to_string())
    print(audit.to_string(index=False))
    print(summary.to_string(index=False))
    print(matched_path)
    print(audit_path)
    print(summary_path)
    print(scatter)
    print(ratio_plot)


if __name__ == "__main__":
    main()
