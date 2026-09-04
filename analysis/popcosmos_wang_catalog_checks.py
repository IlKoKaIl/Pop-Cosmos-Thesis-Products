"""Small Wang catalogue sanity checks.

This does not replace reading the Wang paper. It just records what the local
`master.dat` file actually contains and how many raw/SNR-selected detections are
available in each SPIRE/SCUBA band.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from astropy.table import Table


NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
PROJECT_CODING = NB_DIR.parents[1]
OUT_DIR = ROOT / "outputs"
WANG_MASTER = PROJECT_CODING / "catalog data/wang/master.dat.gz"
WANG_README = PROJECT_CODING / "catalog data/wang/ReadMe.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA_SCENARIOS_DEG2 = {
    "wang_farmer_flag_combined0": 1.278,
    "old_full_cosmos_approx": 2.0,
}
FLUX_CUTS_MJY = [5, 10, 20, 50, 100]


def load_wang():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = Table.read(WANG_MASTER, format="ascii.cds", readme=WANG_README)
    df = tab.to_pandas()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    df = load_wang()
    ra = df["RAdeg"].to_numpy(float)
    dec = df["DEdeg"].to_numpy(float)
    rough_rect = (np.nanmax(ra) - np.nanmin(ra)) * (np.nanmax(dec) - np.nanmin(dec))
    rough_rect *= np.cos(np.deg2rad(np.nanmedian(dec)))

    summary = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "positive_cosmos2020_ids": int((df["ID"] > 0).sum()),
                "negative_radio_only_ids": int((df["ID"] < 0).sum()),
                "ra_min_deg": np.nanmin(ra),
                "ra_max_deg": np.nanmax(ra),
                "dec_min_deg": np.nanmin(dec),
                "dec_max_deg": np.nanmax(dec),
                "rough_coordinate_box_deg2_not_effective_area": rough_rect,
                "paper_farmer_flag_combined0_area_deg2": 1.278,
                "full_cosmos_rough_area_deg2": 2.0,
                "notes": "Coordinate box is not the same as effective masked area. Use paper area for counts.",
            }
        ]
    )

    rows = []
    scenario_rows = []
    for band in [250, 350, 500, 850]:
        f = df[f"F{band}"].to_numpy(float)
        sf = df[f"s_F{band}"].to_numpy(float)
        valid = np.isfinite(f) & np.isfinite(sf) & (f > 0) & (sf > 0)
        snr = np.full_like(f, np.nan, dtype=float)
        snr[valid] = f[valid] / sf[valid]
        rows.append(
            {
                "band_um": band,
                "n_positive_flux_and_error": int(valid.sum()),
                "n_snr_ge_3": int((valid & (snr >= 3)).sum()),
                "n_flux_ge_10_mjy": int((valid & (f >= 10)).sum()),
                "n_flux_ge_20_mjy": int((valid & (f >= 20)).sum()),
                "n_flux_ge_50_mjy": int((valid & (f >= 50)).sum()),
                "n_flux_ge_100_mjy": int((valid & (f >= 100)).sum()),
                "median_snr_for_positive_flux": float(np.nanmedian(snr[valid])),
            }
        )

        populations = {
            "all_wang_prior_rows": np.ones(len(df), dtype=bool),
            "positive_cosmos2020_ids_only": (df["ID"].to_numpy(float) > 0),
        }
        for population_name, population_mask in populations.items():
            for cut in FLUX_CUTS_MJY:
                flux_cut = population_mask & valid & (f >= cut)
                snr3_cut = flux_cut & (snr >= 3)
                snr5_cut = flux_cut & (snr >= 5)
                for area_name, area_deg2 in AREA_SCENARIOS_DEG2.items():
                    scenario_rows.append(
                        {
                            "band_um": band,
                            "population": population_name,
                            "flux_cut_mjy": cut,
                            "area_name": area_name,
                            "area_deg2": area_deg2,
                            "N_positive_flux_ge_cut": int(flux_cut.sum()),
                            "N_snr3_flux_ge_cut": int(snr3_cut.sum()),
                            "N_snr5_flux_ge_cut": int(snr5_cut.sum()),
                            "N_positive_per_deg2": float(flux_cut.sum() / area_deg2),
                            "N_snr3_per_deg2": float(snr3_cut.sum() / area_deg2),
                            "N_snr5_per_deg2": float(snr5_cut.sum() / area_deg2),
                            "snr3_fraction_of_flux_cut": float(snr3_cut.sum() / flux_cut.sum())
                            if flux_cut.sum() > 0
                            else np.nan,
                        }
                    )
    detections = pd.DataFrame(rows)
    count_scenarios = pd.DataFrame(scenario_rows)

    summary_path = OUT_DIR / "wang_master_catalog_area_summary.csv"
    detections_path = OUT_DIR / "wang_master_catalog_detection_counts.csv"
    count_scenarios_path = OUT_DIR / "wang_master_catalog_count_scenarios.csv"
    notes_path = OUT_DIR / "wang_master_catalog_discrepancy_note.md"
    summary.to_csv(summary_path, index=False)
    detections.to_csv(detections_path, index=False)
    count_scenarios.to_csv(count_scenarios_path, index=False)

    key_rows = count_scenarios[
        (count_scenarios["population"] == "positive_cosmos2020_ids_only")
        & (count_scenarios["area_name"] == "wang_farmer_flag_combined0")
        & (count_scenarios["flux_cut_mjy"].isin([10, 20, 50, 100]))
    ].copy()
    key_rows = key_rows[
        [
            "band_um",
            "flux_cut_mjy",
            "N_positive_flux_ge_cut",
            "N_snr3_flux_ge_cut",
            "N_positive_per_deg2",
            "N_snr3_per_deg2",
        ]
    ]
    key_rows["N_positive_per_deg2"] = key_rows["N_positive_per_deg2"].round(1)
    key_rows["N_snr3_per_deg2"] = key_rows["N_snr3_per_deg2"].round(1)
    key_table_lines = [
        "| band um | flux cut mJy | N positive | N SNR>=3 | N positive / deg2 | N SNR>=3 / deg2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in key_rows.itertuples(index=False):
        key_table_lines.append(
            f"| {int(row.band_um)} | {int(row.flux_cut_mjy)} | "
            f"{int(row.N_positive_flux_ge_cut)} | {int(row.N_snr3_flux_ge_cut)} | "
            f"{row.N_positive_per_deg2:.1f} | {row.N_snr3_per_deg2:.1f} |"
        )

    notes = [
        "# Wang Catalogue Discrepancy Note",
        "",
        "This is a quick bookkeeping note generated from the local Wang `master.dat.gz` catalogue.",
        "",
        "## What The File Is",
        "",
        "- Wang Table 4 / CDS `master.dat` is a deblended point-source catalogue, not a published corrected number-count table.",
        "- `F250`, `F350`, `F500`, and `F850` are median flux densities in `mJy`.",
        "- `s_F250`, `s_F350`, `s_F500`, and `s_F850` are one-sigma-style flux errors in `mJy`.",
        "- Negative IDs are radio-prior sources rather than normal positive COSMOS2020 IDs.",
        "",
        "## Area Choice",
        "",
        f"- catalogue rows: `{len(df)}`",
        f"- positive COSMOS2020 IDs: `{int((df['ID'] > 0).sum())}`",
        f"- negative radio-only IDs: `{int((df['ID'] < 0).sum())}`",
        "- the COSMOS2020/Farmer `FLAG_COMBINED=0` area from Wang is `1.278 deg2`",
        "- the old `2 deg2` COSMOS value is only a rough full-field reference and should not be used for this catalogue's raw count density",
        "",
        "## Raw Count Sanity Check",
        "",
        "Positive COSMOS2020 IDs only, using `1.278 deg2`:",
        "",
        "\n".join(key_table_lines),
        "",
        "## Simple Interpretation",
        "",
        "- At 250 um, Wang has many positive-ID sources above 10-20 mJy, so matched-object checks are meaningful.",
        "- At 500 um and especially 850 um, the bright raw-count statistics get very small fast.",
        "- The raw Wang curve can sit away from external published counts because it is prior-selected, deblended, small-area, and not corrected like Clements/Oliver/Pearson count tables.",
        "- So for the thesis, Wang should mainly diagnose per-object residuals and population failures, while published corrected differential counts should carry the formal count evaluator.",
        "",
    ]
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    print(summary_path)
    print(detections_path)
    print(count_scenarios_path)
    print(notes_path)
    print(summary.to_string(index=False))
    print(detections.to_string(index=False))
    print(count_scenarios.to_string(index=False))


if __name__ == "__main__":
    main()
