from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parents[1]
SOURCE_COUNTS_CSV = ROOT / "catalog data/external_number_counts/external_spire_number_counts_starter.csv"
VARNISH_COUNTS_CSV = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_starter.csv"
GLENN_COUNTS_CSV = ROOT / "catalog data/external_number_counts/external_spire_glenn_2010_pd_counts.csv"
VALIANTE_COUNTS_CSV = ROOT / "catalog data/external_number_counts/valiante_2016_hatlas_dr1_number_counts_area_weighted.csv"
OUT_CSV = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
SR_PER_DEG2 = 3282.806350011744
VARNISH_LOG10_ERROR_FLOOR_DEX = 0.08


def to_float(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    return float(value)


def convert_source_row(row):
    flux_mjy = to_float(row["flux_mjy"])
    flux_jy = flux_mjy / 1000.0
    unit = str(row["differential_unit"]).strip()
    raw = to_float(row["differential_value"])
    raw_err = to_float(row.get("differential_err", np.nan))
    if not np.isfinite(raw):
        return None

    if unit == "gal sr^-1 Jy^1.5":
        euclidean = raw / SR_PER_DEG2
        euclidean_err = raw_err / SR_PER_DEG2 if np.isfinite(raw_err) else np.nan
        note = "Converted from Jy^1.5 sr^-1 to Jy^1.5 deg^-2."

    elif unit == "10^7 mJy^1.5 sr^-1":
        # Pearson tables are already Euclidean-normalised as S^2.5 dN/dS.
        # Only convert the quoted mJy^1.5 unit into Jy^1.5.
        mjy15_to_jy15 = (1.0e-3) ** 1.5
        euclidean = raw * 1.0e7 * mjy15_to_jy15 / SR_PER_DEG2
        euclidean_err = (
            raw_err * 1.0e7 * mjy15_to_jy15 / SR_PER_DEG2
            if np.isfinite(raw_err)
            else np.nan
        )
        note = "Multiplied by 1e7, converted mJy^1.5 to Jy^1.5 using (1e-3)^1.5, then sr^-1 to deg^-2."

    elif unit == "sr^-1 Jy^-1":
        # Oliver gives raw dN/dS, so here we apply the Euclidean S^2.5 weighting.
        dnds_deg2_jy = raw / SR_PER_DEG2
        euclidean = (flux_jy**2.5) * dnds_deg2_jy
        euclidean_err = (
            (flux_jy**2.5) * (raw_err / SR_PER_DEG2)
            if np.isfinite(raw_err)
            else np.nan
        )
        note = "Raw dN/dS converted from sr^-1 Jy^-1 to deg^-2 Jy^-1, then multiplied by S^2.5."

    else:
        raise ValueError(f"Unhandled differential unit: {unit}")

    out = {
        "paper": row["paper"],
        "year": row["year"],
        "survey": row["survey"],
        "method_or_table": row["method_or_table"],
        "band_um": int(float(row["band_um"])),
        "flux_mjy": flux_mjy,
        "flux_jy": flux_jy,
        "euclidean_best_jy15_deg2": euclidean,
        "euclidean_err_jy15_deg2": euclidean_err if np.isfinite(euclidean_err) else "",
        "log10_euclidean_best_jy15_deg2": np.log10(euclidean) if euclidean > 0 else "",
        "source_n_galaxies": row.get("source_n_galaxies", ""),
        "flux_correction": row.get("flux_correction", ""),
        "source_differential_value": raw,
        "source_differential_err": raw_err if np.isfinite(raw_err) else "",
        "source_differential_unit": unit,
        "standard_unit": "Jy^1.5 deg^-2",
        "notes": f"{row.get('notes', '')} {note}".strip(),
        "source_url": row["source_url"],
    }
    return out


def convert_varnish_row(row):
    euclidean = to_float(row["euclidean_best_jy15_deg2"])
    log_best = to_float(row["euclidean_log10_best"])
    log_lower = to_float(row.get("euclidean_log10_lower", np.nan))
    log_upper = to_float(row.get("euclidean_log10_upper", np.nan))

    # Varnish table entries are already Euclidean-normalised log10 values.
    # Convert the published lower/upper 1-sigma bounds into one conservative
    # symmetric log error for the simple chi-square evaluator. Some high-flux
    # spline knots have lower=best=upper, so keep a floor to avoid giving those
    # correlated P(D) constraints infinite weight.
    log_spread = np.nanmax(np.abs([log_best - log_lower, log_upper - log_best]))
    if not np.isfinite(log_spread):
        log_spread = VARNISH_LOG10_ERROR_FLOOR_DEX
    log_err = max(float(log_spread), VARNISH_LOG10_ERROR_FLOOR_DEX)
    euclidean_err = euclidean * np.log(10.0) * log_err if euclidean > 0 else np.nan

    return {
        "paper": row["paper"],
        "year": row["year"],
        "survey": row["survey"],
        "method_or_table": row["method_or_table"],
        "band_um": int(float(row["band_um"])),
        "flux_mjy": to_float(row["flux_mjy"]),
        "flux_jy": to_float(row["flux_jy"]),
        "euclidean_best_jy15_deg2": euclidean,
        "euclidean_err_jy15_deg2": euclidean_err if np.isfinite(euclidean_err) else "",
        "log10_euclidean_best_jy15_deg2": log_best,
        "source_n_galaxies": "",
        "flux_correction": "",
        "source_differential_value": log_best,
        "source_differential_err": log_err,
        "source_differential_unit": "log10 Jy^1.5 deg^-2",
        "standard_unit": "Jy^1.5 deg^-2",
        "notes": (
            f"{row['notes']} Converted published log10 lower/upper bounds into "
            f"a conservative symmetric log error with {VARNISH_LOG10_ERROR_FLOOR_DEX:.2f} dex floor; "
            "treat as a P(D) sensitivity constraint, not fully independent bins."
        ),
        "source_url": row["source_url"],
    }


def convert_glenn_row(row):
    flux_mjy = to_float(row["flux_mjy"])
    flux_jy = flux_mjy / 1000.0
    log_dnds = to_float(row["log10_dnds_deg2_jyminus1"])
    log_err_plus = to_float(row["log10_err_plus"])
    log_err_minus = to_float(row["log10_err_minus"])
    log_sys = to_float(row["log10_sys_err"])

    stat_log_err = np.nanmean([log_err_plus, log_err_minus])
    combined_log_err = np.sqrt(stat_log_err**2 + log_sys**2)

    dnds_deg2_jy = 10.0**log_dnds
    euclidean = (flux_jy**2.5) * dnds_deg2_jy
    euclidean_err = euclidean * np.log(10.0) * combined_log_err

    return {
        "paper": row["paper"],
        "year": row["year"],
        "survey": row["survey"],
        "method_or_table": row["method_or_table"],
        "band_um": int(float(row["band_um"])),
        "flux_mjy": flux_mjy,
        "flux_jy": flux_jy,
        "euclidean_best_jy15_deg2": euclidean,
        "euclidean_err_jy15_deg2": euclidean_err,
        "log10_euclidean_best_jy15_deg2": np.log10(euclidean) if euclidean > 0 else "",
        "source_n_galaxies": "",
        "flux_correction": "",
        "source_differential_value": log_dnds,
        "source_differential_err": combined_log_err,
        "source_differential_unit": "log10 deg^-2 Jy^-1",
        "standard_unit": "Jy^1.5 deg^-2",
        "notes": (
            f"{row['notes']} Converted from log10 dN/dS [deg^-2 Jy^-1] "
            "to S^2.5 dN/dS [Jy^1.5 deg^-2]."
        ),
        "source_url": row["source_url"],
    }


def convert_valiante_row(row):
    euclidean = to_float(row["euclidean_best_jy15_deg2"])
    euclidean_err = to_float(row["euclidean_err_jy15_deg2"])
    flux_mjy = to_float(row["flux_mjy"])
    return {
        "paper": row["paper"],
        "year": row["year"],
        "survey": row["survey"],
        "method_or_table": row["method_or_table"],
        "band_um": int(float(row["band_um"])),
        "flux_mjy": flux_mjy,
        "flux_jy": to_float(row["flux_jy"]),
        "euclidean_best_jy15_deg2": euclidean,
        "euclidean_err_jy15_deg2": euclidean_err if np.isfinite(euclidean_err) else "",
        "log10_euclidean_best_jy15_deg2": np.log10(euclidean) if euclidean > 0 else "",
        "source_n_galaxies": "",
        "flux_correction": "",
        "source_differential_value": euclidean,
        "source_differential_err": euclidean_err if np.isfinite(euclidean_err) else "",
        "source_differential_unit": "Jy^1.5 deg^-2",
        "standard_unit": "Jy^1.5 deg^-2",
        "notes": (
            f"{row['notes']} Added as the area-weighted H-ATLAS DR1 "
            "GAMA9/GAMA12/GAMA15 curve; use as wide-area bright-end counts."
        ),
        "source_url": row["source_url"],
    }


def main():
    rows = []
    source = pd.read_csv(SOURCE_COUNTS_CSV)
    for _, row in source.iterrows():
        converted = convert_source_row(row)
        if converted is not None:
            rows.append(converted)

    if VARNISH_COUNTS_CSV.exists():
        varnish = pd.read_csv(VARNISH_COUNTS_CSV)
        rows.extend(convert_varnish_row(row) for _, row in varnish.iterrows())

    if GLENN_COUNTS_CSV.exists():
        glenn = pd.read_csv(GLENN_COUNTS_CSV)
        rows.extend(convert_glenn_row(row) for _, row in glenn.iterrows())

    if VALIANTE_COUNTS_CSV.exists():
        valiante = pd.read_csv(VALIANTE_COUNTS_CSV)
        rows.extend(convert_valiante_row(row) for _, row in valiante.iterrows())

    out = pd.DataFrame(rows)
    out = out.sort_values(["band_um", "flux_mjy", "paper", "method_or_table"])
    out.to_csv(OUT_CSV, index=False)
    print(OUT_CSV)
    print(out.groupby(["paper", "year", "survey", "method_or_table"]).size())


if __name__ == "__main__":
    main()
