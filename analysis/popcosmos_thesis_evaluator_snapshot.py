"""Generate a compact thesis-facing snapshot of the FIR evaluator.

This script does not recompute the science products. It reads the latest CSV
outputs from the evaluator diagnostics and writes a small markdown report that
can be pasted into meeting notes, email, or thesis planning docs.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"

SCORE_SUMMARY = OUT_DIR / "popcosmos_model_family_score_summary.csv"
BEST_BY_REGIME = OUT_DIR / "popcosmos_model_family_best_by_regime.csv"
SOURCE_TENSION = OUT_DIR / "popcosmos_model_family_source_tension_summary.csv"
FLUX_REGIME = OUT_DIR / "popcosmos_model_family_flux_regime_summary.csv"
LEAVE_ONE = OUT_DIR / "popcosmos_model_family_leave_one_source_out.csv"
WANG_AREA = OUT_DIR / "wang_master_catalog_area_summary.csv"
PD_SENSITIVITY = OUT_DIR / "popcosmos_model_family_pd_sensitivity.csv"
DOF_CHECK = OUT_DIR / "popcosmos_evaluator_chi2_dof_check.csv"
FORMAL_SUMMARY = OUT_DIR / "popcosmos_formal_evaluator_summary.csv"

OUT_MD = OUT_DIR / "popcosmos_thesis_evaluator_snapshot.md"


def fmt(value, digits=2):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def dex_to_factor(dex):
    if pd.isna(dex):
        return ""
    return f"{10 ** float(dex):.1f}x"


def markdown_table(rows, headers):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def top_model_table(score_summary, n=8):
    rows = []
    for _, row in score_summary.head(n).iterrows():
        rows.append(
            {
                "model": row["model_label"],
                "family": row["model_family"],
                "N": int(row["N_points"]),
                "red chi2": fmt(row["reduced_chi2_log"], 2),
                "median log(model/obs)": fmt(row["median_log10_model_over_obs"], 2),
            }
        )
    return markdown_table(
        rows,
        ["model", "family", "N", "red chi2", "median log(model/obs)"],
    )


def best_regime_table(best_by_regime):
    rows = []
    for _, row in best_by_regime.iterrows():
        rows.append(
            {
                "regime": row["regime"],
                "best model": row["model_label"],
                "N": int(row["N_points"]),
                "red chi2": fmt(row["reduced_chi2_log"], 2),
                "median log(model/obs)": fmt(row["median_log10_model_over_obs"], 2),
            }
        )
    return markdown_table(
        rows,
        ["regime", "best model", "N", "red chi2", "median log(model/obs)"],
    )


def pd_sensitivity_table(pd_sensitivity):
    cols = [
        "all scored counts reduced_chi2_log",
        "resolved/prior counts reduced_chi2_log",
        "P(D) statistical counts reduced_chi2_log",
    ]
    key_labels = [
        "Casey T30K a=2.5",
        "Casey T30K a=3.0",
        "25% ALESS",
        "50% ALESS",
        "MBB 35 K",
        "FSPS",
        "ALESS",
    ]
    key = pd_sensitivity[pd_sensitivity["model_label"].isin(key_labels)].copy()
    key = key.sort_values("all scored counts reduced_chi2_log")
    rows = []
    for _, row in key.iterrows():
        rows.append(
            {
                "model": row["model_label"],
                "all": fmt(row[cols[0]], 2),
                "resolved/prior": fmt(row[cols[1]], 2),
                "P(D)": fmt(row[cols[2]], 2),
            }
        )
    return markdown_table(rows, ["model", "all", "resolved/prior", "P(D)"])


def dof_check_table(dof_check):
    rows = []
    if dof_check.empty:
        return "- Chi-square dof check table not found."
    for regime in ["all scored counts", "resolved/prior counts", "P(D) statistical counts"]:
        sub = dof_check[dof_check["regime"] == regime].copy()
        if sub.empty:
            continue
        best = sub.sort_values("reduced_chi2_log_simple_dof").iloc[0]
        rows.append(
            {
                "regime": regime,
                "best model": best["model_label"],
                "N": int(best["N_points"]),
                "k": int(best["template_param_count_simple"]),
                "chi2/N": fmt(best["reduced_chi2_log"], 2),
                "chi2/(N-k)": fmt(best["reduced_chi2_log_simple_dof"], 2),
            }
        )
    return markdown_table(rows, ["regime", "best model", "N", "k", "chi2/N", "chi2/(N-k)"])


def formal_summary_table(formal_summary):
    if formal_summary.empty:
        return "- Formal evaluator summary not found."
    rows = []
    for _, row in formal_summary.head(8).iterrows():
        rows.append(
            {
                "model": row["model_label"],
                "family": row["model_family"],
                "formal chi2/(N-k)": fmt(row["formal_resolved_prior_chi2_over_dof"], 2),
                "median log(model/obs)": fmt(
                    row["formal_resolved_prior_median_log_model_over_obs"], 2
                ),
                "P(D) chi2/N": fmt(row["pd_sensitivity_chi2_over_n"], 2),
            }
        )
    return markdown_table(
        rows,
        ["model", "family", "formal chi2/(N-k)", "median log(model/obs)", "P(D) chi2/N"],
    )


def source_tension_table(source_tension):
    rows = []
    for _, row in source_tension.iterrows():
        rows.append(
            {
                "source": row["source_short"],
                "type": row["count_regime"],
                "best model": row["best_model_label"],
                "best chi2": fmt(row["best_reduced_chi2"], 2),
                "FSPS chi2": fmt(row["fsps_reduced_chi2"], 2),
                "FSPS/best": fmt(row["fsps_over_best_chi2_ratio"], 1),
            }
        )
    return markdown_table(
        rows,
        ["source", "type", "best model", "best chi2", "FSPS chi2", "FSPS/best"],
    )


def fsps_flux_regime_table(flux_regime):
    fsps = flux_regime[flux_regime["model_label"] == "FSPS"].copy()
    rows = []
    for _, row in fsps.iterrows():
        rows.append(
            {
                "band": f"{int(row['band_um'])} um",
                "flux": row["flux_regime"],
                "N": int(row["N_points"]),
                "red chi2": fmt(row["reduced_chi2_log"], 2),
                "median log(model/obs)": fmt(row["median_log10_model_over_obs"], 2),
                "rough factor": dex_to_factor(row["median_log10_model_over_obs"]),
            }
        )
    return markdown_table(
        rows,
        ["band", "flux", "N", "red chi2", "median log(model/obs)", "rough factor"],
    )


def leave_one_table(leave_one):
    rows = []
    for _, row in leave_one.iterrows():
        source = str(row["heldout_external_source"])
        source = (
            source.replace("Clements et al. / Table 1", "Clements")
            .replace("Oliver et al. / Table 2", "Oliver")
            .replace("Pearson et al. / Table 3 SUSSEXtractor", "Pearson SUSSEX")
            .replace("Pearson et al. / Table 4 XID", "Pearson XID")
            .replace("Glenn et al. / Table 4 P(D) spline no FIRAS", "Glenn P(D)")
            .replace("Varnish et al. / Table 4 P(D) best-fit spline", "Varnish P(D)")
        )
        rows.append(
            {
                "held out": source,
                "picked from others": row["selected_model_label"],
                "held-out chi2": fmt(row["heldout_reduced_chi2"], 2),
                "oracle chi2": fmt(row["heldout_oracle_reduced_chi2"], 2),
                "FSPS chi2": fmt(row["fsps_reduced_chi2"], 2),
            }
        )
    return markdown_table(
        rows,
        ["held out", "picked from others", "held-out chi2", "oracle chi2", "FSPS chi2"],
    )


def wang_area_note():
    if not WANG_AREA.exists():
        return "- Wang area summary file not found in outputs."
    wang = pd.read_csv(WANG_AREA)
    if wang.empty:
        return "- Wang area summary file is empty."
    row = wang.iloc[0]
    return (
        f"- Wang master.dat has `{int(row['n_rows'])}` rows: "
        f"`{int(row['positive_cosmos2020_ids'])}` positive COSMOS2020 IDs and "
        f"`{int(row['negative_radio_only_ids'])}` negative radio-only IDs.\n"
        f"- The Wang/Farmer FLAG_COMBINED=0 area used for matched counts is "
        f"`{float(row['paper_farmer_flag_combined0_area_deg2']):.3f} deg2`, not the old `2.0 deg2` counterfactual."
    )


def main():
    score_summary = pd.read_csv(SCORE_SUMMARY)
    best_by_regime = pd.read_csv(BEST_BY_REGIME)
    source_tension = pd.read_csv(SOURCE_TENSION)
    flux_regime = pd.read_csv(FLUX_REGIME)
    leave_one = pd.read_csv(LEAVE_ONE)
    pd_sensitivity = pd.read_csv(PD_SENSITIVITY) if PD_SENSITIVITY.exists() else pd.DataFrame()
    dof_check = pd.read_csv(DOF_CHECK) if DOF_CHECK.exists() else pd.DataFrame()
    formal_summary = pd.read_csv(FORMAL_SUMMARY) if FORMAL_SUMMARY.exists() else pd.DataFrame()

    pooled_best = score_summary.iloc[0]
    pooled_fsps = score_summary[score_summary["model_label"] == "FSPS"].iloc[0]
    pooled_improvement = float(pooled_fsps["reduced_chi2_log"]) / float(
        pooled_best["reduced_chi2_log"]
    )

    if not formal_summary.empty:
        formal_best = formal_summary.iloc[0]
        formal_fsps = formal_summary[formal_summary["model_label"] == "FSPS"].iloc[0]
        formal_improvement = float(
            formal_fsps["formal_resolved_prior_chi2_over_dof"]
        ) / float(formal_best["formal_resolved_prior_chi2_over_dof"])
        formal_headline = f"""Thesis-facing formal score:

- formal data: corrected resolved/prior SPIRE differential counts
- best formal model: `{formal_best['model_label']}`
- formal rough chi2/(N-k): `{float(formal_best['formal_resolved_prior_chi2_over_dof']):.2f}`
- median log10(model/observed): `{float(formal_best['formal_resolved_prior_median_log_model_over_obs']):+.2f} dex`
- baseline FSPS formal chi2/(N-k): `{float(formal_fsps['formal_resolved_prior_chi2_over_dof']):.2f}`
- baseline FSPS is about `{formal_improvement:.1f}x` worse than the current best formal model
"""
    else:
        formal_headline = "Formal resolved/prior evaluator summary was not found."

    text = f"""# pop-cosmos FIR evaluator snapshot

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Current Headline

The thesis-facing formal evaluator now uses corrected resolved/prior SPIRE differential counts as the main score, with P(D) counts kept as a sensitivity check.

{formal_headline}

Pooled all-count sanity check:

- best pooled model: `{pooled_best['model_label']}`
- pooled rough chi2/N: `{float(pooled_best['reduced_chi2_log']):.2f}`
- pooled median log10(model/observed): `{float(pooled_best['median_log10_model_over_obs']):+.2f} dex`
- baseline FSPS pooled rough chi2/N: `{float(pooled_fsps['reduced_chi2_log']):.2f}`
- baseline FSPS is about `{pooled_improvement:.1f}x` worse than the current best pooled model

Simple thesis wording:

> The exact best dust template is not unique, but the direction is stable. Corrected SPIRE differential counts prefer a warmer/broader FIR dust SED than the baseline FSPS/pop-cosmos far-IR treatment, while keeping each galaxy's original L_IR fixed.

## Thesis-Facing Formal Score

{formal_summary_table(formal_summary)}

Interpretation:

- formal score = Clements / Oliver / Pearson resolved-or-prior counts
- P(D) is shown separately because those constraints are statistical and correlated
- Wang remains a matched-object diagnostic, not the formal count truth

## Overall Model Ranking

{top_model_table(score_summary)}

## Best Model By Count Regime

{best_regime_table(best_by_regime)}

## P(D) Sensitivity

{pd_sensitivity_table(pd_sensitivity) if not pd_sensitivity.empty else "- P(D) sensitivity table not found."}

Interpretation:

- all counts together prefer `Casey T30K a=2.5`
- resolved/prior counts alone prefer `Casey T30K a=3.0`
- P(D) statistical counts alone prefer `Casey T30K a=2.5`
- this is not exactly identical, but it is the same broad warm/Casey-like dust family

## Chi-Square Wording Check

{dof_check_table(dof_check)}

Interpretation:

- the current score is `chi2 / N_points`
- a simple `chi2 / (N_points - k)` correction barely changes the result
- still, for thesis wording I should call it a rough chi-square score unless I build a stricter likelihood model

## Source-To-Source Tension

{source_tension_table(source_tension)}

Interpretation:

- different count products do not pick the exact same template
- every source still prefers something better than baseline FSPS
- this supports "dust SED flexibility is the lever", not "one magic template is truth"

## Leave-One-Source-Out Check

{leave_one_table(leave_one)}

Interpretation:

- the selected warm-dust correction beats FSPS on every held-out count source
- it is not always the held-out oracle/best possible model
- that is good nuance: the result generalises, but the external counts have real tension

## FSPS Flux-Regime Residuals

{fsps_flux_regime_table(flux_regime)}

Interpretation:

- FSPS is high almost everywhere in the SPIRE comparison
- the bright end is the worst regime
- the overprediction gets worse from 250 to 350 to 500 um

Simple wording:

> FSPS is not terrible at 250 um low/mid fluxes, but it increasingly overpredicts counts at longer wavelengths and especially at the bright end.

## Wang Catalogue Role

{wang_area_note()}

Current use:

- Wang is best kept as a matched-object sanity check
- published corrected differential counts should carry the formal number-count evaluator
- raw Wang counts depend on prior selection, radio-only sources, area choice, and SNR/flux cuts

## Caveats To Say Out Loud

- P(D) constraints are statistical/map-level constraints and their bins are correlated.
- Varnish is included with approximate symmetric errors from published lower/upper bounds, so treat it as a sensitivity test.
- The evaluator is diagnostic, not a final likelihood model.
- A high reduced chi2 means mismatch / missing physics / underestimated errors, not automatically overfitting.
- Overfitting is checked with source hold-out tests and physical guardrails.

## Next Best Questions

1. Is the Casey-like grid enough as the thesis model-extension demonstration, or should I compare proper CIGALE/Dale/Draine-Li templates too?
2. Should Varnish P(D) stay in the formal chi-square score with approximate errors, or be visual-only?
3. Should the thesis headline be the bright-end SPIRE excess, or the broader conclusion that FIR dust SED shape is the weak lever?
4. For final plots, should Wang be shown only as a matched-object diagnostic, separate from the published-count evaluator?
"""

    OUT_MD.write_text(text, encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
