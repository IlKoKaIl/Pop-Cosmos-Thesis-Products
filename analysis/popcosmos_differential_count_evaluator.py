"""First-pass differential-count evaluator for pop-cosmos FIR templates.

This is intentionally a small scorecard rather than a final statistical model.

Inputs:
- corrected external SPIRE differential counts compiled from papers
- pop-cosmos/Wang-matched differential counts from the rest-frame hybrid test

Main idea:
- use differential counts because the flux-bin errors are closer to independent
- compare in log space because counts span orders of magnitude
- keep an error floor because the published surveys differ in extraction,
  completeness corrections, fields, and cosmic variance

The current model-count file is generated with the Wang/COSMOS2020-Farmer
area=1.278 deg^2. The evaluator also keeps an old 2.0 deg^2 counterfactual
so earlier plots can be understood.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
NB_DIR = CODE_DIR.parent
ROOT = NB_DIR.parent
OUT_DIR = NB_DIR / "outputs"
EXTERNAL_COUNTS = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
CLEAN_SOURCE_TABLE = ROOT / "catalog data/external_number_counts/external_spire_clean_independent_count_sources.csv"
MODEL_COUNTS = OUT_DIR / "popcosmos_restframe_hybrid_sed_differential_counts.csv"

MODEL_COUNT_AREA_DEG2 = 1.278
AREA_SCENARIOS = {
    "wang_farmer_1p278deg2": 1.278,
    "old_2deg2_counterfactual": 2.0,
}

MIN_FLUX_MJY = 10.0
MAX_FLUX_MJY = 300.0
LOG10_ERROR_FLOOR_DEX = 0.08  # about 20 percent in linear space

MODEL_LABELS = {
    "fsps_lirnorm": "FSPS",
    "resthybrid25": "25% ALESS",
    "resthybrid50": "50% ALESS",
    "resthybrid75": "75% ALESS",
    "aless": "ALESS",
    "wang_raw": "Wang raw",
}

MODEL_ORDER = ["fsps_lirnorm", "resthybrid25", "resthybrid50", "resthybrid75", "aless", "wang_raw"]


def log_interp(x, xp, fp):
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    good = np.isfinite(xp) & np.isfinite(fp) & (xp > 0) & (fp > 0)
    xp = xp[good]
    fp = fp[good]
    out = np.full_like(x, np.nan, dtype=float)
    if len(xp) < 2:
        return out
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    ok = np.isfinite(x) & (x >= xp.min()) & (x <= xp.max()) & (x > 0)
    out[ok] = 10 ** np.interp(np.log10(x[ok]), np.log10(xp), np.log10(fp))
    return out


def prepare_external_counts():
    external = pd.read_csv(EXTERNAL_COUNTS)
    external["euclidean_best_jy15_deg2"] = pd.to_numeric(
        external["euclidean_best_jy15_deg2"], errors="coerce"
    )
    external["euclidean_err_jy15_deg2"] = pd.to_numeric(
        external["euclidean_err_jy15_deg2"], errors="coerce"
    )
    external["flux_mjy"] = pd.to_numeric(external["flux_mjy"], errors="coerce")
    external = external[
        np.isfinite(external["flux_mjy"])
        & np.isfinite(external["euclidean_best_jy15_deg2"])
        & np.isfinite(external["euclidean_err_jy15_deg2"])
        & (external["euclidean_best_jy15_deg2"] > 0)
        & (external["euclidean_err_jy15_deg2"] > 0)
        & (external["flux_mjy"] >= MIN_FLUX_MJY)
        & (external["flux_mjy"] <= MAX_FLUX_MJY)
    ].copy()

    # The very wide H-ATLAS bright-end points are useful, but extremely bright
    # sparse bins can dominate a first-pass score. Keep the range explicit.
    external["source_key"] = external["paper"] + " / " + external["method_or_table"]
    return external


def prepare_clean_independent_external_counts():
    """Use only the count sources marked as clean independent anchors."""
    external = prepare_external_counts()
    clean = pd.read_csv(CLEAN_SOURCE_TABLE)
    clean = clean[clean["use_in_clean_chi_square"].astype(str).str.lower() == "yes"].copy()
    clean = clean[["clean_source_key", "paper", "method_or_table", "sky_field", "survey_family", "role"]]
    merged = external.merge(clean, on=["paper", "method_or_table"], how="inner")
    if merged.empty:
        raise ValueError(f"No clean independent rows matched {CLEAN_SOURCE_TABLE}")
    return merged


def prepare_model_counts(area_deg2):
    model = pd.read_csv(MODEL_COUNTS)
    scale = MODEL_COUNT_AREA_DEG2 / area_deg2
    for col in ["euclidean_jy15_deg2", "euclidean_err_jy15_deg2"]:
        model[col] = pd.to_numeric(model[col], errors="coerce") * scale
    model["flux_mjy"] = pd.to_numeric(model["flux_mjy"], errors="coerce")
    return model


def compare_one_model_to_external(model_counts, external_group, band, model_name):
    m = model_counts[
        (model_counts["band_um"] == band)
        & (model_counts["model"] == model_name)
        & (model_counts["N_bin"] > 0)
        & (model_counts["euclidean_jy15_deg2"] > 0)
    ].copy()
    if len(m) < 2:
        return None

    obs = external_group[external_group["band_um"] == band].copy()
    if obs.empty:
        return None

    pred = log_interp(
        obs["flux_mjy"].to_numpy(float),
        m["flux_mjy"].to_numpy(float),
        m["euclidean_jy15_deg2"].to_numpy(float),
    )
    pred_err = log_interp(
        obs["flux_mjy"].to_numpy(float),
        m["flux_mjy"].to_numpy(float),
        m["euclidean_err_jy15_deg2"].to_numpy(float),
    )

    obs_y = obs["euclidean_best_jy15_deg2"].to_numpy(float)
    obs_err = obs["euclidean_err_jy15_deg2"].to_numpy(float)
    ok = np.isfinite(pred) & np.isfinite(obs_y) & np.isfinite(obs_err) & (pred > 0) & (obs_y > 0)
    if ok.sum() < 2:
        return None

    obs_y = obs_y[ok]
    obs_err = obs_err[ok]
    pred = pred[ok]
    pred_err = pred_err[ok]

    obs_log = np.log10(obs_y)
    pred_log = np.log10(pred)
    obs_log_err = obs_err / (obs_y * np.log(10))

    pred_log_err = np.zeros_like(obs_log_err)
    good_pred_err = np.isfinite(pred_err) & (pred_err > 0)
    pred_log_err[good_pred_err] = pred_err[good_pred_err] / (pred[good_pred_err] * np.log(10))

    sigma_log = np.sqrt(obs_log_err**2 + pred_log_err**2 + LOG10_ERROR_FLOOR_DEX**2)
    delta_log = pred_log - obs_log
    chi2 = float(np.sum((delta_log / sigma_log) ** 2))
    n = int(ok.sum())
    dof = max(n, 1)

    return {
        "band_um": band,
        "model": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "N_points": n,
        "chi2_log": chi2,
        "reduced_chi2_log": chi2 / dof,
        "median_log10_model_over_obs": float(np.nanmedian(delta_log)),
        "rms_log10_model_over_obs": float(np.sqrt(np.nanmean(delta_log**2))),
        "mean_abs_log10_model_over_obs": float(np.nanmean(np.abs(delta_log))),
    }


def make_scorecard():
    external = prepare_external_counts()
    rows = []

    for scenario, area in AREA_SCENARIOS.items():
        model_counts = prepare_model_counts(area)
        for source_key, group in external.groupby("source_key"):
            for model_name in MODEL_ORDER:
                band_rows = []
                for band in [250, 350, 500]:
                    result = compare_one_model_to_external(model_counts, group, band, model_name)
                    if result is not None:
                        band_rows.append(result)
                        result.update(
                            {
                                "area_scenario": scenario,
                                "area_deg2": area,
                                "external_source": source_key,
                            }
                        )
                        rows.append(result)

                if band_rows:
                    n_total = sum(r["N_points"] for r in band_rows)
                    chi2_total = sum(r["chi2_log"] for r in band_rows)
                    deltas = np.repeat(
                        [r["median_log10_model_over_obs"] for r in band_rows],
                        [r["N_points"] for r in band_rows],
                    )
                    rows.append(
                        {
                            "area_scenario": scenario,
                            "area_deg2": area,
                            "external_source": source_key,
                            "band_um": "all",
                            "model": model_name,
                            "model_label": MODEL_LABELS.get(model_name, model_name),
                            "N_points": n_total,
                            "chi2_log": chi2_total,
                            "reduced_chi2_log": chi2_total / max(n_total, 1),
                            "median_log10_model_over_obs": float(np.nanmedian(deltas)),
                            "rms_log10_model_over_obs": np.nan,
                            "mean_abs_log10_model_over_obs": np.nan,
                        }
                    )

    out = pd.DataFrame(rows)
    out = out.sort_values(["area_scenario", "external_source", "band_um", "model"])
    return out


def make_clean_independent_scorecard():
    external = prepare_clean_independent_external_counts()
    rows = []

    for scenario, area in AREA_SCENARIOS.items():
        model_counts = prepare_model_counts(area)
        for clean_source_key, group in external.groupby("clean_source_key"):
            for model_name in MODEL_ORDER:
                band_rows = []
                for band in [250, 350, 500]:
                    result = compare_one_model_to_external(model_counts, group, band, model_name)
                    if result is not None:
                        band_rows.append(result)
                        result.update(
                            {
                                "area_scenario": scenario,
                                "area_deg2": area,
                                "external_source": clean_source_key,
                                "sky_field": "; ".join(sorted(group["sky_field"].dropna().unique())),
                                "survey_family": "; ".join(sorted(group["survey_family"].dropna().unique())),
                                "score_set": "clean_independent",
                            }
                        )
                        rows.append(result)

                if band_rows:
                    n_total = sum(r["N_points"] for r in band_rows)
                    chi2_total = sum(r["chi2_log"] for r in band_rows)
                    deltas = np.repeat(
                        [r["median_log10_model_over_obs"] for r in band_rows],
                        [r["N_points"] for r in band_rows],
                    )
                    rows.append(
                        {
                            "area_scenario": scenario,
                            "area_deg2": area,
                            "external_source": clean_source_key,
                            "sky_field": "; ".join(sorted(group["sky_field"].dropna().unique())),
                            "survey_family": "; ".join(sorted(group["survey_family"].dropna().unique())),
                            "score_set": "clean_independent",
                            "band_um": "all",
                            "model": model_name,
                            "model_label": MODEL_LABELS.get(model_name, model_name),
                            "N_points": n_total,
                            "chi2_log": chi2_total,
                            "reduced_chi2_log": chi2_total / max(n_total, 1),
                            "median_log10_model_over_obs": float(np.nanmedian(deltas)),
                            "rms_log10_model_over_obs": np.nan,
                            "mean_abs_log10_model_over_obs": np.nan,
                        }
                    )

    out = pd.DataFrame(rows)
    out = out.sort_values(["area_scenario", "external_source", "band_um", "model"])
    return out


def plot_scorecard(scorecard):
    sub = scorecard[
        (scorecard["area_scenario"] == "wang_farmer_1p278deg2")
        & (scorecard["band_um"].astype(str) == "all")
        & (scorecard["model"] != "wang_raw")
    ].copy()
    if sub.empty:
        return None

    pivot = sub.pivot_table(
        index="external_source",
        columns="model_label",
        values="reduced_chi2_log",
        aggfunc="first",
    )
    cols = [MODEL_LABELS[m] for m in MODEL_ORDER if m != "wang_raw" and MODEL_LABELS[m] in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(np.log10(pivot.to_numpy(float)), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(
        "First-pass evaluator: log10 reduced chi-square vs published differential counts\n"
        "Wang/Farmer area=1.278 deg^2, log-space score with 0.08 dex error floor"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\chi^2_\nu)$")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="white" if val > 10 else "black")

    fig.tight_layout()
    path = OUT_DIR / "popcosmos_differential_count_evaluator_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_clean_independent_scorecard(scorecard):
    sub = scorecard[
        (scorecard["area_scenario"] == "wang_farmer_1p278deg2")
        & (scorecard["band_um"].astype(str) == "all")
        & (scorecard["model"] != "wang_raw")
    ].copy()
    if sub.empty:
        return None

    pivot = sub.pivot_table(
        index="external_source",
        columns="model_label",
        values="reduced_chi2_log",
        aggfunc="first",
    )
    cols = [MODEL_LABELS[m] for m in MODEL_ORDER if m != "wang_raw" and MODEL_LABELS[m] in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    im = ax.imshow(np.log10(pivot.to_numpy(float)), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(
        "Clean independent-count evaluator\n"
        "main anchors only: Valiante H-ATLAS, Oliver HerMES, Pearson Dark Field XID"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\chi^2_\nu)$")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="white" if val > 10 else "black")

    fig.tight_layout()
    path = OUT_DIR / "popcosmos_clean_independent_count_evaluator_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_pooled_summary(scorecard, out_path):
    pooled = scorecard[
        (scorecard["area_scenario"] == "wang_farmer_1p278deg2")
        & (scorecard["band_um"].astype(str) != "all")
        & (scorecard["model"] != "wang_raw")
    ].groupby("model_label", as_index=False).agg(
        N_points=("N_points", "sum"),
        chi2_log=("chi2_log", "sum"),
        median_log10_model_over_obs=("median_log10_model_over_obs", "median"),
    )
    pooled["reduced_chi2_log"] = pooled["chi2_log"] / pooled["N_points"]
    pooled = pooled.sort_values("reduced_chi2_log")
    pooled.to_csv(out_path, index=False)
    return pooled


def plot_area_corrected_overlay():
    external = pd.read_csv(EXTERNAL_COUNTS)
    external["euclidean_best_jy15_deg2"] = pd.to_numeric(
        external["euclidean_best_jy15_deg2"], errors="coerce"
    )
    external["euclidean_err_jy15_deg2"] = pd.to_numeric(
        external["euclidean_err_jy15_deg2"], errors="coerce"
    )
    external["flux_mjy"] = pd.to_numeric(external["flux_mjy"], errors="coerce")
    external = external[
        np.isfinite(external["flux_mjy"])
        & np.isfinite(external["euclidean_best_jy15_deg2"])
        & (external["euclidean_best_jy15_deg2"] > 0)
        & (external["flux_mjy"] >= 10)
        & (external["flux_mjy"] <= 300)
    ].copy()

    model = prepare_model_counts(AREA_SCENARIOS["wang_farmer_1p278deg2"])
    lines = [
        ("fsps_lirnorm", "FSPS", "#0072B2", "-"),
        ("resthybrid25", "25% ALESS", "#56B4E9", "--"),
        ("resthybrid50", "50% ALESS", "#D55E00", "-"),
        ("resthybrid75", "75% ALESS", "#E69F00", "--"),
        ("aless", "ALESS", "#E69F00", ":"),
        ("wang_raw", "Wang raw", "#000000", "-"),
    ]
    markers = {
        "Clements et al.": ("s", "#0072B2"),
        "Glenn et al.": ("o", "0.45"),
        "Oliver et al.": ("D", "#009E73"),
        "Pearson et al.": ("^", "#D55E00"),
        "Varnish et al.": ("o", "#CC79A7"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, band in zip(axes, [250, 350, 500]):
        for paper, group in external[external["band_um"] == band].groupby("paper"):
            marker, color = markers.get(paper, ("o", "0.5"))
            yerr = pd.to_numeric(group["euclidean_err_jy15_deg2"], errors="coerce")
            ax.errorbar(
                group["flux_mjy"],
                group["euclidean_best_jy15_deg2"],
                yerr=yerr if np.isfinite(yerr).any() else None,
                fmt=marker,
                color=color,
                alpha=0.55,
                ms=3.4,
                lw=0.7,
                capsize=1.2,
                label=paper,
            )

        for model_name, label, color, ls in lines:
            sub = model[(model["band_um"] == band) & (model["model"] == model_name)]
            ax.plot(
                sub["flux_mjy"],
                sub["euclidean_jy15_deg2"],
                color=color,
                ls=ls,
                lw=2.0 if model_name in {"fsps_lirnorm", "resthybrid50", "wang_raw"} else 1.4,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10, 300)
        ax.set_ylim(0.03, 30)
        ax.set_title(f"{band} um")
        ax.set_xlabel("Flux density S [mJy]")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"$S^{2.5}dN/dS$ [Jy$^{1.5}$ deg$^{-2}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[0].legend(dedup.values(), dedup.keys(), fontsize=6.8, ncol=1)
    fig.suptitle(
        "Differential counts overlay using Wang/Farmer area=1.278 deg^2\n"
        "Wang curve is raw positive-flux catalogue counts, not a published corrected count table"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = OUT_DIR / "popcosmos_differential_count_area_corrected_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_leave_one_source_out_validation(scorecard, area_scenario="wang_farmer_1p278deg2"):
    """Choose the best template on all-but-one sources and test on the held-out source."""
    base = scorecard[
        (scorecard["area_scenario"] == area_scenario)
        & (scorecard["band_um"].astype(str) == "all")
        & (scorecard["model"] != "wang_raw")
    ].copy()

    rows = []
    for heldout in sorted(base["external_source"].unique()):
        train = base[base["external_source"] != heldout].copy()
        test = base[base["external_source"] == heldout].copy()
        if train.empty or test.empty:
            continue

        train_score = train.groupby(["model", "model_label"], as_index=False).agg(
            train_N_points=("N_points", "sum"),
            train_chi2=("chi2_log", "sum"),
        )
        train_score["train_reduced_chi2"] = train_score["train_chi2"] / train_score["train_N_points"]
        picked = train_score.sort_values("train_reduced_chi2").iloc[0]

        picked_test = test[test["model"] == picked["model"]].iloc[0]
        oracle = test.sort_values("reduced_chi2_log").iloc[0]
        rows.append(
            {
                "heldout_external_source": heldout,
                "selected_model": picked["model"],
                "selected_model_label": picked["model_label"],
                "train_N_points": int(picked["train_N_points"]),
                "train_reduced_chi2": float(picked["train_reduced_chi2"]),
                "heldout_N_points": int(picked_test["N_points"]),
                "heldout_reduced_chi2": float(picked_test["reduced_chi2_log"]),
                "heldout_median_log10_model_over_obs": float(picked_test["median_log10_model_over_obs"]),
                "heldout_oracle_model": oracle["model"],
                "heldout_oracle_model_label": oracle["model_label"],
                "heldout_oracle_reduced_chi2": float(oracle["reduced_chi2_log"]),
                "selected_minus_oracle_reduced_chi2": float(
                    picked_test["reduced_chi2_log"] - oracle["reduced_chi2_log"]
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_leave_one_source_out(validation):
    if validation.empty:
        return None

    short_names = {
        "Clements et al. / Table 1": "Clements",
        "Glenn et al. / Table 4 P(D) spline no FIRAS": "Glenn P(D)",
        "Oliver et al. / Table 2": "Oliver",
        "Pearson et al. / Table 3 SUSSEXtractor": "Pearson SUSSEX",
        "Pearson et al. / Table 4 XID": "Pearson XID",
    }

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(validation))
    width = 0.38
    ax.bar(
        x - width / 2,
        validation["heldout_reduced_chi2"],
        width,
        label="model selected from other sources",
        color="#56B4E9",
    )
    ax.bar(
        x + width / 2,
        validation["heldout_oracle_reduced_chi2"],
        width,
        label="best possible on held-out source",
        color="#E69F00",
    )
    ax.set_ylabel(r"held-out reduced $\chi^2$")
    ax.set_title("Leave-one-source-out template validation")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [short_names.get(s, s) for s in validation["heldout_external_source"]],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)

    top = max(validation["heldout_reduced_chi2"].max(), validation["heldout_oracle_reduced_chi2"].max())
    ax.set_ylim(0, top * 1.25)
    for xi, label in zip(x, validation["selected_model_label"]):
        y = validation.loc[xi, "heldout_reduced_chi2"]
        ax.text(xi - width / 2, y + top * 0.035, label, ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    path = OUT_DIR / "popcosmos_differential_count_leave_one_source_out.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_regime_summary(scorecard, area_scenario="wang_farmer_1p278deg2"):
    """Aggregate scores by broad observational regime.

    This is a small overfitting guard. If the same template only wins for one
    paper, or only for P(D) statistical counts, that is weaker than also doing
    well for resolved/prior-extracted source counts.
    """
    base = scorecard[
        (scorecard["area_scenario"] == area_scenario)
        & (scorecard["band_um"].astype(str) != "all")
        & (scorecard["model"] != "wang_raw")
    ].copy()

    regimes = {
        "all_scored_counts": base,
        "resolved_or_prior_counts_only": base[~base["external_source"].str.contains(r"P\(D\)", regex=True)],
        "pd_statistical_counts_only": base[base["external_source"].str.contains(r"P\(D\)", regex=True)],
    }

    rows = []
    for regime, sub in regimes.items():
        if sub.empty:
            continue
        grouped = sub.groupby(["model", "model_label"], as_index=False).agg(
            N_points=("N_points", "sum"),
            chi2_log=("chi2_log", "sum"),
            median_log10_model_over_obs=("median_log10_model_over_obs", "median"),
        )
        grouped["reduced_chi2_log"] = grouped["chi2_log"] / grouped["N_points"]
        grouped["regime"] = regime
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["regime", "reduced_chi2_log"])
    return out


def plot_regime_summary(regime_summary):
    if regime_summary.empty:
        return None

    display = {
        "all_scored_counts": "all scored",
        "resolved_or_prior_counts_only": "resolved/prior",
        "pd_statistical_counts_only": "P(D) only",
    }
    model_labels = [MODEL_LABELS[m] for m in MODEL_ORDER if m != "wang_raw"]
    x = np.arange(len(model_labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for i, regime in enumerate(["all_scored_counts", "resolved_or_prior_counts_only", "pd_statistical_counts_only"]):
        sub = regime_summary[regime_summary["regime"] == regime].set_index("model_label")
        y = [sub.loc[label, "reduced_chi2_log"] if label in sub.index else np.nan for label in model_labels]
        ax.bar(
            x + (i - 1) * width,
            y,
            width,
            label=display.get(regime, regime),
        )

    ax.set_yscale("log")
    ax.set_ylabel(r"rough reduced $\chi^2$")
    ax.set_title("Evaluator split by observed-count type")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = OUT_DIR / "popcosmos_differential_count_evaluator_regime_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    scorecard = make_scorecard()
    out_csv = OUT_DIR / "popcosmos_differential_count_evaluator_scorecard.csv"
    scorecard.to_csv(out_csv, index=False)
    heatmap = plot_scorecard(scorecard)
    clean_scorecard = make_clean_independent_scorecard()
    clean_csv = OUT_DIR / "popcosmos_clean_independent_count_evaluator_scorecard.csv"
    clean_scorecard.to_csv(clean_csv, index=False)
    clean_heatmap = plot_clean_independent_scorecard(clean_scorecard)
    clean_pooled = make_pooled_summary(
        clean_scorecard,
        OUT_DIR / "popcosmos_clean_independent_count_evaluator_pooled_summary.csv",
    )
    overlay = plot_area_corrected_overlay()
    validation = make_leave_one_source_out_validation(scorecard)
    validation_csv = OUT_DIR / "popcosmos_differential_count_leave_one_source_out.csv"
    validation.to_csv(validation_csv, index=False)
    validation_plot = plot_leave_one_source_out(validation)
    regime_summary = make_regime_summary(scorecard)
    regime_csv = OUT_DIR / "popcosmos_differential_count_evaluator_regime_summary.csv"
    regime_summary.to_csv(regime_csv, index=False)
    regime_plot = plot_regime_summary(regime_summary)

    print(out_csv)
    if heatmap is not None:
        print(heatmap)
    print(clean_csv)
    if clean_heatmap is not None:
        print(clean_heatmap)
    print(overlay)
    print(validation_csv)
    if validation_plot is not None:
        print(validation_plot)
    print(regime_csv)
    if regime_plot is not None:
        print(regime_plot)

    best = scorecard[
        (scorecard["area_scenario"] == "wang_farmer_1p278deg2")
        & (scorecard["band_um"].astype(str) == "all")
        & (scorecard["model"] != "wang_raw")
    ].copy()
    best = best.sort_values(["external_source", "reduced_chi2_log"])
    print("\nBest model per external source, Wang/Farmer area scenario:")
    print(best.groupby("external_source").head(2)[
        ["external_source", "model_label", "N_points", "reduced_chi2_log", "median_log10_model_over_obs"]
    ].to_string(index=False))

    pooled = scorecard[
        (scorecard["area_scenario"] == "wang_farmer_1p278deg2")
        & (scorecard["band_um"].astype(str) != "all")
        & (scorecard["model"] != "wang_raw")
    ].groupby("model_label", as_index=False).agg(
        N_points=("N_points", "sum"),
        chi2_log=("chi2_log", "sum"),
        median_log10_model_over_obs=("median_log10_model_over_obs", "median"),
    )
    pooled["reduced_chi2_log"] = pooled["chi2_log"] / pooled["N_points"]
    pooled = pooled.sort_values("reduced_chi2_log")
    pooled.to_csv(OUT_DIR / "popcosmos_differential_count_evaluator_pooled_summary.csv", index=False)
    print("\nPooled rough summary:")
    print(pooled.to_string(index=False))

    print("\nClean independent pooled summary:")
    print(clean_pooled.to_string(index=False))

    print("\nLeave-one-source-out validation:")
    print(validation.to_string(index=False))

    print("\nRegime summary:")
    print(regime_summary.to_string(index=False))


if __name__ == "__main__":
    main()
