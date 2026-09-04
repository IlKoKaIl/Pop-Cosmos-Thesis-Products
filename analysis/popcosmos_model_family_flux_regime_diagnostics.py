"""Flux-regime diagnostics for the FIR count evaluator.

This answers a practical thesis question:

    Where does baseline FSPS fail?

Instead of one pooled chi-square, split the observed SPIRE differential-count
comparison into broad flux regimes:

- 10-30 mJy: faint/resolved-confusion edge
- 30-100 mJy: middle bright counts
- 100-300 mJy: rare bright end

The exact regime labels are deliberately simple. The point is to see whether
the model mismatch is a faint-end issue, a bright-end issue, or broad across
the whole SPIRE comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
OUT_DIR = NB_DIR / "outputs"

EXTERNAL_COUNTS = ROOT / "catalog data/external_number_counts/external_spire_differential_counts_compiled.csv"
HYBRID_COUNTS = OUT_DIR / "popcosmos_restframe_hybrid_sed_differential_counts.csv"
MBB_COUNTS = OUT_DIR / "popcosmos_mbb_temperature_grid_differential_counts.csv"
CASEY_COUNTS = OUT_DIR / "popcosmos_casey_like_template_grid_differential_counts.csv"

OUT_RESIDUALS = OUT_DIR / "popcosmos_model_family_flux_regime_residuals.csv"
OUT_SUMMARY = OUT_DIR / "popcosmos_model_family_flux_regime_summary.csv"
OUT_HEATMAP = OUT_DIR / "popcosmos_model_family_flux_regime_residual_heatmap.png"
OUT_CHI2 = OUT_DIR / "popcosmos_model_family_flux_regime_chi2.png"

MIN_FLUX_MJY = 10.0
MAX_FLUX_MJY = 300.0
LOG10_ERROR_FLOOR_DEX = 0.08

KEY_MODEL_ORDER = [
    "FSPS",
    "25% ALESS",
    "50% ALESS",
    "MBB 35 K",
    "Casey T30K a=2.5",
    "Casey T30K a=3.0",
    "ALESS",
]

FLUX_REGIMES = [
    (10.0, 30.0, "10-30 mJy"),
    (30.0, 100.0, "30-100 mJy"),
    (100.0, 300.0, "100-300 mJy"),
]


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
    out[ok] = 10.0 ** np.interp(np.log10(x[ok]), np.log10(xp), np.log10(fp))
    return out


def flux_regime(flux_mjy):
    for lo, hi, label in FLUX_REGIMES:
        if lo <= flux_mjy < hi:
            return label
    if np.isclose(flux_mjy, FLUX_REGIMES[-1][1]):
        return FLUX_REGIMES[-1][2]
    return None


def load_external():
    external = pd.read_csv(EXTERNAL_COUNTS)
    for col in ["band_um", "flux_mjy", "euclidean_best_jy15_deg2", "euclidean_err_jy15_deg2"]:
        external[col] = pd.to_numeric(external[col], errors="coerce")
    external = external[
        np.isfinite(external["band_um"])
        & np.isfinite(external["flux_mjy"])
        & np.isfinite(external["euclidean_best_jy15_deg2"])
        & np.isfinite(external["euclidean_err_jy15_deg2"])
        & (external["flux_mjy"] >= MIN_FLUX_MJY)
        & (external["flux_mjy"] <= MAX_FLUX_MJY)
        & (external["euclidean_best_jy15_deg2"] > 0)
        & (external["euclidean_err_jy15_deg2"] > 0)
    ].copy()
    external["band_um"] = external["band_um"].astype(int)
    external["external_source"] = external["paper"] + " / " + external["method_or_table"]
    external["flux_regime"] = external["flux_mjy"].map(flux_regime)
    return external.dropna(subset=["flux_regime"])


def load_model_counts():
    pieces = []

    hybrid_labels = {
        "fsps_lirnorm": "FSPS",
        "resthybrid25": "25% ALESS",
        "resthybrid50": "50% ALESS",
        "resthybrid75": "75% ALESS",
        "aless": "ALESS",
    }
    hybrid = pd.read_csv(HYBRID_COUNTS)
    hybrid["model_label"] = hybrid["model"].map(hybrid_labels)
    pieces.append(hybrid.dropna(subset=["model_label"]))

    mbb = pd.read_csv(MBB_COUNTS)
    mbb["model_label"] = pd.to_numeric(mbb["model"].str.extract(r"T(\d+)")[0]).map(
        lambda t: f"MBB {int(t)} K" if pd.notna(t) else np.nan
    )
    pieces.append(mbb.dropna(subset=["model_label"]))

    casey = pd.read_csv(CASEY_COUNTS)
    pieces.append(casey.dropna(subset=["model_label"]))

    model = pd.concat(pieces, ignore_index=True)
    for col in ["band_um", "flux_mjy", "N_bin", "euclidean_jy15_deg2", "euclidean_err_jy15_deg2"]:
        model[col] = pd.to_numeric(model[col], errors="coerce")
    model = model[
        model["model_label"].isin(KEY_MODEL_ORDER)
        & np.isfinite(model["band_um"])
        & np.isfinite(model["flux_mjy"])
        & np.isfinite(model["euclidean_jy15_deg2"])
        & (model["N_bin"] > 0)
        & (model["euclidean_jy15_deg2"] > 0)
    ].copy()
    model["band_um"] = model["band_um"].astype(int)
    return model


def make_residuals(external, model_counts):
    rows = []
    for model_label, model_group in model_counts.groupby("model_label"):
        for source, source_group in external.groupby("external_source"):
            for band, obs in source_group.groupby("band_um"):
                m = model_group[model_group["band_um"] == band]
                if len(m) < 2 or obs.empty:
                    continue

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
                obs_flux = obs["flux_mjy"].to_numpy(float)
                ok = np.isfinite(pred) & np.isfinite(obs_y) & (pred > 0) & (obs_y > 0)
                if not ok.any():
                    continue

                obs_log_err = obs_err[ok] / (obs_y[ok] * np.log(10.0))
                pred_log_err = np.zeros_like(obs_log_err)
                good_pred_err = np.isfinite(pred_err[ok]) & (pred_err[ok] > 0)
                pred_log_err[good_pred_err] = pred_err[ok][good_pred_err] / (
                    pred[ok][good_pred_err] * np.log(10.0)
                )
                sigma_log = np.sqrt(obs_log_err**2 + pred_log_err**2 + LOG10_ERROR_FLOOR_DEX**2)
                delta_log = np.log10(pred[ok]) - np.log10(obs_y[ok])

                for i, row_idx in enumerate(obs.index[ok]):
                    rows.append(
                        {
                            "external_source": source,
                            "paper": obs.loc[row_idx, "paper"],
                            "method_or_table": obs.loc[row_idx, "method_or_table"],
                            "band_um": int(band),
                            "flux_mjy": obs_flux[ok][i],
                            "flux_regime": obs.loc[row_idx, "flux_regime"],
                            "model_label": model_label,
                            "observed_euclidean_jy15_deg2": obs_y[ok][i],
                            "model_euclidean_jy15_deg2": pred[ok][i],
                            "log10_model_over_obs": delta_log[i],
                            "sigma_log": sigma_log[i],
                            "chi2_log": (delta_log[i] / sigma_log[i]) ** 2,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_residuals(residuals):
    group_cols = ["model_label", "band_um", "flux_regime"]
    summary = (
        residuals.groupby(group_cols, as_index=False)
        .agg(
            N_points=("chi2_log", "size"),
            chi2_log=("chi2_log", "sum"),
            median_log10_model_over_obs=("log10_model_over_obs", "median"),
            mean_abs_log10_model_over_obs=("log10_model_over_obs", lambda x: np.mean(np.abs(x))),
        )
        .copy()
    )
    summary["reduced_chi2_log"] = summary["chi2_log"] / summary["N_points"].clip(lower=1)
    summary["model_rank"] = summary["model_label"].map(
        {label: i for i, label in enumerate(KEY_MODEL_ORDER)}
    )
    summary["regime_rank"] = summary["flux_regime"].map(
        {label: i for i, (_, _, label) in enumerate(FLUX_REGIMES)}
    )
    return summary.sort_values(["model_rank", "band_um", "regime_rank"]).drop(
        columns=["model_rank", "regime_rank"]
    )


def plot_residual_heatmap(summary):
    summary = summary.copy()
    summary["column"] = summary["band_um"].astype(str) + "um\n" + summary["flux_regime"]
    column_order = [
        f"{band}um\n{label}"
        for band in [250, 350, 500]
        for _, _, label in FLUX_REGIMES
    ]
    pivot = summary.pivot_table(
        index="model_label",
        columns="column",
        values="median_log10_model_over_obs",
        aggfunc="first",
    )
    pivot = pivot.loc[[m for m in KEY_MODEL_ORDER if m in pivot.index], [c for c in column_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    data = pivot.to_numpy(float)
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-0.55, vmax=0.55)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Median log10(model / observed) by band and flux regime")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("dex; positive = model too high")
    fig.tight_layout()
    fig.savefig(OUT_HEATMAP, dpi=180)
    plt.close(fig)
    return OUT_HEATMAP


def plot_chi2(summary):
    key = summary[summary["model_label"].isin(["FSPS", "MBB 35 K", "Casey T30K a=2.5", "Casey T30K a=3.0"])].copy()
    key["x_label"] = key["band_um"].astype(str) + "um " + key["flux_regime"]
    x_order = [
        f"{band}um {label}"
        for band in [250, 350, 500]
        for _, _, label in FLUX_REGIMES
    ]
    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    for model_label in ["FSPS", "MBB 35 K", "Casey T30K a=2.5", "Casey T30K a=3.0"]:
        sub = key[key["model_label"] == model_label].set_index("x_label")
        y = [sub.loc[x, "reduced_chi2_log"] if x in sub.index else np.nan for x in x_order]
        ax.plot(x_order, y, marker="o", lw=1.8, label=model_label)
    ax.set_yscale("log")
    ax.set_ylabel(r"rough reduced $\chi^2$")
    ax.set_title("Flux-regime score: baseline FSPS vs warm-dust corrections")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_CHI2, dpi=180)
    plt.close(fig)
    return OUT_CHI2


def main():
    external = load_external()
    model_counts = load_model_counts()
    residuals = make_residuals(external, model_counts)
    summary = summarize_residuals(residuals)

    residuals.to_csv(OUT_RESIDUALS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    heatmap = plot_residual_heatmap(summary)
    chi2_plot = plot_chi2(summary)

    print(OUT_RESIDUALS)
    print(OUT_SUMMARY)
    print(heatmap)
    print(chi2_plot)
    print("\nFSPS flux-regime summary:")
    print(
        summary[summary["model_label"] == "FSPS"][
            [
                "band_um",
                "flux_regime",
                "N_points",
                "reduced_chi2_log",
                "median_log10_model_over_obs",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
