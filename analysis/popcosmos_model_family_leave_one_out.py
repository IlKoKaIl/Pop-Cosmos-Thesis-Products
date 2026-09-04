"""Leave-one-source-out validation for the full FIR model family.

This is an overfitting guard for the thesis evaluator.

The earlier leave-one-source-out check only tested the FSPS/ALESS hybrid
family. Now that the model set includes modified blackbodies and Casey-like
templates, this script repeats the same idea across the full current family.

For each external count source:

1. train on all other published count sources
2. choose the model with the lowest rough reduced chi-square
3. test that chosen model on the held-out source
4. compare against the best possible model on the held-out source
5. compare against baseline FSPS on the held-out source
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NB_DIR = Path(__file__).resolve().parent
OUT_DIR = NB_DIR / "outputs"

HYBRID_SCORECARD = OUT_DIR / "popcosmos_differential_count_evaluator_scorecard.csv"
MBB_SCORECARD = OUT_DIR / "popcosmos_mbb_temperature_grid_scorecard.csv"
CASEY_SCORECARD = OUT_DIR / "popcosmos_casey_like_template_grid_scorecard.csv"

OUT_PER_SOURCE = OUT_DIR / "popcosmos_model_family_per_source_scores.csv"
OUT_LEAVE_ONE = OUT_DIR / "popcosmos_model_family_leave_one_source_out.csv"
OUT_PLOT = OUT_DIR / "popcosmos_model_family_leave_one_source_out.png"


def load_hybrid():
    df = pd.read_csv(HYBRID_SCORECARD)
    df = df[df["area_scenario"] == "wang_farmer_1p278deg2"].copy()
    df["model_family"] = df["model_label"].map(
        lambda label: "baseline"
        if label == "FSPS"
        else "empirical template"
        if label == "ALESS"
        else "FSPS/ALESS hybrid"
    )
    df["model_uid"] = "hybrid::" + df["model"].astype(str)
    return df


def load_mbb():
    df = pd.read_csv(MBB_SCORECARD)
    df["model_label"] = pd.to_numeric(df["T_dust_K"]).map(lambda t: f"MBB {int(t)} K")
    df["model_family"] = "modified blackbody"
    df["model_uid"] = "mbb::" + df["model"].astype(str)
    return df


def load_casey():
    df = pd.read_csv(CASEY_SCORECARD)
    df["model_family"] = "Casey-like template"
    df["model_uid"] = "casey::" + df["model"].astype(str)
    return df


def load_all_scorecards():
    pieces = [load_hybrid(), load_mbb()]
    if CASEY_SCORECARD.exists():
        pieces.append(load_casey())

    cols = [
        "external_source",
        "band_um",
        "model_uid",
        "model",
        "model_label",
        "model_family",
        "N_points",
        "chi2_log",
        "median_log10_model_over_obs",
    ]
    df = pd.concat([piece[cols] for piece in pieces], ignore_index=True)
    df["N_points"] = pd.to_numeric(df["N_points"], errors="coerce")
    df["chi2_log"] = pd.to_numeric(df["chi2_log"], errors="coerce")
    df["median_log10_model_over_obs"] = pd.to_numeric(
        df["median_log10_model_over_obs"], errors="coerce"
    )
    return df.dropna(subset=["N_points", "chi2_log"])


def aggregate_by_model(df, group_cols):
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            N_points=("N_points", "sum"),
            chi2_log=("chi2_log", "sum"),
            median_log10_model_over_obs=("median_log10_model_over_obs", "median"),
        )
        .copy()
    )
    grouped["reduced_chi2_log"] = grouped["chi2_log"] / grouped["N_points"].clip(lower=1)
    return grouped.sort_values("reduced_chi2_log")


def make_per_source_scores(scorecards):
    return aggregate_by_model(
        scorecards,
        ["external_source", "model_uid", "model", "model_label", "model_family"],
    )


def leave_one_source_out(scorecards, per_source):
    rows = []
    sources = sorted(scorecards["external_source"].dropna().unique())

    for heldout in sources:
        train = scorecards[scorecards["external_source"] != heldout]
        train_scores = aggregate_by_model(
            train,
            ["model_uid", "model", "model_label", "model_family"],
        )
        picked = train_scores.iloc[0]

        heldout_scores = per_source[per_source["external_source"] == heldout].copy()
        picked_test = heldout_scores[heldout_scores["model_uid"] == picked["model_uid"]]
        if picked_test.empty:
            continue
        picked_test = picked_test.iloc[0]
        oracle = heldout_scores.sort_values("reduced_chi2_log").iloc[0]

        fsps = heldout_scores[heldout_scores["model_label"] == "FSPS"]
        fsps_score = float(fsps["reduced_chi2_log"].iloc[0]) if not fsps.empty else np.nan

        rows.append(
            {
                "heldout_external_source": heldout,
                "selected_model_family": picked["model_family"],
                "selected_model_label": picked["model_label"],
                "train_N_points": int(picked["N_points"]),
                "train_reduced_chi2": float(picked["reduced_chi2_log"]),
                "heldout_N_points": int(picked_test["N_points"]),
                "heldout_reduced_chi2": float(picked_test["reduced_chi2_log"]),
                "heldout_median_log10_model_over_obs": float(
                    picked_test["median_log10_model_over_obs"]
                ),
                "heldout_oracle_model_family": oracle["model_family"],
                "heldout_oracle_model_label": oracle["model_label"],
                "heldout_oracle_reduced_chi2": float(oracle["reduced_chi2_log"]),
                "fsps_reduced_chi2": fsps_score,
                "selected_minus_oracle_reduced_chi2": float(
                    picked_test["reduced_chi2_log"] - oracle["reduced_chi2_log"]
                ),
                "selected_minus_fsps_reduced_chi2": float(
                    picked_test["reduced_chi2_log"] - fsps_score
                )
                if np.isfinite(fsps_score)
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def short_source_name(source):
    replacements = {
        "Clements et al. / Table 1": "Clements",
        "Glenn et al. / Table 4 P(D) spline no FIRAS": "Glenn P(D)",
        "Oliver et al. / Table 2": "Oliver",
        "Pearson et al. / Table 3 SUSSEXtractor": "Pearson SUSSEX",
        "Pearson et al. / Table 4 XID": "Pearson XID",
        "Varnish et al. / Table 4 P(D) best-fit spline": "Varnish P(D)",
    }
    return replacements.get(source, source)


def plot_leave_one(validation):
    labels = [short_source_name(src) for src in validation["heldout_external_source"]]
    x = np.arange(len(validation))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(
        x - width,
        validation["heldout_reduced_chi2"],
        width,
        label="selected from other sources",
        color="#CC79A7",
    )
    ax.bar(
        x,
        validation["heldout_oracle_reduced_chi2"],
        width,
        label="best possible on held-out source",
        color="#009E73",
    )
    ax.bar(
        x + width,
        validation["fsps_reduced_chi2"],
        width,
        label="baseline FSPS",
        color="#0072B2",
    )

    ymax = np.nanmax(
        [
            validation["heldout_reduced_chi2"].max(),
            validation["heldout_oracle_reduced_chi2"].max(),
            validation["fsps_reduced_chi2"].max(),
        ]
    )
    ax.set_ylim(0, ymax * 1.28)
    for xi, row in validation.iterrows():
        ax.text(
            xi - width,
            row["heldout_reduced_chi2"] + ymax * 0.025,
            row["selected_model_label"],
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(r"held-out rough reduced $\chi^2$")
    ax.set_title("Full FIR model-family leave-one-count-source-out check")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=180)
    plt.close(fig)
    return OUT_PLOT


def main():
    scorecards = load_all_scorecards()
    per_source = make_per_source_scores(scorecards)
    validation = leave_one_source_out(scorecards, per_source)

    per_source.to_csv(OUT_PER_SOURCE, index=False)
    validation.to_csv(OUT_LEAVE_ONE, index=False)
    plot_path = plot_leave_one(validation)

    print(OUT_PER_SOURCE)
    print(OUT_LEAVE_ONE)
    print(plot_path)
    print("\nLeave-one-source-out summary:")
    print(
        validation[
            [
                "heldout_external_source",
                "selected_model_label",
                "heldout_reduced_chi2",
                "heldout_oracle_model_label",
                "heldout_oracle_reduced_chi2",
                "fsps_reduced_chi2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
