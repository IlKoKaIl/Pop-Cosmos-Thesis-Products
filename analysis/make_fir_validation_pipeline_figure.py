"""Create the report-facing schematic of the pop-cosmos FIR validation pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


CODE_DIR = Path(__file__).resolve().parent
NB_DIR = CODE_DIR.parent
FIGURE_DIR = NB_DIR / "thesis_figures"
OUTPUT_STEM = FIGURE_DIR / "popcosmos_fir_validation_pipeline"


COLORS = {
    "ink": "#17212B",
    "muted": "#53606C",
    "model": "#DCECF2",
    "model_edge": "#2E6F89",
    "transform": "#F8E7C4",
    "transform_edge": "#A86B00",
    "test": "#E7E3F3",
    "test_edge": "#65558F",
    "observation": "#DDEDDD",
    "observation_edge": "#397047",
    "background": "#FFFFFF",
}


def add_box(
    ax,
    xy,
    width,
    height,
    title,
    body,
    facecolor,
    edgecolor,
    *,
    title_size=11.0,
    body_size=8.8,
    linewidth=1.5,
):
    """Add a compact rounded box in axes coordinates."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["muted"],
        linespacing=1.25,
        transform=ax.transAxes,
    )
    return patch


def add_arrow(ax, start, end, *, color=None, connectionstyle="arc3", linewidth=1.7):
    """Add an arrow between points given in axes coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color or COLORS["ink"],
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        clip_on=False,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    return arrow


def build_figure():
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "From pop-cosmos predictions to far-infrared validation",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.915,
        "The same predicted SPIRE fluxes feed two complementary observational tests",
        ha="center",
        va="top",
        fontsize=10.5,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    # Shared model-prediction stages.
    add_box(
        ax,
        (0.035, 0.64),
        0.235,
        0.17,
        "pop-cosmos catalogue",
        r"Galaxy parameters $\theta$ and redshift $z$",
        COLORS["model"],
        COLORS["model_edge"],
    )
    add_box(
        ax,
        (0.365, 0.64),
        0.25,
        0.17,
        "FSPS model prediction",
        r"Rest-frame attenuated SED and total $L_{\rm IR}$",
        COLORS["model"],
        COLORS["model_edge"],
    )
    add_box(
        ax,
        (0.71, 0.64),
        0.255,
        0.17,
        "Observed-frame SPIRE fluxes",
        r"Apply redshift and luminosity distance" "\n" r"to predict $F_{250}$, $F_{350}$ and $F_{500}$",
        COLORS["transform"],
        COLORS["transform_edge"],
        title_size=10.5,
    )
    add_arrow(ax, (0.27, 0.725), (0.365, 0.725))
    add_arrow(ax, (0.615, 0.725), (0.71, 0.725))

    # Branch labels make the distinction between the two tests explicit.
    ax.text(
        0.29,
        0.52,
        "Population-level test",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["test_edge"],
        transform=ax.transAxes,
    )
    ax.text(
        0.735,
        0.52,
        "Object-level test",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["test_edge"],
        transform=ax.transAxes,
    )

    add_box(
        ax,
        (0.055, 0.285),
        0.255,
        0.16,
        "Bin the simulated population",
        "Bin by flux and divide by catalogue area\n"
        r"to obtain $S^{2.5}\,dN/dS$ in each band",
        COLORS["test"],
        COLORS["test_edge"],
        title_size=10.6,
    )
    add_box(
        ax,
        (0.37, 0.285),
        0.235,
        0.16,
        "Published count comparison",
        "Corrected SPIRE counts from\nindependent survey families",
        COLORS["observation"],
        COLORS["observation_edge"],
        title_size=10.6,
    )
    add_box(
        ax,
        (0.665, 0.285),
        0.275,
        0.16,
        "Matched-object residuals",
        r"For the same COSMOS source:" "\n" r"$\log_{10}(F_{\rm model}/F_{\rm observed})$",
        COLORS["test"],
        COLORS["test_edge"],
        title_size=10.6,
    )

    # The shared predicted-flux stage branches into the two validations.
    add_arrow(
        ax,
        (0.815, 0.64),
        (0.31, 0.445),
        connectionstyle="arc3,rad=0.10",
    )
    add_arrow(
        ax,
        (0.855, 0.64),
        (0.805, 0.445),
        connectionstyle="arc3,rad=-0.04",
    )
    add_arrow(ax, (0.31, 0.365), (0.37, 0.365), color=COLORS["test_edge"])

    # Observational inputs enter from below, visually separate from model processing.
    add_box(
        ax,
        (0.37, 0.065),
        0.235,
        0.115,
        "Observed input",
        "Valiante, Oliver and Pearson counts",
        COLORS["observation"],
        COLORS["observation_edge"],
        title_size=9.8,
        body_size=8.2,
        linewidth=1.2,
    )
    add_box(
        ax,
        (0.665, 0.065),
        0.275,
        0.115,
        "Observed input",
        "Wang and Jin deblended COSMOS fluxes",
        COLORS["observation"],
        COLORS["observation_edge"],
        title_size=9.8,
        body_size=8.2,
        linewidth=1.2,
    )
    add_arrow(ax, (0.4875, 0.18), (0.4875, 0.285), color=COLORS["observation_edge"])
    add_arrow(ax, (0.8025, 0.18), (0.8025, 0.285), color=COLORS["observation_edge"])

    ax.text(
        0.182,
        0.245,
        "Tests the abundance of the\nmodel population in flux bins",
        ha="center",
        va="top",
        fontsize=8.2,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    ax.text(
        0.802,
        0.245,
        "Tests prediction accuracy\nfor individual galaxies",
        ha="center",
        va="top",
        fontsize=8.2,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    return fig


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_STEM}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_STEM}.png")
    print(f"Wrote {OUTPUT_STEM}.pdf")
    print(f"Wrote {OUTPUT_STEM}.svg")


if __name__ == "__main__":
    main()
