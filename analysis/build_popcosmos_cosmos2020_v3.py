from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main():
    nb_dir = Path(__file__).resolve().parent
    src = nb_dir / "popcosmos_cosmos2020_comparison_v2.ipynb"
    dst = nb_dir / "popcosmos_cosmos2020_comparison_v3.ipynb"

    nb = nbf.read(src.open("r", encoding="utf-8"), as_version=4)

    new_cells = [
        md(
            """## Step 13: Direct SFR comparison across catalogs

Here I switch from the earlier starburst-style summary to a simpler direct comparison:

- pop-cosmos SFR vs COSMOS2020 LePhare SFR
- pop-cosmos SFR vs COSMOS2020 EAZY SFR

This is closer to the supervisor suggestion: just cross-match and compare the SFR estimates directly.

Important caveat:
- the SFR methods are not identical, so offsets do not automatically mean one catalog is "wrong"
- but large, systematic offsets are still useful to see"""
        ),
        code(
            """def paired_sfr_summary(df, left_col, right_col, z_col, m_col, label, subset_mask):
    m = subset_mask.fillna(False)
    d = (df.loc[m, left_col] - df.loc[m, right_col]).astype(float)
    d = d[np.isfinite(d)]

    if len(d) == 0:
        return {
            "pair": label,
            "subset": "empty",
            "N": 0,
            "median_dlogSFR": np.nan,
            "median_ratio": np.nan,
            "sigma_mad_dlogSFR": np.nan,
            "p16_dlogSFR": np.nan,
            "p84_dlogSFR": np.nan,
            "median_z": np.nan,
            "median_log10M": np.nan,
        }

    return {
        "pair": label,
        "N": int(m.sum()),
        "median_dlogSFR": float(np.nanmedian(d)),
        "median_ratio": float(10 ** np.nanmedian(d)),
        "sigma_mad_dlogSFR": float(1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d)))),
        "p16_dlogSFR": float(np.nanpercentile(d, 16)),
        "p84_dlogSFR": float(np.nanpercentile(d, 84)),
        "median_z": float(np.nanmedian(df.loc[m, z_col])),
        "median_log10M": float(np.nanmedian(df.loc[m, m_col])),
    }


matched_sfr = matched.copy()
matched_sfr["dlogSFR_lp_minus_pop"] = matched_sfr["lp_SFR_med"] - matched_sfr["log10SFR_pop"]
matched_sfr["dlogSFR_ez_minus_pop"] = matched_sfr["ez_sfr"] - matched_sfr["log10SFR_pop"]

base_pop = (
    np.isfinite(matched_sfr["log10M_pop"])
    & np.isfinite(matched_sfr["log10SFR_pop"])
    & np.isfinite(matched_sfr["log10sSFR_pop"])
    & np.isfinite(matched_sfr["z_pop"])
    & (matched_sfr["log10M_pop"] >= 8.5)
    & (matched_sfr["log10M_pop"] <= 11.5)
    & (matched_sfr["z_pop"] >= 0.0)
    & (matched_sfr["z_pop"] < 4.0)
    & (matched_sfr["log10sSFR_pop"] > -11.0)
)

subsets = {
    "all_sf_like": base_pop,
    "binA_1<=z<2_9<=logM<=11.5": base_pop & (matched_sfr["z_pop"] >= 1.0) & (matched_sfr["z_pop"] < 2.0) & (matched_sfr["log10M_pop"] >= 9.0),
    "binB_1.5<=z<2.5_logM>=10": base_pop & (matched_sfr["z_pop"] >= 1.5) & (matched_sfr["z_pop"] < 2.5) & (matched_sfr["log10M_pop"] >= 10.0),
    "narrow_1.0<=z<1.5": base_pop & (matched_sfr["z_pop"] >= 1.0) & (matched_sfr["z_pop"] < 1.5) & (matched_sfr["log10M_pop"] >= 9.0),
    "narrow_1.5<=z<2.0": base_pop & (matched_sfr["z_pop"] >= 1.5) & (matched_sfr["z_pop"] < 2.0) & (matched_sfr["log10M_pop"] >= 9.0),
}

rows = []
for subset_name, subset_mask in subsets.items():
    row_lp = paired_sfr_summary(
        matched_sfr,
        "lp_SFR_med",
        "log10SFR_pop",
        "z_pop",
        "log10M_pop",
        "lp_minus_pop",
        subset_mask & np.isfinite(matched_sfr["lp_SFR_med"]),
    )
    row_lp["subset"] = subset_name
    rows.append(row_lp)

    row_ez = paired_sfr_summary(
        matched_sfr,
        "ez_sfr",
        "log10SFR_pop",
        "z_pop",
        "log10M_pop",
        "ez_minus_pop",
        subset_mask & np.isfinite(matched_sfr["ez_sfr"]),
    )
    row_ez["subset"] = subset_name
    rows.append(row_ez)

sfr_offset_summary = pd.DataFrame(rows)
sfr_offset_summary"""
        ),
        md(
            """What this table is saying:

- `median_dlogSFR` is the typical SFR offset between methods
- positive means the first method in `pair` gives a higher SFR
- `median_ratio` is the same thing but as a factor instead of dex
- `sigma_mad_dlogSFR` is the scatter in that offset

This is the cleanest "same galaxies, different SFR estimate" comparison in the current local data."""
        ),
        code(
            """sfr_offset_csv = OUT_DIR / "popcosmos_cosmos2020_sfr_offsets_summary.csv"
sfr_offset_summary.to_csv(sfr_offset_csv, index=False)
print("Saved:", sfr_offset_csv)

binA_plot = subsets["binA_1<=z<2_9<=logM<=11.5"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, ycol, title in [
    (axes[0], "lp_SFR_med", "LePhare vs pop-cosmos SFR"),
    (axes[1], "ez_sfr", "EAZY vs pop-cosmos SFR"),
]:
    d = matched_sfr.loc[binA_plot & np.isfinite(matched_sfr[ycol]), ["log10SFR_pop", ycol]].copy()
    hb = ax.hexbin(d["log10SFR_pop"], d[ycol], gridsize=65, mincnt=1, cmap="viridis")
    lo = np.nanmin([d["log10SFR_pop"].min(), d[ycol].min()])
    hi = np.nanmax([d["log10SFR_pop"].max(), d[ycol].max()])
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1 line")
    ax.set_xlabel(r"$\\log_{10}(\\mathrm{SFR}_{pop})$")
    ax.set_ylabel(r"$\\log_{10}(\\mathrm{SFR}_{other})$")
    ax.set_title(title + "\\nBin A")
    ax.legend(loc="upper left", fontsize=8)

cbar = fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.92)
cbar.set_label("counts per hexbin")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4.5))
for col, label, color in [
    ("dlogSFR_lp_minus_pop", "LePhare - pop", "#2ca02c"),
    ("dlogSFR_ez_minus_pop", "EAZY - pop", "#ff7f0e"),
]:
    arr = matched_sfr.loc[binA_plot & np.isfinite(matched_sfr[col]), col]
    plt.hist(arr, bins=90, density=True, histtype="step", lw=1.8, label=label, color=color)
plt.axvline(0.0, color="k", ls="--", lw=1)
plt.xlabel(r"$\\Delta \\log_{10}(\\mathrm{SFR})$")
plt.ylabel("density")
plt.title("Bin A direct SFR offsets")
plt.legend()
plt.tight_layout()
plt.show()"""
        ),
        md(
            """Quick interpretation of the plots:

- If points lie near the red `1:1` line, the two SFR estimates broadly agree
- if a cloud sits above or below the line, that method is systematically higher or lower
- the histogram shows the same thing in a simpler 1D way

This is useful because it removes the extra Speagle step and just compares the estimated SFR values directly."""
        ),
        md(
            """## Step 14: Wang direct comparison note + closest local test

The local Wang `master.dat` file does **not** provide a direct SFR column.

So for Wang, I cannot do a true SFR-vs-SFR table yet from the currently downloaded file.

What I can do right now is the next-best local check:

- cross-match Wang and pop-cosmos by ID
- compare pop-cosmos SFR against Wang long-wavelength fluxes / detection strength

This tests whether galaxies that pop-cosmos says are more star-forming also tend to look brighter in the far-IR/sub-mm."""
        ),
        code(
            """import warnings
from astropy.table import Table

wang_master = ROOT / "catalog data/wang/master.dat.gz"
wang_readme = ROOT / "catalog data/wang/ReadMe.txt"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    wang_tab = Table.read(wang_master, format="ascii.cds", readme=wang_readme)

wang_df = wang_tab.to_pandas()
wang_keep = ["ID", "F24", "s_F24", "F250", "s_F250", "F350", "s_F350", "F500", "s_F500", "F850", "s_F850"]
wang_df = wang_df[[c for c in wang_keep if c in wang_df.columns]].copy()
wang_df = wang_df[wang_df["ID"] > 0].copy()

for c in wang_df.columns:
    wang_df[c] = pd.to_numeric(wang_df[c], errors="coerce")

for lam in [24, 250, 350, 500, 850]:
    fcol = f"F{lam}"
    ecol = f"s_F{lam}"
    snr_col = f"SNR{lam}"
    if fcol in wang_df and ecol in wang_df:
        wang_df[snr_col] = wang_df[fcol] / wang_df[ecol].replace(0, np.nan)

wang_matched = pop_df.merge(wang_df, on="ID", how="inner")
wang_matched["long_detect"] = (
    (wang_matched.get("SNR250", np.nan) >= 3)
    | (wang_matched.get("SNR350", np.nan) >= 3)
    | (wang_matched.get("SNR500", np.nan) >= 3)
    | (wang_matched.get("SNR850", np.nan) >= 3)
)

def wang_flux_sfr_summary(df, band, base_mask):
    snr_col = f"SNR{band}"
    fcol = f"F{band}"
    m = base_mask & np.isfinite(df[fcol]) & np.isfinite(df[snr_col]) & (df[snr_col] >= 3) & (df[fcol] > 0)
    if m.sum() == 0:
        return {
            "band_um": band,
            "N_detect": 0,
            "median_log10SFR_pop": np.nan,
            "median_log10M_pop": np.nan,
            "median_z_pop": np.nan,
            "spearman_rho_logflux_vs_logSFR": np.nan,
        }
    tmp = df.loc[m, [fcol, "log10SFR_pop", "log10M_pop", "z_pop"]].copy()
    tmp[f"log10F{band}"] = np.log10(tmp[fcol])
    rho = tmp[f"log10F{band}"].corr(tmp["log10SFR_pop"], method="spearman")
    return {
        "band_um": band,
        "N_detect": int(m.sum()),
        "median_log10SFR_pop": float(np.nanmedian(tmp["log10SFR_pop"])),
        "median_log10M_pop": float(np.nanmedian(tmp["log10M_pop"])),
        "median_z_pop": float(np.nanmedian(tmp["z_pop"])),
        "spearman_rho_logflux_vs_logSFR": float(rho),
    }

wang_base = (
    np.isfinite(wang_matched["log10M_pop"])
    & np.isfinite(wang_matched["log10SFR_pop"])
    & np.isfinite(wang_matched["log10sSFR_pop"])
    & np.isfinite(wang_matched["z_pop"])
    & (wang_matched["log10M_pop"] >= 8.5)
    & (wang_matched["log10M_pop"] <= 11.5)
    & (wang_matched["z_pop"] >= 0.0)
    & (wang_matched["z_pop"] < 4.0)
    & (wang_matched["log10sSFR_pop"] > -11.0)
)

wang_binA = wang_base & (wang_matched["z_pop"] >= 1.0) & (wang_matched["z_pop"] < 2.0) & (wang_matched["log10M_pop"] >= 9.0)

wang_flux_summary = pd.DataFrame(
    [wang_flux_sfr_summary(wang_matched, band, wang_binA) for band in [24, 250, 350, 500, 850]]
)
wang_flux_summary"""
        ),
        md(
            """What this Wang table means:

- `N_detect` tells me how many Bin A galaxies are securely detected in that band
- `median_log10SFR_pop` tells me the typical pop-cosmos SFR for those detected galaxies
- `spearman_rho_logflux_vs_logSFR` tells me whether brighter long-wavelength flux tends to line up with higher pop-cosmos SFR

So this is not yet a Wang-SFR table, but it is still a useful consistency check with the local data I already have."""
        ),
        code(
            """wang_flux_csv = OUT_DIR / "popcosmos_wang_flux_vs_sfr_summary.csv"
wang_flux_summary.to_csv(wang_flux_csv, index=False)
print("Saved:", wang_flux_csv)

plot_mask_250 = wang_binA & np.isfinite(wang_matched["F250"]) & np.isfinite(wang_matched["SNR250"]) & (wang_matched["SNR250"] >= 3) & (wang_matched["F250"] > 0)
plot_mask_850 = wang_binA & np.isfinite(wang_matched["F850"]) & np.isfinite(wang_matched["SNR850"]) & (wang_matched["SNR850"] >= 3) & (wang_matched["F850"] > 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mask, col, title in [
    (axes[0], plot_mask_250, "F250", "Wang 250um vs pop-cosmos SFR"),
    (axes[1], plot_mask_850, "F850", "Wang 850um vs pop-cosmos SFR"),
]:
    d = wang_matched.loc[mask, ["log10SFR_pop", col]].copy()
    d[f"log10_{col}"] = np.log10(d[col])
    hb = ax.hexbin(d["log10SFR_pop"], d[f"log10_{col}"], gridsize=55, mincnt=1, cmap="magma")
    ax.set_xlabel(r"$\\log_{10}(\\mathrm{SFR}_{pop})$")
    ax.set_ylabel(r"$\\log_{10}(F_{\\lambda}/\\mathrm{mJy})$")
    ax.set_title(title + "\\nBin A, SNR>=3")

cbar = fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.92)
cbar.set_label("counts per hexbin")
plt.tight_layout()
plt.show()

wang_detect_compare = pd.DataFrame(
    {
        "group": ["long_detected", "not_long_detected"],
        "N": [
            int((wang_binA & wang_matched["long_detect"]).sum()),
            int((wang_binA & ~wang_matched["long_detect"]).sum()),
        ],
        "median_log10SFR_pop": [
            float(np.nanmedian(wang_matched.loc[wang_binA & wang_matched["long_detect"], "log10SFR_pop"])),
            float(np.nanmedian(wang_matched.loc[wang_binA & ~wang_matched["long_detect"], "log10SFR_pop"])),
        ],
        "median_log10M_pop": [
            float(np.nanmedian(wang_matched.loc[wang_binA & wang_matched["long_detect"], "log10M_pop"])),
            float(np.nanmedian(wang_matched.loc[wang_binA & ~wang_matched["long_detect"], "log10M_pop"])),
        ],
    }
)
wang_detect_compare"""
        ),
        md(
            """Quick reading:

- if long-detected galaxies have higher median pop-cosmos SFR and mass, that is at least qualitatively sensible
- if the flux-vs-SFR trend is weak, it means I should be careful about over-interpreting the Wang split without deriving an IR-based SFR"""
        ),
        md(
            """## Step 15: Main-sequence plane in narrower redshift bins

The earlier MS-plane figure used a broad `1 <= z < 2` bin.

Here I split that into narrower bins:
- `1.0 <= z < 1.5`
- `1.5 <= z < 2.0`

This reduces the amount of redshift evolution mixed into each panel."""
        ),
        code(
            """narrow_bins = [
    ("1.0 <= z < 1.5", 1.0, 1.5),
    ("1.5 <= z < 2.0", 1.5, 2.0),
]

def plot_ms_row(ax, work_df, title, zlo, zhi):
    base = sf_base_mask(work_df)
    m = base & (work_df["z"] >= zlo) & (work_df["z"] < zhi) & (work_df["log10M"] >= 9.0) & (work_df["log10M"] <= 11.5)
    d = work_df.loc[m].copy()
    if len(d) == 0:
        ax.set_title(title + "\\n(no data)")
        return None
    hb = ax.hexbin(d["log10M"], d["log10SFR"], gridsize=50, mincnt=1, cmap="viridis")
    z_med = np.nanmedian(d["z"])
    x = np.linspace(9.0, 11.5, 120)
    ax.plot(x, speagle_log10sfr(x, np.full_like(x, z_med)), "r-", lw=2)
    ax.set_xlabel(r"$\\log_{10}(M_\\star/M_\\odot)$")
    ax.set_ylabel(r"$\\log_{10}(\\mathrm{SFR}/M_\\odot\\,\\mathrm{yr}^{-1})$")
    ax.set_title(title + f"\\n{zlo} <= z < {zhi}")
    return hb

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
for row, (label, zlo, zhi) in enumerate(narrow_bins):
    hb0 = plot_ms_row(axes[row, 0], pop_work, "pop-cosmos", zlo, zhi)
    hb1 = plot_ms_row(axes[row, 1], lp_work, "COSMOS2020 LePhare", zlo, zhi)
    hb2 = plot_ms_row(axes[row, 2], ez_work, "COSMOS2020 EAZY", zlo, zhi)

cbar = fig.colorbar(hb2 if hb2 is not None else hb1 if hb1 is not None else hb0, ax=axes.ravel().tolist(), shrink=0.92)
cbar.set_label("counts per hexbin")
plt.suptitle("Main-sequence plane comparison in narrower z bins", y=1.01)
plt.tight_layout()
plt.show()"""
        ),
        md(
            """What I want to look for here:

- whether the broad-bin differences were partly caused by mixing too much redshift evolution together
- whether the relative ordering of pop-cosmos / LePhare / EAZY stays the same when I narrow the bin

If the same pattern survives in both narrower bins, that is stronger evidence the difference is method-related, not just bin-width related."""
        ),
        md(
            """## Step 16: Spectroscopic redshift check (local files only)

The next question is whether I can benchmark pop-cosmos, LePhare, and EAZY against spectroscopic redshifts **using the files currently downloaded**.

Before doing a full comparison, I first check whether a local spec-z column actually exists in the files I have."""
        ),
        code(
            """farmer_spec_like_cols = [c for c in farmer_df.columns if "spec" in c.lower()]

with h5py.File(pop_h5, "r") as f:
    pop_meta_keys = list(f["metadata"].keys())

pop_spec_like_keys = [k for k in pop_meta_keys if "spec" in k.lower()]

spec_check = pd.DataFrame(
    {
        "source": ["COSMOS2020 farmer.dat.gz", "pop-cosmos mcmc_summaries.h5 metadata"],
        "spec_like_fields_found": [str(farmer_spec_like_cols), str(pop_spec_like_keys)],
        "n_fields_found": [len(farmer_spec_like_cols), len(pop_spec_like_keys)],
    }
)
spec_check"""
        ),
        md(
            """Interpretation:

- if this table is empty / zero-field for both files, then a direct local spec-z comparison is **not yet runnable**
- in that case, the honest next step is to download the spectroscopic compilation / matching file and then compare

When I do have spec-z, the standard comparison would be:

$$
\\Delta z = \\frac{z_{est} - z_{spec}}{1 + z_{spec}}
$$

and then report:
- median bias
- scatter (`sigma_MAD`)
- outlier fraction, for example `|Delta z| > 0.15`"""
        ),
        md(
            """## Step 17: Mass/luminosity function note (`1 / V_max`)

This is a standard next step, but I am treating it as methodology only for now.

Why I am not doing the full thing yet:

- a proper `1 / V_max` mass or luminosity function needs a clear survey selection limit
- it also needs a consistent way to compute `z_max` for each object
- and ideally a completeness cut or mask treatment

The current matched notebook is good for direct estimator comparisons, but not yet clean enough for a publishable function measurement.

If I do this next, the simple plan is:

1. choose one parent selection band and magnitude limit
2. compute `z_max` for each galaxy under that selection
3. sum `1 / V_max` in bins of mass or luminosity
4. compare the resulting curves across pop-cosmos / LePhare / EAZY in the same redshift bin"""
        ),
        md(
            """## Extra quick summary from the new sections

- Direct SFR comparison is the cleanest immediate check across pop-cosmos and COSMOS2020, because all three already have SFR columns locally.
- Wang is still useful, but with the currently downloaded file it is a flux-vs-SFR consistency test, not a direct SFR-vs-SFR comparison.
- Narrower redshift bins are a good idea because they reduce redshift evolution inside a single MS panel.
- Direct spec-z benchmarking is not possible from the currently downloaded Farmer and pop-cosmos summary files alone if no spec-z field is present locally."""
        ),
    ]

    nb.cells.extend(new_cells)

    with dst.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
