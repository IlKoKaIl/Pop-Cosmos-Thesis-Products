# Pop-Cosmos FIR Validation Thesis Products

Code, observational count tables, result summaries, and figures produced for the thesis
*Extending Pop-cosmos to the Far-Infrared: A Number-Count Validation*.

The project tests predictions from the pop-cosmos generative galaxy-population model at
250, 350, and 500 micrometres. The main comparison uses published Herschel/SPIRE
differential number counts, supported by per-object comparisons with the Wang and Jin
deblended COSMOS catalogues.

## Main findings

- The baseline pop-cosmos/FSPS prediction produces too many sources in the SPIRE count
  bins. Over 30--100 mJy, the median excess is about 1.7 times at 250 and 350 micrometres
  and 3.6 times at 500 micrometres.
- Low-AGN model galaxies have an FIR peak that is colder and less luminosity-dependent
  than the observational relation used for comparison.
- Warmer MBB, ALESS/FSPS hybrid, and Casey-inspired templates improve the population
  counts at fixed infrared luminosity, but the counts do not select a unique dust model.
- Large per-object model scatter is a separate issue. It can inflate bright counts even
  while the median matched galaxy is predicted too faintly.

## Repository layout

| Path | Contents |
|---|---|
| `analysis/` | Python scripts used throughout the project |
| `data/published_counts/` | Published SPIRE count tables and source-selection records |
| `data/templates/` | Public empirical SED template used in the tests |
| `results/summary/` | Final compact numerical results used in the thesis |
| `results/intermediate/` | Smaller intermediate outputs retained for auditability |
| `figures/` | Final and supporting plots |
| `docs/PROJECT_RECORD.md` | Consolidated record of methods, findings, and caveats |
| `docs/FIR_VALIDATION_REPORT.md` | Focused technical validation report |
| `paper/` | LaTeX report source and bibliography |

The long weekly meeting notes, superseded drafts, downloaded papers, LaTeX build files,
and multi-gigabyte catalogues are deliberately omitted.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The compact outputs can be inspected without the original catalogues. A full rerun
requires the external inputs described in [DATA.md](DATA.md). The scripts retain clear
path constants near their tops so those local files can be configured without committing
them.

## Suggested starting points

- `analysis/popcosmos_full_sed_250_counts.py` builds observed FIR fluxes from the stored
  pop-cosmos/FSPS SEDs.
- `analysis/compile_external_spire_differential_counts.py` standardises published counts.
- `analysis/popcosmos_differential_count_evaluator.py` compares model and observed count
  curves.
- `analysis/popcosmos_wang_jin_fsps_compare.py` performs the matched-catalogue check.
- `analysis/run_fir_evaluator_pipeline.py` records the intended evaluator order.

The report describes statistical limitations explicitly. In particular, overlapping
count products are not treated as independent evidence, and the chi-square values are
used as comparative diagnostics rather than formal likelihoods.

