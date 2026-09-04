# Analysis scripts

These scripts generate the thesis diagnostics, count comparisons, SED tests, and
summary figures. They were developed during an exploratory research project, so
several scripts contain input-path constants near the top that must be pointed at
the local catalogues listed in [`../DATA.md`](../DATA.md).

Useful entry points include:

- `popcosmos_aless_sed_flux_check.py` for the first ALESS and FSPS comparison.
- `popcosmos_restframe_hybrid_sed.py` for rest-frame hybrid templates.
- `popcosmos_full_sed_counts_compare.py` for SPIRE number-count predictions.
- `popcosmos_external_differential_counts.py` for published count compilation.
- `popcosmos_differential_count_evaluator.py` for count-based model scoring.
- `popcosmos_wang_jin_comparison.py` for matched-catalogue diagnostics.

The scripts are preserved as research products rather than presented as a single
turnkey pipeline. Generated compact tables and final figures are included in the
repository so the main results can be inspected without the raw catalogues.
