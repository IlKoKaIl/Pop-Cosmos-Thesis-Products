# Data manifest

The repository contains the small published-count tables and ALESS template needed to
understand the analysis. The full catalogues are not committed because they are large,
externally maintained, or were supplied directly for the thesis.

## Included

- `data/published_counts/`: transcribed and standardised SPIRE count tables from
  Clements et al. (2010), Oliver et al. (2010), Glenn et al. (2010), Valiante et al.
  (2016), Pearson et al. (2025), and Varnish et al. (2025).
- `data/templates/aless_average_seds.dat.txt`: public ALESS average SED template used
  for the empirical-template tests.
- `results/summary/`: compact tables behind the headline thesis results.
- `results/intermediate/`: small intermediate CSV outputs retained for traceability.

## Required for a full rerun

The original scripts expect local copies of the following products. Their exact paths
vary between scripts because the project developed through several exploratory stages.

| Product | Purpose |
|---|---|
| `fsps_map_median_full.h5` | Rest-frame FSPS SEDs and `L_IR` supplied for the project |
| `mcmc_summaries.h5` | Pop-cosmos posterior-median parameter catalogue |
| Wang `master.dat.gz` and `ReadMe` | Deblended COSMOS FIR photometry |
| Jin et al. FIR/mm FITS catalogue | Independent super-deblended COSMOS comparison |
| COSMOS2020 Farmer catalogue | IDs, photometry, coordinates, and selection flags |
| MIR photometry HDF5 products | IRAC control comparisons |

Place local copies under `data/raw/` and update the path constants near the top of the
relevant script. These files are intentionally covered by `.gitignore`.

Published count tables should be preferred over recomputing counts from source
catalogues because the publications already apply completeness, reliability, and flux
bias corrections.

