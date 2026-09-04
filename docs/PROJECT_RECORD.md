# Project record: extending pop-cosmos into the far-infrared

The study of galaxy evolution- how galaxies change over time, their relationship to each
other and other phenomena such as active galactic nuclei, and how various observed scaling
relations emerge- is central to modern astrophysics. This work depends on studying large
samples of galaxies at a variety of wavelengths using a range of statistical, physical and
other techniques. Recently deployed telescopes such as Euclid and the Rubin-LSST will
produce catalogues of billions of galaxies, for which traditional parametric models will not
work. To address this problem we have developed the pop-cosmos framework to model the
galaxy population using a combination of physical models, machine learning methods and
traditional statistics. This Masters project will validate and extend the pop-cosmos galaxy
population synthesis model by comparing its predictions to observational data at wavelengths
beyond its current optical/near-IR training range (0.3-5 Âµm). Students will leverage existing
multi-wavelength survey data from X-COSMOS, Herschel, and radio observations to test
model consistency across different emission mechanisms. Key tasks include comparing pop
cosmos star formation rate predictions with direct FIR/submm measurements, validating AGN
classifications using X-ray and radio flux cross-checks, and testing the radio-FIR relation
for non-AGN sources. The project will also involve extending the pop-cosmos framework
by running additional Flexible Stellar Population Synthesis (FSPS) calculations to generate
predictions at 24 microns and other far-infrared bands for direct comparison with MIPS,
Herschel, and ALMA data, and providing predictions for future projects such as the NASA
PRIMA mission. This work will provide crucial independent validation of the modelâ€˜s physical
assumptions while exploring its potential for multi-wavelength applications in galaxy evolution
studies.

**What this file is.** One consolidated read of everything I've done, Feb to Aug 2026, written
as a story so the thesis has a spine. It replaces the three separate notes files
(`THESIS_NARRATIVE_NOTES.md`, `thesis_narrative_notes_raw.md`, `part2_narrative_raw.md`) —
those were a mess of my own making and can be deleted.

**Sources.** `supervisor_meeting_prep.md` (6,390 lines, Feb–Aug) and `Thesis_part2.md`
(1,468 lines, Aug, including seven appended extensions). Line numbers below point back to the
originals so I can find specifics.

**Every figure named here has been checked to exist on disk (26 of 26).**

The one-line version:

> pop-cosmos was validated where it was trained, then pushed into the far-infrared where it
> was not, and the extension found two specific, physical failures that the training data
> could not have revealed.

The early chapters are what make that sentence work. Don't cut them.

---

# Part I — The narrative arc

## Act 1 (Feb–Mar): does the model reproduce known relations?

*Log lines 28–120.*

**Why.** Before pushing into new wavelengths, check the model against something established.
The Speagle+2014 main sequence was the obvious ruler.

**Found.** pop-cosmos SFRs sit systematically low: median ΔMS ≈ −0.53 dex, worst at low
redshift (−0.68 at z<0.5), converging by z~3 (−0.19). Confirmed independently against LePhare
(+0.22 to +0.36 dex) and EAZY (+0.08 to +0.17 dex) on the same matched galaxies.

**Role in the thesis.** Establishes that a systematic offset exists and isn't a binning
artifact. Sets up the honesty theme — I found a discrepancy and chased its cause rather than
reporting it as a result.

**Dead end worth one sentence.** Starburst fractions (ΔMS ≥ 0.6 dex) came out implausibly low
(~0.3% vs ~2% in Rodighiero+2011). I diagnosed this as a definition/selection mismatch rather
than physics and paused. My own words, worth keeping: *"We likely got very low starburst
numbers because this was a much stricter and different comparison than before, not because
starbursts disappear physically."*

**Figure.** SFR–M* hexbin with the Speagle line. Appendix at most.

## Act 2 (Mar–Apr): does the model work where it was trained?

*Log lines 1007–1499.*

**Why.** Dave asked whether model fluxes could be loaded and checked. This is the control
experiment for everything later.

**Found.** IRAC Ch1: N=423,272, median residual +0.020 mag, MAD 0.084. Ch2: N=414,272,
−0.012 mag, MAD 0.041. Both track the observations well, Ch2 tighter. Ch3/Ch4 noisier with
lower coverage (55%, 22%). MIPS24 unavailable.

**Role in the thesis — load-bearing.** Without this, "the far-IR is wrong" could just be "the
model is wrong". With it, I can say the model is sound where constrained and fails specifically
where it isn't. **Keep in main text, compressed to a paragraph and possibly one panel.** It's
what makes the later result diagnostic rather than merely negative.

My framing was already right: *"Ch1 and Ch2 are close enough to the fitted regime that they let
me check whether the modelling pipeline is behaving sensibly; Ch3 and Ch4 start to push further
out in wavelength, so they are more like a real extension test."*

**Figures.** `popcosmos_irac_redshift_histograms.png` (appendix). Boris's
`fig1_speculator_vs_stored.png`, `fig3_coverage_map.png`, `fig6_example_seds.png` are internal
consistency checks — appendix, and credit Boris.

## Act 3 (Apr–May): the first far-IR attempt, and why it failed

*Log lines 1499–2179.*

**Why.** Bridge from optical fitting to far-IR. First attempt compared model `L_IR` against
Kennicutt and against single-band Wang fluxes.

**Found.** Model `L_IR` correlates strongly with SFR (rho 0.82–0.95), so the internal physics
is coherent. But `L_IR` sits 0.3–0.6 dex above the Kennicutt line, peaking ~0.44 dex at
z~1.75, and single-band flux correlations were weak (rho 0.16–0.23).

**Role in the thesis — the most instructive dead end, deserves a short subsection.** The weak
correlations looked like model failure but weren't: a single observed band samples different
rest-frame wavelengths across a redshift range. My realisation — *"This makes me less worried
that the low direct Spearman coefficients automatically mean pop-cosmos is failing"* — is the
kind of reasoning an examiner wants to see. It also motivates why I needed a statistic that
doesn't require rest-frame interpretation.

**Figures.** `popcosmos_lir_vs_sfr.png`, `popcosmos_lir_offset_by_redshift.png`. One in main
text as the "first attempt" figure, or both in an appendix.

## Act 4 (late May): the turning point

*Log lines 2179–2497.*

Two supervisor interventions changed the project. Both should be visible in the report.

1. **Dave, May 25:** *"the validation does not have to be only on physical quantities like
   L_IR or SFR. It can also be on observed quantities like number counts."* This decoupled the
   thesis from dust-template and IMF assumptions.
2. **Dave, later:** differential counts rather than integral counts, because integral bins
   re-use sources and therefore have correlated errors.

My restatement is the cleanest version and should be paraphrased in the Methods motivation:
*"Number counts feel like the best route because they test observed fluxes directly, without
having to turn real 250/350/500 µm fluxes into more model-dependent physical quantities like
L_IR or SFR."*

**Also here:** scope narrowed from FIR + radio + X-ray to FIR alone. State it plainly as a
scoping decision with the reason (each regime needed its own systematics work), not as an
omission.

**Cut entirely:** the agentic-pipeline automation sketch. Already ruled out as scope creep.
One clause in Further Work at most.

## Act 5 (June): finding the actual physics

*Log lines 3664–4211.*

**Why.** With counts as the metric, compare properly — first with the ALESS template as a
stand-in, then with the real FSPS SEDs.

**Found, and this is the core of the thesis:**

- pop-cosmos far-IR SEDs peak at ~135–160 µm; ALESS submillimetre galaxies peak at ~80–100 µm.
  The model's dust is colder.
- The bright-count mismatch **grows with wavelength** (250 → 350 → 500 → 850 µm). That ladder
  is the signature of excess cold dust and the single most persuasive piece of evidence in the
  log.
- The top-5 SFR objects have log `L_IR` 12.8–14.1, so the cold peak isn't a low-luminosity
  artifact.

**The key quote in the entire log** (Boris, recorded by me):

> "pop-cosmos gets these SEDs from FSPS with energy balance. Energy balance fixes total L_IR.
> But the dust temperature / far-IR shape is basically not fitted by COSMOS data. COSMOS only
> really constrains out to IRAC, not the far-IR."

That's the central mechanism and belongs early in the Introduction, in my own words.

My own framing, which the Results section should build toward: *"the model may have roughly
the right total dust luminosity, but puts it at the wrong wavelengths."* Plus: *"FIR validation
is finding something optical/NIR validation could not see."*

**Figures (strong).**

- `popcosmos_full_sed_multiband_counts.png` — the wavelength ladder. MAIN.
- `popcosmos_full_sed_median_sfr_seds.png` — median SED by SFR bin; shows the cold feature is
  population-wide. MAIN.
- `popcosmos_full_sed_top5_shape_normalized.png` — shape-only FSPS vs ALESS. MAIN or appendix.
- `popcosmos_full_sed_agn_parameter_median_seds.png` — high-AGN tail peaks ~33 µm vs ~135 µm.
  Appendix, and pre-empts an obvious viva question.

**Cut:** the 30–100 µm narrow spikes. Correctly identified as nebular fine-structure lines,
cosmetic for broadband work. One clause at most.

**Flag as unresolved-then-resolved:** the per-object vs count contradiction first appears here
(line 4406). Present it as the puzzle it was; the resolution is the scatter finding from August.

## Act 6 (July–Aug): making it quantitative

*Log lines 4682–6170.*

**Why.** Show the diagnosis is right by fixing the SED shape at fixed `L_IR` and scoring
against published counts.

**Found.**

- ALESS/FSPS hybrids move the bright 350/500 µm counts the right way. Rest-frame
  implementation verified: FSPS integral / stored `L_IR` = 0.999994.
- Modified blackbody grid (20–50 K, beta=1.8): best at **35 K**. Physically interpretable, and
  both resolved and P(D) count types prefer ~35 K independently.
- Leave-one-source-out: 25% ALESS chosen in every held-out run.
- The **area correction was critical** — 1.278 deg² (Farmer `FLAG_COMBINED`) not 2.0 deg², a
  factor 1.56. With the wrong area the ranking reverses. My note is the honest version:
  *"I need to settle the correct area before claiming template X is better."*
- Clean-independent source set (Valiante wide-area bright, Oliver mid-flux, Pearson XID deep)
  still prefers a 25–50% ALESS blend. Reduced chi-sq: 25% ALESS ~3.34 vs FSPS ~8.43.

**Role in the thesis — this is Results proper.** The MBB grid is the better story than ALESS
mixing because a temperature is a physical parameter and a mixing fraction isn't, even though
ALESS 25% scored marginally better (3.92 vs 4.50). Say that plainly.

My conclusion should survive to the report: *"the answer seems to be somewhere between FSPS
and ALESS, not simply replacing FSPS with ALESS."*

**Figures (strong).**

- `popcosmos_mbb_temperature_grid_shapes.png` + `popcosmos_mbb_temperature_grid_counts.png` —
  the template experiment.
- `popcosmos_differential_count_area_corrected_overlay.png` — my original overlay. Superseded
  by the version with a residual panel, but the honest ancestor.
- `popcosmos_differential_count_leave_one_source_out.png` — cross-validation. MAIN.
- `external_spire_differential_counts_july21_3dex.png` — the compiled observational ruler.
  MAIN, probably in Data.
- `popcosmos_clean_independent_count_evaluator_heatmap.png` — appendix.
- `popcosmos_wang_jin_fsps_ratio_summary.png` — Wang/Jin/FSPS ratios. Appendix.

## Act 7 (Aug): the second problem, and the literature

*`Thesis_part2.md`, all seven extensions.*

August split into two strands: resolving a contradiction that had been open since June, and
finding the literature that either supports or complicates the result.

**The contradiction resolved.** Per-object comparison said the model was too *faint* (−0.16
dex); number counts said too *bright* (up to 2.7×). Same model, same galaxies, same bands. The
cause is a **selection asymmetry combined with large per-object scatter**: conditioning on
Wang-detected objects selects against model-faint ones, conditioning on model-bright objects
selects the opposite. With 0.44–0.54 dex of intrinsic scatter and a steeply falling count
distribution, far more faint galaxies scatter up into a bright bin than bright ones scatter
down. Of 422 galaxies the model predicts above 20 mJy, only 21 are genuinely bright; 401 are
promoted by scatter.

**This is a second, independent finding.** Early drafts treated it as a complication of the
cold-dust result. It isn't — it's a separate problem, and one that a dust-template change
cannot fix. Present them as two.

**Whether the scatter was the model's or Wang's deblending — closed.** Cross-matched to Jin
2018 (independent deblending, same field, 80,271 objects at 1″): the model shows the same
scatter against both catalogues (0.42–0.51 dex against either), and the two catalogues agree
with each other to 0.12–0.15 dex. Deblending accounts for **6.7% / 7.3% / 12.5% of the
variance at 250 / 350 / 500 µm** — under a tenth at the two shorter bands, but 12.5% at
500 µm, so quote it per band rather than as a single figure. Intrinsic model scatter is
0.41–0.48 dex. The scatter is predominantly the model's, most cleanly so at 250 µm.

**Jin's area confirmed from the paper** (p13): `goodArea` = UltraVISTA 1.7 deg², 191,624 prior
galaxies. My independent estimate was 1.70 deg². Last open caveat on the Wang analysis closed.

The extensions in order, with what each contributes:

| ext | topic                                        | thesis value                                         |
| --- | -------------------------------------------- | ---------------------------------------------------- |
| 1   | Gravitational lensing and the bright counts  | Closes an escape route — see Part II                |
| 2   | Supervisor feedback: AGN, SNR cuts, Negrello | The 20 µm peaks are AGN; SNR robustness             |
| 3   | Clements 1999 on flux boosting               | Citable justification for the 5σ cut                |
| 4   | Literature citing Wang 2024                  | Independent support for excluding raw Wang counts    |
| 5   | Test against the Drew & Casey relation       | **Strongest unreported result** — see Part II |
| 6   | Why f_AGN matters, questions for Boris       | Mechanism for the AGN effect                         |
| 7   | Two corrections from Dave                    | Eddington vs Malmquist; Thorp 2025 on the AGN prior  |

---

# Part II — The findings, ranked

Ordered by how much work each does in the thesis, not chronologically.

## 1. pop-cosmos overpredicts SPIRE bright counts, and the excess grows with wavelength

**Compared against:** my compiled table of published differential counts
(`external_spire_differential_counts_compiled.csv`) — Clements, Oliver, Pearson, Glenn,
Varnish, Valiante.

**Result.** Baseline FSPS is too bright in **7 of 7 independent count sources, in all three
bands**: +0.25 / +0.41 / +0.55 dex median at 250 / 350 / 500 µm. That excess is 2.6–3.3× larger
than the inter-survey scatter (0.12 dex), so it isn't a calibration disagreement between
surveys.

**Robustness.** Block bootstrap over sky fields: FSPS excluded at 95% (CI +0.155 to +0.519,
never consistent with zero) while warm templates straddle zero. Leave-one-source-out picks 25%
ALESS in every held-out run. Conclusion unchanged across five different source-selection
choices and four error-floor values.

**Figures.** `popcosmos_count_overlay_thesis.png` (with residual panel),
`popcosmos_differential_count_leave_one_source_out.png`, `bootstrap_forest_plot.png`.

## 2. The dust is too cold, and the model has no luminosity–temperature relation

*The second half of this is the strongest thing in the project and is not yet in any draft.*

**Compared against:** Drew & Casey 2022 (ApJ 930, 142), who calibrated
λ_peak = 92 µm × (L_IR/10¹²)^−0.09 across IRAS, Herschel and SCUBA-2 — and showed **no redshift
evolution** over 0 < z < 2, so it applies to the whole sample without redshift matching.

**Two distinct failures.**

|                       | pop-cosmos (low AGN) |   Drew & Casey |
| --------------------- | -------------------: | -------------: |
| slope η              |     **+0.005** | −0.09 ± 0.01 |
| peak at L_IR = 10¹² |    **126 µm** |    92 ± 2 µm |

The normalisation offset is the cold-dust problem, now measured against a calibrated relation
rather than against ALESS templates I chose. **The flat slope is a different kind of failure:
a missing relationship, not a wrong value.** No single dust temperature can reproduce a trend
across luminosity, so this can't be patched by adjusting a template.

Per luminosity decade the offset changes sign — the model is slightly too warm for faint
galaxies (−0.13 dex at log L_IR 8–9) and clearly too cold for luminous ones (+0.11 dex at
11–12), crossing around 10¹⁰.

**Why this matters most.** Bright counts are dominated by luminous galaxies, exactly where the
model is coldest. So the SED shape, measured entirely independently, predicts the count excess
in finding 1. Two unconnected lines of evidence landing on one defect.

**Related result — the model's dust temperature is implausibly uniform.** 16–84 percentile
spread in peak wavelength is only **0.18 dex**, median 135 µm in *every* luminosity decade.
Verified not a gridding artifact: the grid has ~1 µm spacing near the peak, and sub-grid
parabolic interpolation gives η = +0.005 either way. Real galaxies are far more varied. This is
the same story from the other side — nothing in optical/NIR data tells the model what dust
temperature to assign, so it assigns nearly the same one to everything.

**Figures.** `drew_casey_peak_relation.png`, `fsps_fir_sed_diagnostic.png`.
**Tables.** `drew_casey_peak_comparison.csv`, `drew_casey_slope_fit.csv`.

**Caveat.** I compare median-posterior model SEDs against fitted observational peaks. Both are
luminosity-weighted peak measures but the fitting procedures differ (MCIRSED vs FSPS energy
balance), so a small definitional offset can't be excluded. The flat slope isn't sensitive to
this. Drew & Casey's relation is calibrated on IR-bright samples, so the faintest decade is an
extrapolation of their fit and the least secure row.

## 3. Per-object scatter inflates the bright counts — a second, independent problem

**Compared against:** the Wang 2024 deblended COSMOS catalogue, galaxy by galaxy.

**Result.** Intrinsic per-object scatter of 0.44–0.54 dex. Combined with a steeply falling
count distribution, this promotes faint galaxies into bright bins: of 422 galaxies predicted
above 20 mJy, **only 21 are genuinely bright and 401 are promoted by scatter**.

**Robustness — this is the strong part.** The scatter is stable at SNR cuts of 3, 5, 7 and 10.
If it were observational noise near the detection limit, tightening the cut would shrink it. It
doesn't. And it's the model's, not Wang's deblending — see the Jin cross-match above.

**How much of the count excess does it explain?** It fully accounts for 250 µm, but only about
a third to two-thirds at 350 and 500 µm. **Do not write this up as a closed budget** — the
remainder is where the cold-dust problem sits, and if the scatter explained everything the
cold-dust result would have nowhere to act. See `paradox_decomposition_residuals.csv`.

**Figure.** `scatter_mechanism_explained.png` — predicted vs measured flux per galaxy, and the
same galaxies counted above 20 mJy split 21 real / 401 promoted.

## 4. The far-IR excess localises in redshift, and it shifts with wavelength

**Result.** 250 µm excess comes from z < 1; 500 µm excess from z ≈ 1.5–2.5, the cosmic
star-formation peak. The localisation is **band-dependent** — an earlier version of this figure
claimed a single redshift for all bands, which was wrong.

**Figure.** `z_resolved_fir_diagnosis.png`. **Table.** `z_resolved_excess_localisation.csv`.

## 5. The AGN parameter controls the far-IR peak, and 37% of the catalogue sits high

**Result.** `lnfAGN` controls the FIR peak almost entirely: below f_AGN = 0.1, under 1% of
galaxies peak hot; above 0.3, 67% do. **93.9% of hot-peaking galaxies have f_AGN > 0.3.**
Wien's law confirms Dave's 150 K estimate (20 µm → 145 K), a temperature star formation alone
cannot reach.

**Mechanism.** Energy balance fixes total L_IR but not its distribution, so energy assigned to
a hot torus is energy not available at 250–500 µm. High-f_AGN galaxies put 55% of L_IR at
8–40 µm against 16% for low-f_AGN. At fixed L_IR **and redshift**, high-f_AGN galaxies have
about **half** the predicted 250 µm flux (median ratio 0.53).

*Note on that last number: controlling for redshift is essential. At fixed L_IR alone,
low-f_AGN galaxies sit at systematically higher redshift (median z 2.8 vs 2.0) and are
therefore fainter, which masks the effect and flips the apparent sign.*

**Where the high values sit — answering Dave's question.** Catalogue-wide, not restricted to
FIR-bright objects, and actually *lowest* among them:

| predicted 250 µm flux | fraction f_AGN > 0.3 |
| ---------------------- | -------------------: |
| < 1 mJy                |                  36% |
| 1–5 mJy               |                  48% |
| 10–20 mJy             |                  23% |
| 20–50 mJy             |                  12% |
| > 50 mJy               |                   9% |

It rises steeply with luminosity (12% at log L_IR 8–9 → 78% above 10¹²) and peaks at
z = 1.5–2 (62%).

**This is a known pop-cosmos property, not a bug I found.** Thorp et al. 2025 Sect. 4.5 states
the ln(f_AGN) prior is **bimodal** with modes either side of f_AGN ≈ 3%, and that *"~40% of
model galaxies have f_AGN > 3%"* — the same population as my 37%. They validate against 1,951
Chandra X-ray detections (67% skew to the high mode) and name a more flexible AGN model as
future work. Table 2 confirms the bounds, matching the max f_AGN = 2.82 I measured.

**So the framing is:** a documented bimodality has a previously unexamined consequence for
far-infrared predictions. That gives proper credit and their own caveat supports it.

**Comparison with Kim et al. 1995 / Sanders & Mirabel 1996** (Dave's pointer: observationally
you only reach >50% with AGN contribution above L_FIR = 10¹¹). The model crosses 50% at ~10¹¹,
so the *shape* is right. The problem is the faint end — 27% at 10⁹–10¹⁰ and 12% at 10⁸–10⁹,
where the observational expectation is near zero. **The sharper criticism is a too-high AGN
floor at low luminosity**, consistent with a bimodal prior assigning galaxies to the high mode
absent constraining data. Caveat: Kim uses L_FIR, I quote L_IR (8–1000 µm), which runs higher.

**Figures.** `agn_hot_dust_diagnosis.png`, `agn_fraction_prevalence.png`,
`popcosmos_full_sed_agn_parameter_median_seds.png`.

**Retracted — do not use.** I earlier argued that narrow high-f_AGN posteriors (width 1.09 vs
3.04 for low-f_AGN) showed the fits confidently preferring a large AGN component. With a
bimodal prior, objects pulled into the narrow mode inherit a narrow posterior — that's prior
structure, not information content. Drop any "confidently data-driven" claim.

## 6. Lensing runs the wrong way to help

**Why it came up.** Dave suggested lensing might account for the brightest 500 µm sources.

**Result, and it's counterintuitive.** Published counts already *contain* lensed sources
because nobody removes them, while the model applies no magnification at all. So correcting for
lensing **lowers the observations and widens the model excess** (+1.165 → +1.262 dex at 500 µm
for 20% lensed). Verified symmetric: removing lensed sources from the observations and adding
the missing lensed population to the model give an identical shift in the same direction.

**This closes off the "maybe the observations are inflated" escape route entirely** — a useful
thing to have done, even though the answer is negative.

**Retracted from an earlier version:** I claimed a wavelength trend in Valiante's bright-end
uptick. Propagating errors showed the ratios are consistent with each other well within
uncertainty. What survives: the uptick is real and best measured at 250 µm, and the predicted
lensed fraction is too small to produce it alone.

**Figure.** `lensing_bright_end_assessment.png`. **Table.** `lensing_correction_sensitivity.csv`.

## 7. Literature support from papers citing Wang 2024

Found via ADS citation search.

- **Malefahlo 2026** — same tool (XID+), same field, different wavelength (1.4 GHz radio).
  Independently confirms prior-catalogue purity controls recovered counts: a high-purity prior
  recovers accurate fluxes, a complete prior overestimates counts through spurious detections.
  **This validates my decision to exclude raw Wang counts from the formal comparison** — a
  methods choice I'd otherwise be defending on my own judgement.
- **Quirós-Rojas 2026** — 474 multiple systems in ALMA follow-up of Herschel sources; in
  blended doubles the brightest component contributes on average 64 / 48 / 42% of the flux at
  250 / 350 / 500 µm. A caveat I hadn't considered: bright 500 µm "sources" are often several
  galaxies.
- **Farrah 2026** — the most important factors in recovering obscured luminosities from SED
  fitting are wavelength coverage spanning the SED peak and dense sampling. pop-cosmos has
  neither. **A published, independent statement of my central argument.**

**Honesty note.** All three are read from **abstracts only**. Farrah in particular should be
read in full before it carries weight in the Discussion. Quoted fragments were verified
character-by-character against the ADS abstract text (14/14 verbatim), but four interpretive
statements connecting these papers to pop-cosmos are **mine, not theirs** — see the attribution
audit in `Thesis_part2.md`. Phrase those as "consistent with", not as their findings.

---

# Part III — Figure plan

Ordered as they'd appear. Every file confirmed on disk.

## Main text

| #  | figure                                                          | what it shows                                       | why it's in                            |
| -- | --------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------- |
| 1  | `external_spire_differential_counts_july21_3dex.png`          | The compiled published counts                       | Data section: the observational ruler  |
| 2  | `popcosmos_irac_redshift_histograms.png`                      | IRAC Ch1/Ch2 residuals                              | The control: model works where trained |
| 3  | `popcosmos_count_overlay_thesis.png`                          | Model vs published counts, 3 bands + residual panel | **The headline result**          |
| 4  | `popcosmos_full_sed_multiband_counts.png`                     | Mismatch growing 250→850 µm                       | The wavelength ladder                  |
| 5  | `drew_casey_peak_relation.png`                                | Model vs the observed L_IR–λ_peak relation        | **The strongest new result**     |
| 6  | `popcosmos_full_sed_median_sfr_seds.png`                      | Median SED by SFR bin                               | Cold peak is population-wide           |
| 7  | `popcosmos_mbb_temperature_grid_shapes.png` + `_counts.png` | The MBB template experiment                         | The fix, and 35 K as a physical number |
| 8  | `scatter_mechanism_explained.png`                             | Per-galaxy flux, and 21 real / 401 promoted         | **The second finding**           |
| 9  | `z_resolved_fir_diagnosis.png`                                | Where in redshift the excess lives                  | Diagnostic depth                       |
| 10 | `popcosmos_differential_count_leave_one_source_out.png`       | Cross-validation                                    | Robustness                             |
| 11 | `bootstrap_forest_plot.png`                                   | Block-bootstrap model ranking                       | Robustness, and the 95% exclusion      |

## Appendix

`agn_fraction_prevalence.png`, `agn_hot_dust_diagnosis.png`,
`popcosmos_full_sed_agn_parameter_median_seds.png` (the AGN thread);
`fsps_fir_sed_diagnostic.png`, `popcosmos_full_sed_top5_shape_normalized.png` (SED shape);
`lensing_bright_end_assessment.png`; `popcosmos_clean_independent_count_evaluator_heatmap.png`;
`popcosmos_wang_jin_fsps_ratio_summary.png`;
`popcosmos_differential_count_area_corrected_overlay.png` (the honest ancestor of Fig 3);
`popcosmos_lir_vs_sfr.png`, `popcosmos_lir_offset_by_redshift.png` (the first attempt);
Boris's `fig1_speculator_vs_stored.png`, `fig3_coverage_map.png`, `fig6_example_seds.png`
(credit Boris).

---

# Part IV — Report structure

1. **Introduction** — obscured star formation; pop-cosmos; the energy-balance gap (Boris's
   point, in my words); why counts are the right statistic; scope narrowing.
2. **Data** — model outputs; the nine published count tables and independence criteria; Wang
   and Jin as per-object references. *Fig 1.*
3. **Methods** — flux prediction; template families (ALESS hybrid, MBB grid, Casey); the
   evaluator, error floor, block bootstrap; the caveat list in Part V. *Fig 7 shapes panel.*
4. **Results**
   - 4.1 Validation where the model was trained (IRAC, one paragraph). *Fig 2.*
   - 4.2 First far-IR attempt and why single-band correlations mislead (short).
   - 4.3 The count comparison. *Fig 3.* **Headline.**
   - 4.4 The cause: SED shape and the wavelength ladder. *Figs 4, 6.*
   - 4.5 The observed luminosity–temperature relation. *Fig 5.* **Strongest new result.**
   - 4.6 The template experiment: MBB 35 K, hybrids, degeneracy. *Fig 7.*
   - 4.7 Per-object scatter as a second, independent problem. *Fig 8.*
   - 4.8 Redshift localisation. *Fig 9.*
   - 4.9 Robustness. *Figs 10, 11.*
5. **Discussion** — two independent problems and why only one is fixable by templates; what
   counts can and cannot constrain (template families are degenerate at ≤0.09 dex vs 0.12 dex
   inter-survey scatter); the AGN parameter and Thorp's bimodal prior; the ~30 mJy reliability
   ceiling in a COSMOS-sized field; comparison to Schreiber+2017.
6. **Conclusions** — against the aims; recommendations to the pop-cosmos team.

**The strongest narrative device available:** open Results with the IRAC agreement, then
immediately the count mismatch. Same model, same pipeline, one regime works and one doesn't.
That contrast does more work than either result alone.

**Structural decision to make before writing.** There are **two independent findings** — cold
dust, and per-object scatter. Early drafts treated the second as a complication of the first.
Presenting them as two separate problems, only one of which a template change can fix, is
cleaner and more honest.

---

# Part V — Caveats to state once, in Methods

An examiner will look for these.

1. Wang is an XID+ prior-based deblended catalogue, not a corrected count product. Per-object
   work only. Malefahlo 2026 independently supports this.
2. Wang faint-flux underestimation: ~10% at 250 µm, ~15% at 350, ~25% at 500.
3. **Eddington bias, not Malmquist.** Malmquist biases the mean luminosity of a flux-limited
   sample; Eddington biases counts near a threshold through noise scatter. Mine is Eddington.
   Note that Clements+1999 describe Eddington bias but label it Malmquist — define the term
   explicitly on first use.
4. The 5σ cut follows Clements+1999, which is the validity limit of the analytic flux-boosting
   correction. **The counts comparison itself is unchanged by the cut** — model counts have no
   noise threshold and published counts already carry the authors' own corrections. Above
   ~35 mJy the cut removes zero sources.
5. The 1.278 deg² area (Farmer `FLAG_COMBINED`), and why 2.0 deg² was wrong (factor 1.56;
   ranking reverses with the wrong value).
6. Jin's area is 1.7 deg² (UltraVISTA `goodArea`), confirmed from the paper.
7. IMF conventions: Kennicutt is Salpeter, convert to Chabrier explicitly. *"I am not going to
   mix IMF conventions silently."*
8. SPIRE counts at 350/500 µm in the 5–50 mJy range can be biased high by ~2× from blending
   (SIDES / Béthermin).
9. Median-posterior parameters are used throughout. Given Thorp's bimodal f_AGN prior, **34% of
   galaxies have a 16–84 range spanning the mode boundary**, so their median may correspond to
   neither state. Open question with Boris; state as a limitation.

---

# Part VI — What's left

## Blocking

- No new science analysis blocks starting the report.

- **Scorecard issue resolved for writing.** Use
  `tables/validation_unified_scorecard.csv`, which evaluates the Casey, MBB, ALESS/hybrid and
  FSPS models on the same 174-point, nine-table set. The older family-specific scorecards and
  `table_D_model_family_common.csv` are legacy snapshots and should not supply final quoted
  chi-square values. Keep the clean 74-point, three-survey-family subset as the independence
  sensitivity check rather than calling all nine source tables independent.

## Worth doing

- Read Farrah 2026 in full. It supports the central argument and I only have the abstract.
- Complete bibliography: 33 entries, 5 flagged incomplete or unread (Kim 1995 and Sanders &
  Mirabel 1996 are from Dave's message only; Thorp 2025 needs full details from ADS).
- Ask Boris the sampling question, and whether posterior draws exist rather than only five
  percentiles. If they do, the effect on the counts and on the scatter can be tested directly.

## Deliberately not doing

- **Draine & Li or CIGALE dust families.** My own result shows counts can't distinguish
  template families (differences ≤0.09 dex against 0.12 dex inter-survey scatter), which is the
  principled reason to leave this as future work rather than an omission.
- **Energy balance remains untested.** The check I ran was tautological because the SED file is
  already luminosity-normalised. Further Work, not a gap.

## Things not to write

1. **Don't quote a best-fit emissivity index.** The preferred value shifts between 2.5 and 3.0
   depending on which count sources are included. The counts constrain dust temperature
   (~30–35 K), not β.
2. **Don't say the scatter explains the whole count excess.** It fully accounts for 250 µm but
   only a third to two-thirds at 350 and 500 µm.
3. **Don't claim the high f_AGN values are data-driven** because their posteriors are narrow.
   Retracted — see finding 5.
4. **Don't attribute my interpretations to Malefahlo, Quirós-Rojas or Farrah.** Their numbers,
   my connection to pop-cosmos.
