# Step237 final regime evaluation — locked evidence record

**Audit date:** 2026-11-16  
**Repository target:** `pavisriram18-debug/ACHER-Net-Q1-Reproducibility`  
**Evaluation:** final Step237 M5 (ACHER-SAGE) versus the HGB incumbent  
**Decision:** RETAIN HGB; no conditional-adaptation regime is eligible.

> This is a documentation record of the local Step308–Step312 audit results. It is not a substitute for the executable audit scripts or the original machine-readable outputs. Those local source and output files were not available to the GitHub connector in this session and have not been uploaded by this record.

## 1. Evaluation population and authoritative inputs

The frozen evaluation contains **46,622 mine-year observations** from assessment year 2024, with **1,173 positive next-year higher-hazard transitions** (2.516% prevalence). The fixed decision thresholds are HGB = **0.125** and final Step237 M5 = **0.855**.

Authoritative inputs and SHA-256 digests:

| Artifact | SHA-256 |
|---|---|
| Step134 Q1 feature/target matrix | `ee56c464b373a697d348431c36bdc844630a132c7af3002f4d0433d5ebcd3306` |
| Step151 paired HGB/M5 test predictions | `62368701e07606fc00fe072d036a19a29a19c6a09bec2656435251e4e849fd9a` |
| Step237 final M0–M5 test predictions | `2a35f4f175f2a5de5cf3d8408cde463f3b0c20ef305ed5e72d33b90360342181` |

The final Step237 predictions were aligned to the authoritative Step134 rows using the raw mine identifier and assessment year. The alignment check returned 46,622 rows, no unmatched rows, and no target-label mismatches. The final regime assignments were reconstructed using the original E6.4R feature-processing and regime-definition functions; original regime population counts were reproduced.

## 2. Global performance

All differences below are **M5 − HGB**. Values use final Step237 M5 probabilities at threshold 0.855 and HGB probabilities at threshold 0.125.

| Metric | HGB | Final M5 (ACHER-SAGE) | Difference |
|---|---:|---:|---:|
| Balanced accuracy | 0.711959 | 0.678920 | −0.033039 |
| F1-score | 0.378651 | 0.352006 | −0.026645 |
| Matthews correlation coefficient | 0.364687 | 0.334932 | −0.029755 |
| PR-AUC / average precision | 0.231406 | 0.203259 | −0.028147 |
| ROC-AUC | 0.847432 | 0.845496 | −0.001936 |

Confusion matrices (TN, FP, FN, TP):

- **HGB:** 44,374; 1,075; 648; 525.
- **Final M5:** 44,548; 901; 730; 443.

Final M5 produces fewer false positives but more false negatives and lower values on all five reported global metrics. The evidence does not support claiming predictive superiority for ACHER-SAGE.

## 3. Final Step237 regime evaluation

Twelve regimes were evaluated: four missingness quartiles, four feature-extremeness quartiles, and four missingness–extremeness intersections. The table reports sample size, positive count, and point-estimate differences (M5 − HGB).

| Regime | n | Positive | Δ Balanced accuracy | Δ F1 | Δ MCC | Δ PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MISS_Q1 | 12,171 | 700 | −0.041939 | −0.043394 | −0.048139 | −0.032459 |
| MISS_Q2 | 18,890 | 369 | −0.019817 | −0.000485 | −0.001142 | −0.024701 |
| MISS_Q3 | 11,745 | 38 | −0.013072 | −0.002242 | −0.004194 | +0.030247 |
| MISS_Q4 | 3,816 | 66 | 0.000000 | 0.000000 | 0.000000 | −0.004722 |
| EXT_Q1 | 10,546 | 129 | −0.000048 | −0.001595 | −0.002199 | −0.023434 |
| EXT_Q2 | 12,027 | 101 | −0.004867 | −0.005848 | −0.004590 | +0.001376 |
| EXT_Q3 | 12,407 | 317 | −0.033465 | −0.026153 | −0.022441 | −0.041914 |
| EXT_Q4 | 11,642 | 626 | −0.041560 | −0.032174 | −0.040682 | −0.032484 |
| MISS_HIGH_EXT_HIGH | 7,914 | 99 | −0.004923 | −0.002780 | −0.002876 | −0.021315 |
| MISS_HIGH_EXT_LOW | 7,647 | 5 | 0.000000 | 0.000000 | 0.000000 | −0.002004 |
| MISS_LOW_EXT_HIGH | 16,135 | 844 | −0.041802 | −0.033797 | −0.039770 | −0.036476 |
| MISS_LOW_EXT_LOW | 14,926 | 225 | −0.002188 | −0.003623 | −0.003219 | −0.007232 |

### Adaptation gate

The three-metric gate requires a positive observed M5−HGB difference **and** a confidence-interval lower bound above zero for each of F1, MCC, and PR-AUC. **Zero of the 12 regimes passed.** The governance decision is **RETAIN HGB**; no regime-specific adaptation is authorized by this evaluation.

## 4. Bootstrap corrections and Step312 p-value reporting

The regime bootstrap used 2,000 paired row-resamples. Corrected streams preserve point estimates and confidence intervals while fixing metric-specific replicate handling:

- **Balanced accuracy:** one-class target resamples are invalid; 1,991–2,000 valid replicates.
- **F1:** use the `zero_division=0` convention and retain single-class target resamples; 2,000 valid replicates in each regime.
- **MCC:** return zero for a zero denominator when both target classes are present; one-class target resamples are invalid; 1,989–2,000 valid replicates.
- **PR-AUC:** correct zero cumulative-weight handling; one-class target resamples are invalid; 1,989–2,000 valid replicates.

Step312 recomputed add-one Monte Carlo p-values using the row-specific valid replicate count `R`:

`p_corrected = min(1, (p_empirical * R + 2) / (R + 1))`

Verification reported 48 rows, valid replicate counts from 1,989 to 2,000, 12 rows per metric, and 14 empirical zero-p-value rows. All 14 now have non-zero Monte Carlo estimates (0.001 at the displayed precision where `R=2,000`). Point estimates and confidence intervals were explicitly checked and remained unchanged.

An empirical tail proportion of zero is a finite-resampling boundary estimate, **not a literal probability of zero**. Report the add-one Monte Carlo estimate; do not state `p=0`.

## 5. Reproducibility record — local outputs

The following output artefacts were verified during the local audit. They have not been included in this documentation-only GitHub update.

| Step | Local artefact | SHA-256 |
|---|---|---|
| Step308 | `step237_regime_metrics.csv` | `8d9d3a0f580c42fa5f32fcd58fc7227b824dea6e4a01928562895f16e349022c` |
| Step308 | `step237_pairwise_bootstrap.csv` | `ecd126cf123b647c71a6ee37b5e009163941fc5ddc530588317b520d7f6e47e5` |
| Step308 | final test-regime assignment Parquet | `9fddb1e3a4c1c5d06dac338ca5546e1fbfd09e29ba985a45363039957956b9ee` |
| Step309R | `step237_mcc_bootstrap_corrected.csv` | `2560e92f7b2e11589bfe79bd8a3309dbde3ecf55549b65aba6f837d313b46ccc` |
| Step310 | `step237_f1_bootstrap_corrected.csv` | `d6d30e2ef17071c9755d8c75cf7d27589437b3f4804678742021b7d0e55f24d7` |
| Step310 | `step237_final_three_metric_gate_corrected.csv` | `5dfd20012344dd14d0155fc4e0832e72663cfc876851add2bdb5ec6fc11a660f` |
| Step312 input | consolidated bootstrap CSV | `361f8c409595001e3bceae3acbcd46e36e34a6805e2deeccb9663811976b8da6` |

The locked local evidence folder is:

`data/processed/validation/acher_net_q1_final/step237_FINAL_REGIME_EVIDENCE_LOCKED_20261116/`

It contains locked regime metrics, corrected bootstrap results, the final adaptation gate, evidence notes, and the Step311 manifest. Step312 adds the corrected p-value reporting CSV and its manifest in the same folder.

## 6. Scope and interpretation

This audit does **not** fit a new model, change either decision threshold, alter the frozen test population, or tune a candidate on the test set. It corrects regime-level bootstrap computation/reporting and evaluates the final Step237 prediction series against the HGB incumbent.

The Step151 historical M5 series is distinct from final Step237 M5. Do not mix Step151 regime results with the final Step237 comparison.

Describe the frozen evaluation as assessment-year 2024 records with corresponding 2025 outcomes. The 2026→2027 stage remains prospective unless and until future outcomes are available and evaluated.

## 7. Remaining source-code publication task

To make this update fully executable from the repository, add the exact local source files used for Steps308–312 and the corresponding JSON manifests. Copy them from the project working tree, review them against their recorded hashes, and commit them without including confidential or licensed raw mine-level records. Do not reconstruct or substitute scripts based only on this summary.
