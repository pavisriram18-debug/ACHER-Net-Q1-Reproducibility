# ACHER-Net — Step237 Final Regime Evidence

**Evidence freeze:** 16 November 2026
**Scope:** Final Step237 M5 versus HGB regime evaluation and corrected bootstrap reporting.

## 1. Main result

The frozen evaluation contains 46,622 mine-year observations from assessment year 2024, with corresponding 2025 outcomes and 1,173 positive higher-hazard transitions.

Decision thresholds:
- HGB incumbent: 0.125
- Final Step237 M5 (ACHER-SAGE): 0.855

| Metric | HGB | Final M5 |
|---|---:|---:|
| Balanced accuracy | 0.711959 | 0.678920 |
| F1-score | 0.378651 | 0.352006 |
| MCC | 0.364687 | 0.334932 |
| PR-AUC | 0.231406 | 0.203259 |
| ROC-AUC | 0.847432 | 0.845496 |

The final three-metric adaptation gate requires positive observed differences and confidence-interval lower bounds above zero for F1, MCC, and PR-AUC. None of the 12 evaluated regimes passed. The evidence therefore supports **RETAIN HGB**, not conditional adaptation or ACHER-SAGE predictive superiority.

## 2. Source files

- `279R_acher_net_q1_e64r_conditional_adaptation_benefit_locked855.py`: original E6.4R preprocessing, feature and regime definitions.
- `279R_fast_acher_net_q1_e64r_conditional_adaptation_benefit_locked855.py`: fast bootstrap implementation used by the evaluation lineage.
- `step308_step237_final_m5_regime_evaluation.py`: reconstructs and evaluates the final Step237 M5 series against HGB.
- `step309r_correct_mcc_bootstrap.py`: corrected MCC bootstrap handling.
- `step310_correct_f1_bootstrap.py`: corrected F1 bootstrap handling.
- `step311_final_evidence_consolidation.py`: consolidates corrected metric streams and the adaptation gate.
- `reproduce_step312_pvalue_reporting.py`: **post-hoc reproduction helper**, not the original Step312 execution script. It applies the documented row-specific add-one p-value formula and writes a separate output file.

The older `step309_correct_mcc_bootstrap.py` is intentionally excluded; the corrected implementation is Step309R.

## 3. Step312 p-value reporting

The final reporting calculation uses the row-specific number of valid bootstrap replicates, R:

`p_corrected = min(1, (p_two_sided * R + 2) / (R + 1))`

The final locked CSV was independently checked: all 48 rows matched this formula. The Step312 manifest records the output digest and confirms that point estimates, confidence intervals, and the adaptation gate were unchanged.

The Step311 source is retained unchanged as the historical consolidation script. The separate Step312 helper makes the final p-value correction reproducible without misrepresenting its provenance.

## 4. Reproduction prerequisites

The executable evaluation scripts expect the original ACHER-NET project root as the working directory and require the original local project artifacts, including:

- Step134 Q1 feature/target matrix.
- Step151 paired HGB/M5 test predictions.
- Step237 final M0–M5 test predictions.
- Step169R selected-feature manifest.
- Original E6.4R regime-definition and assignment artifacts.

These data files are not included in this release. Raw mine-level identifiers, row-level predictions, Parquet assignments, and licensed or sensitive records must remain outside the public repository.

Install the project's pinned dependencies from its root `requirements.txt` before executing the source scripts.

## 5. Execution sequence

The original evaluation scripts require the full authorized ACHER-NET project artifacts. Run Steps 308?311 from the full project root, not from a documentation-only clone.

The following PowerShell sequence shows the required environment variables and output directories. Restore the local data prerequisites before executing it.

```powershell
$root = (Get-Location).Path

$step308 = Join-Path $root "data\processed\validation\acher_net_q1_final\step237_final_m5_regime_evaluation_20261116"
$step309 = Join-Path $root "data\processed\validation\acher_net_q1_final\step237_MCC_BOOTSTRAP_CORRECTED_20261116"
$step310 = Join-Path $root "data\processed\validation\acher_net_q1_final\step237_F1_BOOTSTRAP_CORRECTED_20261116"
$step311 = Join-Path $root "data\processed\validation\acher_net_q1_final\step237_FINAL_REGIME_EVIDENCE_LOCKED_20261116"

python .\scripts\step308_step237_final_m5_regime_evaluation.py
if ($LASTEXITCODE -ne 0) { throw "Step308 failed." }

$env:S309_ROOT = $root
python .\scripts\step309r_correct_mcc_bootstrap.py
if ($LASTEXITCODE -ne 0) { throw "Step309R failed." }

$env:S310_ROOT = $root
$env:S310_OUT = $step310
python .\scripts\step310_correct_f1_bootstrap.py
if ($LASTEXITCODE -ne 0) { throw "Step310 failed." }

$env:S311_ROOT = $root
$env:S311_STEP308 = $step308
$env:S311_STEP309 = $step309
$env:S311_STEP310 = $step310
$env:S311_OUT = $step311

python .\scripts\step311_final_evidence_consolidation.py
if ($LASTEXITCODE -ne 0) { throw "Step311 failed." }
```

The post-hoc Step312 reproduction helper can be run from the repository root containing this release:

`python scripts/reproduce_step312_pvalue_reporting.py`

It writes a separately named reproduced CSV and manifest. It is not the original Step312 execution source and does not overwrite the historical Step312 output.
## 6. Interpretation boundaries

- The test period is assessment year 2024 with corresponding 2025 outcomes; do not describe this as two separate test-year cohorts.
- The Step151 historical M5 series must not be mixed with the final Step237 M5 series.
- Distribution shift is a reassessment signal, not proof of predictive degradation.
- Bootstrap intervals quantify test-sample resampling uncertainty conditional on the fitted models; they are not between-run training uncertainty.
- Empirical zero tail estimates are finite-bootstrap boundary estimates, not literal zero probabilities.
- Do not claim causal effects or demonstrated operational benefit from these predictive evaluations.
- The 2026-to-2027 stage remains prospective until future outcomes are available and evaluated.

## 7. Evidence files

Aggregate CSVs and manifests are provided under `docs/reproducibility/results/` and `docs/reproducibility/manifests/`. The row-level assignment Parquet is deliberately excluded.