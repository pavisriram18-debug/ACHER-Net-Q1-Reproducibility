# ACHER-Net — Correct Completion Package

This package fixes the main target-lineage problem discovered in the sequential continual-learning experiment.

## Authoritative research rule

The Q1 predictive target is:

`NEXT_YEAR_HIGHER_HAZARD_STATE`

The previous sequential runner used:

`EMERGING_HAZARD_NEXT_YEAR`

Those are different targets. The previous sequential metrics must therefore remain outside the Q1 manuscript.

## What this package does

### 1. `ACHER_Net_target_transition_audit_2026.py`

Run this first.

It reconstructs mine-year state transitions from the project's upstream state artifact, compares the reconstructed `STATE_CHANGE` and `HIGHER_STATE` labels with Step-129 for 2022, 2023 and 2024, and writes the exact transition/negative-class decomposition.

**Acceptance gate:** all three authoritative horizons must show `EXACT_MATCH=True` before an upstream-derived state target is treated as the same lineage as Q1.

It also checks 2025->2026 as an **interim partial-2026 observation**, not as a final annual evaluation.

### 2. `ACHER_Net_authoritative_sequential_q1.py`

Run this second.

It uses the authoritative Step-134 target and runs only these three prequential stages:

- 2021 adaptation → 2022 score → 2023 outcome
- 2022 adaptation → 2023 score → 2024 outcome
- 2023 adaptation → 2024 score → 2025 outcome

The final stage is the locked Q1 2024→2025 evaluation.

`EMERGING_HAZARD_NEXT_YEAR` is explicitly ignored.

### 3. `ACHER_Net_2026_prospective_2027_readiness.py`

Run this third.

It determines whether the project already has a complete **authorized 108-feature 2026 predictor snapshot** for prospective 2027 inference.

It does not invent a 2027 result and never treats incomplete 2026 observations as a completed annual target.

## Windows commands

From the ACHER-NET project root:

```powershell
python .\ACHER_Net_target_transition_audit_2026.py
```

Then:

```powershell
python .\ACHER_Net_authoritative_sequential_q1.py
```

Then:

```powershell
python .\ACHER_Net_2026_prospective_2027_readiness.py
```

The sequential runner auto-discovers an authoritative 108-feature contract CSV. If there is more than one candidate, pass it explicitly:

```powershell
python .\ACHER_Net_authoritative_sequential_q1.py --manifest ".\path\to\authoritative_108_contract.csv"
```

## Outputs to keep

`data\processed\validation\acher_net_q1_final\target_transition_audit_2026\`

- `target_reconstruction_validation.csv`
- `2024_to_2025_state_transition_table.csv`
- `2024_to_2025_negative_class_decomposition.csv`
- `target_transition_audit_summary.json`

`data\processed\validation\acher_net_q1_final\sequential_authoritative_q1\`

- `authoritative_target_coverage.csv`
- `sequential_stage_metrics.csv`
- `sequential_stage_predictions.csv`
- `sequential_stage_comparison.csv`
- `sequential_model_evolution_log.csv`
- `sequential_run_manifest.json`

`data\processed\validation\acher_net_q1_final\prospective_2027_readiness\`

- `prospective_2027_readiness.json`

## Manuscript boundary after completion

Keep the published Q1 evidence centered on the frozen 2024 observations with corresponding 2025 outcomes.

A current 2026 data layer may be described as a prospective update/readiness layer, but it is not an annual 2026 outcome evaluation and it must not be used to create an unsupported 2025→2026 performance claim.
