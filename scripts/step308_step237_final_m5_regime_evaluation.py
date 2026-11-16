import os
import re
import json
import hashlib
import importlib.util
import types
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path.cwd()

ORIGINAL = ROOT / "scripts/279R_acher_net_q1_e64r_conditional_adaptation_benefit_locked855.py"
FAST = ROOT / "scripts/279R_fast_acher_net_q1_e64r_conditional_adaptation_benefit_locked855.py"

STEP151 = ROOT / "data/processed/validation/acher_net_q1_statistical/step151_paired_hgb_m5_test_predictions.parquet"
STEP237 = ROOT / "data/processed/validation/acher_net_q1_acher_sage/final_step237_acher_sage_optimization/step237_final_m0_m5_test_predictions.parquet"

OLD_DIR = ROOT / "data/processed/validation/acher_net_q1_final/stepE64R_conditional_adaptation_benefit"
OUT = ROOT / "data/processed/validation/acher_net_q1_final/step237_final_m5_regime_evaluation_20261116"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

print("========== A. INPUT FILE INTEGRITY ==========")

for p in [ORIGINAL, FAST, STEP151, STEP237]:
    if not p.exists():
        raise FileNotFoundError(p)
    print(p.name, sha256(p))

print("\n========== B. LOAD ORIGINAL PREPROCESSING ==========")

orig = load_module("e64r_original_step308", ORIGINAL)

for attr in ["MATRIX", "FEATURE_MANIFEST", "PAIRED"]:
    if hasattr(orig, attr):
        p = Path(getattr(orig, attr))
        print(attr, "=", p, "| exists:", p.exists())

features = orig.load_features()
print("Selected features:", len(features))

matrix = orig.load_matrix(features)
orig.validate_population(matrix)

print("Feature matrix rows:", len(matrix))
print("Feature matrix columns:", matrix.shape[1])

if len(features) != 108:
    raise RuntimeError(f"Expected 108 selected predictors, got {len(features)}")

if len(matrix) != 233128:
    raise RuntimeError(f"Unexpected Step134 matrix rows: {len(matrix)}")

print("\n========== C. RECONSTRUCT ORIGINAL RELIABILITY VARIABLES ==========")

matrix["ROW_MISSING_FRACTION"] = orig.compute_missingness(
    matrix, features
)

references = orig.fit_development_references(
    matrix, features
)

if len(references) != len(features):
    raise RuntimeError("Development reference count mismatch.")

matrix["COVARIATE_EXTREMENESS"] = orig.compute_extremeness(
    matrix, features, references
)

missingness_cuts = orig.development_quantiles(
    matrix, matrix["ROW_MISSING_FRACTION"]
)

extremeness_cuts = orig.development_quantiles(
    matrix, matrix["COVARIATE_EXTREMENESS"]
)

print("Missingness cutpoints:", missingness_cuts)
print("Extremeness cutpoints:", extremeness_cuts)

print("\n========== D. RECREATE DEVELOPMENT-DEFINED REGIMES ==========")

matrix["MISSINGNESS_QUARTILE"] = orig.make_quartile_regime(
    matrix["ROW_MISSING_FRACTION"],
    missingness_cuts,
    "MISS",
)

matrix["EXTREMENESS_QUARTILE"] = orig.make_quartile_regime(
    matrix["COVARIATE_EXTREMENESS"],
    extremeness_cuts,
    "EXT",
)

missingness_binary = orig.make_binary_regime(
    matrix["ROW_MISSING_FRACTION"],
    missingness_cuts["q50"],
    "MISS",
)

extremeness_binary = orig.make_binary_regime(
    matrix["COVARIATE_EXTREMENESS"],
    extremeness_cuts["q50"],
    "EXT",
)

matrix["MISSINGNESS_EXTREMENESS_2X2"] = orig.make_2x2_regime(
    missingness_binary,
    extremeness_binary,
)

print("Regimes reconstructed using original source functions.")

print("\n========== E. LOAD STEP151 AND FINAL STEP237 ==========")

paired = orig.load_step151()

step237 = pd.read_parquet(STEP237)
final_m5 = step237[
    step237["MODEL"].astype(str) == "M5_ACHER_SAGE"
].copy()

final_m5 = final_m5[
    final_m5["AS_OF_YEAR"].astype(int) == 2024
].copy()

if len(paired) != 46622 or len(final_m5) != 46622:
    raise RuntimeError(
        f"Unexpected model population: Step151={len(paired)}, "
        f"Step237 M5={len(final_m5)}"
    )

if final_m5["THRESHOLD"].nunique() != 1:
    raise RuntimeError("Final M5 has inconsistent thresholds.")

if not np.isclose(float(final_m5["THRESHOLD"].iloc[0]), 0.855):
    raise RuntimeError("Final M5 threshold is not 0.855.")

for frame_name, frame in [
    ("Step151", paired),
    ("Step237 M5", final_m5),
]:
    if frame.duplicated(["MINE_ID", "AS_OF_YEAR"]).any():
        raise RuntimeError(f"{frame_name} has duplicate raw mine/year keys.")

print("Step151 rows:", len(paired))
print("Final Step237 M5 rows:", len(final_m5))
print("Final M5 stored threshold:", float(final_m5["THRESHOLD"].iloc[0]))

print("\n========== F. EXACT RAW-ID AND LABEL MATCH ==========")

lookup_columns = [
    "MINE_ID",
    "AS_OF_YEAR",
    "Y_TRUE",
    "PREDICTION_PROBABILITY",
    "PREDICTION",
    "THRESHOLD",
]

lookup = final_m5[lookup_columns].copy()

joined = paired.merge(
    lookup,
    on=["MINE_ID", "AS_OF_YEAR"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

matched = int((joined["_merge"] == "both").sum())
unmatched = int((joined["_merge"] != "both").sum())
label_mismatches = int(
    (joined["Y_TRUE_x"].astype(int) != joined["Y_TRUE_y"].astype(int)).sum()
)

print("Exact raw ID/year matches:", matched)
print("Unmatched rows:", unmatched)
print("Target label mismatches:", label_mismatches)

if matched != 46622 or unmatched != 0 or label_mismatches != 0:
    raise RuntimeError("Step151-to-Step237 exact identity validation failed.")

if joined["PREDICTION_PROBABILITY"].isna().any():
    raise RuntimeError("Missing final Step237 M5 probabilities.")

# Replace only the M5 score with the exact Step237 probability.
joined["E64R_M5_PROB"] = orig.normalize_probability(
    joined["PREDICTION_PROBABILITY"],
    "Final Step237 M5 probability",
)

joined["E64R_M5_PRED"] = pd.to_numeric(
    joined["PREDICTION"],
    errors="coerce",
)

# Verify stored binary predictions correspond to the locked threshold.
rethresholded = (
    joined["E64R_M5_PROB"].to_numpy() >= 0.855
).astype(np.int8)

if not np.array_equal(
    rethresholded,
    joined["E64R_M5_PRED"].to_numpy(dtype=np.int8),
):
    raise RuntimeError("Final M5 prediction/threshold mismatch.")

print("Final M5 threshold verification passed.")

# Keep only the fields needed by the original alignment function.
paired_final = joined.drop(columns=["_merge", "Y_TRUE_y"]).copy()
paired_final = paired_final.rename(columns={"Y_TRUE_x": "Y_TRUE"})

print("\n========== G. ORIGINAL OCCURRENCE-BASED ALIGNMENT ==========")

aligned = orig.align_authoritative_predictions(
    matrix,
    paired_final,
)

if len(aligned) != 46622:
    raise RuntimeError(f"Aligned population changed: {len(aligned)}")

if aligned["_merge"].astype(str).ne("both").any():
    raise RuntimeError("One or more final predictions did not align.")

print("Final aligned rows:", len(aligned))
print("Final positive labels:", int(aligned["TARGET"].sum()))

print("\n========== H. VALIDATE REGIME DEFINITIONS ==========")

old_assignment = pd.read_csv(
    OLD_DIR / "e64r_test_regime_assignments.csv"
)

old_definition = pd.read_csv(
    OLD_DIR / "e64r_regime_definition.csv"
)

if len(old_assignment) != 46622:
    raise RuntimeError("Original regime assignment population mismatch.")

regime_map = {
    "missingness_quartile": "MISSINGNESS_QUARTILE",
    "extremeness_quartile": "EXTREMENESS_QUARTILE",
    "missingness_extremeness_2x2": "MISSINGNESS_EXTREMENESS_2X2",
}

# Check regime population counts against the original E6.4R assignments.
for family, column in regime_map.items():
    observed = aligned[column].value_counts().sort_index()
    previous = old_assignment[column].value_counts().sort_index()

    comparison = pd.concat(
        [previous.rename("original"), observed.rename("reconstructed")],
        axis=1,
    ).fillna(0).astype(int)

    print(f"\n{family}")
    print(comparison.to_string())

    if not comparison["original"].equals(comparison["reconstructed"]):
        raise RuntimeError(f"Regime counts changed for {family}.")

print("\nAll original regime distributions reproduced.")

print("\n========== I. LOAD CORRECTED FAST BOOTSTRAP ==========")

fast_text = FAST.read_text(encoding="utf-8-sig")

old_block = """        precision = (
            cumulative_positive[valid_rows]
            / cumulative_weight[valid_rows]
        )"""

corrected_block = """        precision = np.divide(
            cumulative_positive[valid_rows],
            cumulative_weight[valid_rows],
            out=np.zeros_like(
                cumulative_positive[valid_rows],
                dtype=np.float64,
            ),
            where=cumulative_weight[valid_rows] > 0.0,
        )"""

if fast_text.count(old_block) != 1:
    raise RuntimeError(
        "The exact previously audited PR-AUC bug block was not found once."
    )

corrected_text = fast_text.replace(old_block, corrected_block, 1)

fast = types.ModuleType("e64r_corrected_bootstrap_step308")
fast.__file__ = str(FAST)
exec(compile(corrected_text, str(FAST), "exec"), fast.__dict__)

fast.N_BOOTSTRAP = 2000
fast.SEED = 20260902
fast.HGB_THRESHOLD = 0.125
fast.M5_THRESHOLD = 0.855

print("Bootstrap replicates:", fast.N_BOOTSTRAP)
print("Seed:", fast.SEED)
print("HGB threshold:", fast.HGB_THRESHOLD)
print("Final Step237 M5 threshold:", fast.M5_THRESHOLD)

print("\n========== J. GLOBAL METRIC CROSS-CHECK ==========")

y = aligned["TARGET"].to_numpy(dtype=np.int8)
hgb = aligned["E64R_HGB_PROB"].to_numpy(dtype=float)
m5 = aligned["E64R_M5_PROB"].to_numpy(dtype=float)

global_rows = []

for metric in ["balanced_accuracy", "f1", "mcc", "pr_auc"]:
    hgb_value = fast.metric_value(y, hgb, 0.125, metric)
    m5_value = fast.metric_value(y, m5, 0.855, metric)

    global_rows.append({
        "metric": metric,
        "HGB_Step151": hgb_value,
        "M5_ACHER_SAGE_Step237": m5_value,
        "delta_M5_minus_HGB": m5_value - hgb_value,
    })

global_df = pd.DataFrame(global_rows)
print(global_df.to_string(index=False))

expected = {
    "balanced_accuracy": (0.7119587234, 0.6789198453),
    "f1": (0.3786512802, 0.3520063568),
    "mcc": (0.3646868299, 0.3349318357),
    "pr_auc": (0.2314056156, 0.2032590572),
}

for _, row in global_df.iterrows():
    e_hgb, e_m5 = expected[row["metric"]]
    if not np.isclose(row["HGB_Step151"], e_hgb, atol=2e-7):
        raise RuntimeError(f"Global HGB {row['metric']} changed.")
    if not np.isclose(row["M5_ACHER_SAGE_Step237"], e_m5, atol=2e-7):
        raise RuntimeError(f"Final Step237 M5 {row['metric']} changed.")

print("[PASS] Global Step151 HGB and Step237 M5 metrics reproduced.")

print("\n========== K. FINAL STEP237 REGIME EVALUATION ==========")

all_metrics = []
all_bootstraps = []

for family, column in regime_map.items():
    print(f"\nEvaluating {family} ...")

    metric_rows, bootstrap_rows = fast.evaluate_family(
        aligned,
        family,
        column,
    )

    all_metrics.extend(metric_rows)
    all_bootstraps.extend(bootstrap_rows)

metric_df = pd.DataFrame(all_metrics)
bootstrap_df = pd.DataFrame(all_bootstraps)

if len(metric_df) != 48 or len(bootstrap_df) != 48:
    raise RuntimeError(
        f"Expected 48 metric and 48 bootstrap rows; received "
        f"{len(metric_df)} and {len(bootstrap_df)}."
    )

# Recreate deterministic seeds used by the original E6.4R evaluator.
seed_match = re.search(
    r"hexdigest\(\)\s*\[\s*:\s*(\d+)\s*\]",
    fast_text,
)
if not seed_match:
    raise RuntimeError("Cannot recover original stable-hash seed rule.")

digest_chars = int(seed_match.group(1))
seeds = []

for _, row in bootstrap_df.iterrows():
    key = f"{row['regime_family']}|{row['regime']}|{row['metric']}"
    stable_hash = int(
        hashlib.sha256(key.encode("utf-8")).hexdigest()[:digest_chars],
        16,
    )
    seeds.append(fast.SEED + (stable_hash % 1_000_000))

bootstrap_df["bootstrap_seed"] = seeds
bootstrap_df["M5_model"] = "M5_ACHER_SAGE_Step237"
bootstrap_df["M5_threshold"] = 0.855
bootstrap_df["HGB_threshold"] = 0.125
bootstrap_df["test_year"] = 2024

metric_df["M5_model"] = "M5_ACHER_SAGE_Step237"
metric_df["M5_threshold"] = 0.855
metric_df["HGB_threshold"] = 0.125
metric_df["test_year"] = 2024

print("\n========== L. REGIME METRICS ==========")
print(
    metric_df[
        [
            "regime_family", "regime", "n", "positives",
            "metric", "HGB", "M5", "delta_M5_minus_HGB"
        ]
    ].to_string(index=False)
)

print("\n========== M. PAIRED BOOTSTRAP INTERVALS ==========")
print(
    bootstrap_df[
        [
            "regime_family", "regime", "metric", "n",
            "observed_diff", "ci_lower", "ci_upper",
            "p_two_sided", "valid_bootstrap", "bootstrap_seed"
        ]
    ].to_string(index=False)
)

print("\n========== N. FINAL THREE-METRIC ADAPTATION GATE ==========")

principal = bootstrap_df[
    bootstrap_df["metric"].astype(str).str.lower().isin(
        ["f1", "mcc", "pr_auc"]
    )
].copy()

if len(principal) != 36:
    raise RuntimeError("Expected 36 principal-metric bootstrap rows.")

eligibility = []

for (family, regime), group in principal.groupby(
    ["regime_family", "regime_family"], dropna=False
):
    pass

for (family, regime), group in principal.groupby(
    ["regime_family", "regime"]
):
    checks = {}
    for metric in ["f1", "mcc", "pr_auc"]:
        selected = group[group["metric"].astype(str).str.lower() == metric]
        if len(selected) != 1:
            raise RuntimeError(f"Missing/duplicate {family}/{regime}/{metric}")
        r = selected.iloc[0]
        checks[metric] = bool(
            r["observed_diff"] > 0 and r["ci_lower"] > 0
        )

    eligibility.append({
        "regime_family": family,
        "regime": regime,
        "F1_eligible": checks["f1"],
        "MCC_eligible": checks["mcc"],
        "PR_AUC_eligible": checks["pr_auc"],
        "ALL_THREE_ELIGIBLE": all(checks.values()),
    })

gate_df = pd.DataFrame(eligibility)

print(gate_df.to_string(index=False))
eligible = gate_df[gate_df["ALL_THREE_ELIGIBLE"]]
print("Final Step237 eligible regimes:", len(eligible))

print("\n========== O. WRITE SUPPLEMENTARY OUTPUTS ==========")

OUT.mkdir(parents=True, exist_ok=True)

metrics_path = OUT / "step237_regime_metrics.csv"
bootstrap_path = OUT / "step237_pairwise_bootstrap.csv"
assignments_path = OUT / "step237_final_m5_regime_assignments.parquet"
global_path = OUT / "step237_global_metrics.csv"
gate_path = OUT / "step237_three_metric_adaptation_gate.csv"
manifest_path = OUT / "step308_run_manifest.json"

metric_df.to_csv(metrics_path, index=False)
bootstrap_df.to_csv(bootstrap_path, index=False)
global_df.to_csv(global_path, index=False)
gate_df.to_csv(gate_path, index=False)

assignment_columns = [
    c for c in [
        "MINE_ID", "AS_OF_YEAR", "TARGET",
        "E64R_HGB_PROB", "E64R_M5_PROB",
        "E64R_M5_PRED",
        "ROW_MISSING_FRACTION", "COVARIATE_EXTREMENESS",
        "MISSINGNESS_QUARTILE", "EXTREMENESS_QUARTILE",
        "MISSINGNESS_EXTREMENESS_2X2",
        "OCCURRENCE",
    ] if c in aligned.columns
]

aligned[assignment_columns].to_parquet(
    assignments_path,
    index=False,
)

manifest = {
    "experiment": "Step237_Final_M5_Regime_Evaluation",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_year": 2024,
    "rows": len(aligned),
    "positive_cases": int(aligned["TARGET"].sum()),
    "negative_cases": int((aligned["TARGET"] == 0).sum()),
    "selected_feature_count": len(features),
    "development_years": [2020, 2021, 2022],
    "validation_year": 2023,
    "HGB_model": "HGB_Step151",
    "M5_model": "M5_ACHER_SAGE_Step237",
    "HGB_threshold": 0.125,
    "M5_threshold": 0.855,
    "bootstrap_replicates": 2000,
    "base_seed": 20260902,
    "regime_specific_seed_digest_characters": digest_chars,
    "comparison": "Final Step237 M5 minus incumbent Step151 HGB",
    "alignment": {
        "Step151_to_Step237_raw_ID_year_matches": matched,
        "unmatched": unmatched,
        "target_label_mismatches": label_mismatches,
        "Step134_to_prediction_alignment": "original occurrence-based function",
        "regime_counts_match_original_E6.4R": True,
    },
    "PR_AUC_correction": (
        "Zero cumulative bootstrap score-group weights are handled "
        "using numpy.divide with where=cumulative_weight > 0."
    ),
    "no_retraining": True,
    "no_threshold_tuning": True,
    "regime_cutpoints_source": "Development years 2020-2022 only",
    "input_sha256": {
        str(p): sha256(p)
        for p in [ORIGINAL, FAST, STEP151, STEP237]
    },
    "outputs": {
        p.name: sha256(p)
        for p in [
            metrics_path, bootstrap_path, assignments_path,
            global_path, gate_path
        ]
    },
    "adaptation_eligible_regimes": int(len(eligible)),
}

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("Metrics:", metrics_path)
print("Bootstrap:", bootstrap_path)
print("Regime assignments:", assignments_path)
print("Global metrics:", global_path)
print("Adaptation gate:", gate_path)
print("Manifest:", manifest_path)

print("\n========== P. INTEGRITY SUMMARY ==========")
print("Original E6.4R files were not modified.")
print("Original Step151 and Step237 Parquet files were not modified.")
print("New metric comparisons:", len(metric_df))
print("New bootstrap comparisons:", len(bootstrap_df))
print("Final Step237 adaptation-eligible regimes:", len(eligible))
print("STEP308_COMPLETE")
