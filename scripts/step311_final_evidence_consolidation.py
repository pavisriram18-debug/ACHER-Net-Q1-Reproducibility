import os, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

root = Path(os.environ["S311_ROOT"])
step308 = Path(os.environ["S311_STEP308"])
step309 = Path(os.environ["S311_STEP309"])
step310 = Path(os.environ["S311_STEP310"])
out = Path(os.environ["S311_OUT"])

assignment_path = step308 / "step237_final_m5_regime_assignments.parquet"
metrics_path = step308 / "step237_regime_metrics.csv"
bootstrap_path = step308 / "step237_pairwise_bootstrap.csv"
f1_path = step310 / "step237_f1_bootstrap_corrected.csv"
mcc_path = step309 / "step237_mcc_bootstrap_corrected.csv"
f1_gate_path = step310 / "step237_final_three_metric_gate_corrected.csv"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("========== A. VERIFY SOURCE ARTIFACTS ==========")
for p in [
    assignment_path, metrics_path, bootstrap_path,
    f1_path, mcc_path, f1_gate_path
]:
    if not p.exists():
        raise FileNotFoundError(p)
    print(p.name, sha256(p))

assign = pd.read_parquet(assignment_path)
old_metrics = pd.read_csv(metrics_path)
old_boot = pd.read_csv(bootstrap_path)
f1_corrected = pd.read_csv(f1_path)
mcc_corrected = pd.read_csv(mcc_path)
step310_gate = pd.read_csv(f1_gate_path)

if len(assign) != 46622 or int(assign["TARGET"].sum()) != 1173:
    raise RuntimeError("Frozen assignment population mismatch.")
if len(old_metrics) != 48 or len(old_boot) != 48:
    raise RuntimeError("Step308 metric table must contain 48 rows.")
if len(f1_corrected) != 12 or len(mcc_corrected) != 12:
    raise RuntimeError("Corrected F1/MCC tables must each contain 12 rows.")

print("\n========== B. TARGETED FIVE-POSITIVE F1 VALIDATION ==========")

small = assign[
    assign["MISSINGNESS_EXTREMENESS_2X2"].astype(str)
    == "MISS_HIGH_EXT_LOW"
].copy()

y = small["TARGET"].to_numpy(dtype=np.int8)
hgb = small["E64R_HGB_PROB"].to_numpy(dtype=np.float64)
m5 = small["E64R_M5_PROB"].to_numpy(dtype=np.float64)

if len(small) != 7647 or int(y.sum()) != 5:
    raise RuntimeError("Expected the 7,647-row, five-positive regime.")

seed_row = f1_corrected[
    (f1_corrected["regime_family"] == "missingness_extremeness_2x2")
    & (f1_corrected["regime"] == "MISS_HIGH_EXT_LOW")
]

if len(seed_row) != 1:
    raise RuntimeError("Cannot identify the original F1 bootstrap seed.")

seed = int(seed_row.iloc[0]["bootstrap_seed"])
rng = np.random.default_rng(seed)
n = len(y)
probability = np.full(n, 1.0/n, dtype=np.float64)

hgb_prediction = (hgb >= 0.125).astype(np.int8)
m5_prediction = (m5 >= 0.855).astype(np.int8)

no_positive = 0
all_positive = 0
replicates = 0
maximum_error = 0.0

for start in range(0, 2000, 50):
    counts = rng.multinomial(
        n, probability, size=min(50, 2000-start)
    )

    sampled_positive = counts @ y
    no_positive += int((sampled_positive == 0).sum())
    all_positive += int((sampled_positive == n).sum())

    for prediction in [hgb_prediction, m5_prediction]:
        tp = counts @ (y * prediction)
        fp = counts @ ((1-y) * prediction)
        fn = counts @ (y * (1-prediction))
        denominator = 2.0*tp + fp + fn

        custom = np.divide(
            2.0*tp,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 0,
        )

        for i in range(len(counts)):
            reference = f1_score(
                y,
                prediction,
                sample_weight=counts[i],
                zero_division=0,
            )
            maximum_error = max(
                maximum_error,
                abs(float(custom[i])-float(reference))
            )
            replicates += 1

print("Regime rows:", n)
print("Regime positives:", int(y.sum()))
print("Bootstrap replicates:", 2000)
print("No-positive resamples:", no_positive)
print("All-positive resamples:", all_positive)
print("Model-wise F1 comparisons:", replicates)
print("Max F1 discrepancy vs sklearn:", maximum_error)

if no_positive == 0:
    raise RuntimeError(
        "This deterministic sample produced no no-positive replicates; "
        "the intended edge-case validation was not exercised."
    )
if maximum_error > 1e-12:
    raise RuntimeError("F1 weighted comparison failed.")
if int(seed_row.iloc[0]["valid_bootstrap"]) != 2000:
    raise RuntimeError("Corrected F1 output does not retain all replicates.")

print("[PASS] Single-class F1 resamples included and match sklearn.")

print("\n========== C. VALIDATE FOUR METRIC STREAMS ==========")

streams = {
    "balanced_accuracy": old_boot[
        old_boot["metric"].astype(str).str.lower()
        == "balanced_accuracy"
    ].copy(),
    "f1": f1_corrected.copy(),
    "mcc": mcc_corrected.copy(),
    "pr_auc": old_boot[
        old_boot["metric"].astype(str).str.lower()
        == "pr_auc"
    ].copy(),
}

for metric, frame in streams.items():
    if len(frame) != 12:
        raise RuntimeError(f"{metric} does not contain 12 regimes.")

    if metric == "f1":
        if not (frame["valid_bootstrap"] == 2000).all():
            raise RuntimeError("F1 replicate count is not 2,000.")
    elif metric in ["mcc", "pr_auc"]:
        if not frame["valid_bootstrap"].between(1989, 2000).all():
            raise RuntimeError(f"Unexpected valid replicates for {metric}.")

    print(
        metric,
        "| rows:", len(frame),
        "| valid replicates:",
        int(frame["valid_bootstrap"].min()),
        "to", int(frame["valid_bootstrap"].max())
    )

print("\n========== D. CONSOLIDATE THE 48 BOOTSTRAP ROWS ==========")

required = [
    "regime_family", "regime", "metric", "n",
    "observed_diff", "bootstrap_mean_diff",
    "ci_lower", "ci_upper", "p_two_sided",
    "valid_bootstrap", "HGB_threshold",
    "M5_threshold", "test_year",
]

base_ba = streams["balanced_accuracy"][required].copy()

def prepare_metric(frame, metric):
    d = frame.copy()
    d["metric"] = metric
    for col in required:
        if col not in d.columns:
            raise RuntimeError(f"Missing {col} from {metric} result.")
    return d[required].copy()

corrected_parts = [
    base_ba,
    prepare_metric(streams["f1"], "f1"),
    prepare_metric(streams["mcc"], "mcc"),
    prepare_metric(streams["pr_auc"], "pr_auc"),
]

final_boot = pd.concat(corrected_parts, ignore_index=True)

final_boot = final_boot.sort_values(
    ["regime_family", "regime", "metric"]
).reset_index(drop=True)

if len(final_boot) != 48:
    raise RuntimeError(f"Consolidated bootstrap rows={len(final_boot)}")

if final_boot.duplicated(
    ["regime_family", "regime", "metric"]
).any():
    raise RuntimeError("Duplicate consolidated bootstrap keys.")

# Verify all point differences against the original Step308 results.
for metric in ["balanced_accuracy", "f1", "mcc", "pr_auc"]:
    old_part = old_boot[
        old_boot["metric"].astype(str).str.lower() == metric
    ].sort_values(["regime_family", "regime"])

    new_part = final_boot[
        final_boot["metric"].astype(str).str.lower() == metric
    ].sort_values(["regime_family", "regime"])

    if len(old_part) != 12 or len(new_part) != 12:
        raise RuntimeError(f"Incorrect row count for {metric}.")

    comp = old_part.merge(
        new_part,
        on=["regime_family", "regime"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )

    if not np.allclose(
        comp["observed_diff_old"],
        comp["observed_diff_new"],
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(f"{metric} point differences changed.")

    print(f"{metric}: all 12 point differences unchanged.")

# Add an add-one Monte Carlo tail estimate so a reported zero is not
# misrepresented as an exactly zero probability.
B = 2000
final_boot["p_two_sided_plus_one"] = np.minimum(
    1.0,
    (final_boot["p_two_sided"] * B + 2.0) / (B + 1.0),
)

print("\nBootstrap p-value note:")
print("Raw zero tail estimates are not literal p=0.")
print("Use p_two_sided_plus_one for Monte Carlo add-one reporting.")
print("Percentile confidence intervals remain the primary adaptation gate.")

print("\n========== E. REBUILD THE FINAL THREE-METRIC GATE ==========")

principal = final_boot[
    final_boot["metric"].astype(str).str.lower().isin(
        ["f1", "mcc", "pr_auc"]
    )
].copy()

gate_rows = []

for (family, regime), group in principal.groupby(
    ["regime_family", "regime"]
):
    checks = {}
    record = {
        "regime_family": family,
        "regime": regime,
    }

    for metric in ["f1", "mcc", "pr_auc"]:
        r = group[
            group["metric"].astype(str).str.lower() == metric
        ]
        if len(r) != 1:
            raise RuntimeError(f"Missing {family}/{regime}/{metric}.")

        row = r.iloc[0]
        record[metric + "_diff"] = float(row["observed_diff"])
        record[metric + "_ci_lower"] = float(row["ci_lower"])
        record[metric + "_ci_upper"] = float(row["ci_upper"])

        checks[metric] = bool(
            row["observed_diff"] > 0
            and row["ci_lower"] > 0
        )

        record[metric.upper() + "_eligible"] = checks[metric]

    record["ALL_THREE_ELIGIBLE"] = all(checks.values())
    gate_rows.append(record)

final_gate = pd.DataFrame(gate_rows)

if len(final_gate) != 12:
    raise RuntimeError("Final gate does not contain 12 regimes.")

eligible = final_gate[final_gate["ALL_THREE_ELIGIBLE"]]

print(final_gate.to_string(index=False))
print("\nFinal adaptation-eligible regimes:", len(eligible))

# Verify the final gate agrees with Step310's already consolidated gate.
left = final_gate.sort_values(["regime_family", "regime"]).reset_index(drop=True)
right = step310_gate.sort_values(["regime_family", "regime"]).reset_index(drop=True)

if len(right) != 12:
    raise RuntimeError("Step310 gate count mismatch.")

for col in ["regime_family", "regime", "ALL_THREE_ELIGIBLE"]:
    if not left[col].equals(right[col]):
        raise RuntimeError(f"Final gate disagrees with Step310 on {col}.")

print("[PASS] Final gate agrees with Step310.")

print("\n========== F. WRITE FINAL LOCKED EVIDENCE ==========")

metrics_out = out / "step237_final_regime_metrics_locked.csv"
bootstrap_out = out / "step237_final_regime_bootstrap_corrected_locked.csv"
gate_out = out / "step237_final_adaptation_gate_locked.csv"
manifest_out = out / "step311_final_evidence_manifest.json"
readme_out = out / "STEP311_EVIDENCE_NOTES.txt"

old_metrics.to_csv(metrics_out, index=False)
final_boot.to_csv(bootstrap_out, index=False)
final_gate.to_csv(gate_out, index=False)

notes = """ACHER-Net Step237 Final Regime Evidence
Evaluation population: 2024 frozen test, N=46,622, positives=1,173.
Comparator: Step151 HGB, threshold 0.125.
Final model: Step237 M5_ACHER_SAGE, threshold 0.855.
Regime cutpoints: original development-only 2020-2022 definitions.

The final bootstrap CSV combines:
- Balanced accuracy: original Step308 implementation; one-class target resamples are invalid.
- F1: Step310 corrected implementation; all resamples are included with zero_division=0.
- MCC: Step309 corrected implementation; zero prediction denominator maps to MCC=0 when both target classes are present; single-class target samples remain invalid.
- PR-AUC: Step308 implementation with zero-weight score groups safely handled; single-class target samples remain invalid.

All corrected bootstrap analyses use 2,000 replicates and the original deterministic regime-specific seeds.
Original scripts, Step308 outputs, Step309 outputs, Step310 outputs, and source Parquet files are preserved.

The three-metric adaptation gate requires positive observed differences and 95% percentile-bootstrap lower bounds above zero for F1, MCC, and PR-AUC simultaneously.
Zero raw bootstrap tail estimates are Monte Carlo boundary values, not literal zero probabilities. Prefer the add-one estimate in p_two_sided_plus_one. Confidence intervals are the primary gate evidence.

This is a supplementary final-Step237 regime evaluation, not a replacement of historical Step151 E6.4R/E6.5 artifacts.
"""

readme_out.write_text(notes, encoding="utf-8")

manifest = {
    "experiment": "Step237_Final_Regime_Evidence_Locked",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_year": 2024,
    "n": len(assign),
    "positives": int(assign["TARGET"].sum()),
    "negative_cases": int((assign["TARGET"] == 0).sum()),
    "HGB_threshold": 0.125,
    "M5_threshold": 0.855,
    "bootstrap_replicates": 2000,
    "base_seed": 20260902,
    "regime_count": 12,
    "metric_comparison_count": len(old_metrics),
    "bootstrap_comparison_count": len(final_boot),
    "corrected_metric_sources": {
        "balanced_accuracy": "Step308 original valid one-class rejection",
        "f1": "Step310 corrected; zero_division=0",
        "mcc": "Step309 corrected; sklearn zero-denominator convention",
        "pr_auc": "Step308 corrected zero-weight score groups",
    },
    "five_positive_regime_validation": {
        "regime": "MISS_HIGH_EXT_LOW",
        "n": len(small),
        "positives": int(y.sum()),
        "no_positive_resamples": no_positive,
        "all_positive_resamples": all_positive,
        "max_weighted_f1_error_vs_sklearn": maximum_error,
    },
    "adaptation_eligible_regimes": int(len(eligible)),
    "decision": "RETAIN_HGB; no conditional regime qualifies for adaptation",
    "raw_zero_p_value_handling": "add-one Monte Carlo reporting column included",
    "inputs": {
        str(p): sha256(p)
        for p in [
            assignment_path, metrics_path, bootstrap_path,
            f1_path, mcc_path, f1_gate_path
        ]
    },
    "outputs": {
        p.name: sha256(p)
        for p in [metrics_out, bootstrap_out, gate_out, readme_out]
    },
}

manifest_out.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("Final metrics:", metrics_out)
print("Final bootstrap:", bootstrap_out)
print("Final gate:", gate_out)
print("Notes:", readme_out)
print("Manifest:", manifest_out)

print("\n========== G. FINAL INTEGRITY ==========")
print("Consolidated rows:", len(final_boot))
print("Balanced accuracy rows:", int((final_boot["metric"] == "balanced_accuracy").sum()))
print("F1 rows:", int((final_boot["metric"] == "f1").sum()))
print("MCC rows:", int((final_boot["metric"] == "mcc").sum()))
print("PR-AUC rows:", int((final_boot["metric"] == "pr_auc").sum()))
print("Eligible adaptation regimes:", len(eligible))
print("Original source outputs preserved.")
print("STEP311_COMPLETE")
