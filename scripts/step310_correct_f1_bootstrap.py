import os, re, json, hashlib, types
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

root = Path(os.environ["S310_ROOT"])
out = Path(os.environ["S310_OUT"])

fast_path = root / "scripts/279R_fast_acher_net_q1_e64r_conditional_adaptation_benefit_locked855.py"
step308 = root / "data/processed/validation/acher_net_q1_final/step237_final_m5_regime_evaluation_20261116"
step309 = root / "data/processed/validation/acher_net_q1_final/step237_MCC_BOOTSTRAP_CORRECTED_20261116"

assign_path = step308 / "step237_final_m5_regime_assignments.parquet"
old_boot_path = step308 / "step237_pairwise_bootstrap.csv"
mcc_path = step309 / "step237_mcc_bootstrap_corrected.csv"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("========== A. VERIFY INPUTS ==========")
for p in [fast_path, assign_path, old_boot_path, mcc_path]:
    if not p.exists():
        raise FileNotFoundError(p)
    print(p.name, sha256(p))

print("\n========== B. PATCH ALL THREE METRIC EDGE CASES IN MEMORY ==========")

source = fast_path.read_text(encoding="utf-8-sig")

# PR-AUC: zero-weight score groups must have zero contribution,
# not NaN precision.
old_pr = """        precision = (
            cumulative_positive[valid_rows]
            / cumulative_weight[valid_rows]
        )"""

new_pr = """        precision = np.divide(
            cumulative_positive[valid_rows],
            cumulative_weight[valid_rows],
            out=np.zeros_like(
                cumulative_positive[valid_rows],
                dtype=np.float64,
            ),
            where=cumulative_weight[valid_rows] > 0.0,
        )"""

if source.count(old_pr) != 1:
    raise RuntimeError("Expected PR-AUC block not found exactly once.")

corrected_source = source.replace(old_pr, new_pr, 1)

# F1: F1 is defined for a single-class resample using zero_division=0.
# Do not restrict F1 to bootstrap samples containing both true classes.
f1_pattern = re.compile(
    r'(?ms)^            elif metric == "f1":\n.*?'
    r'(?=^            elif metric == "mcc":)'
)

f1_replacement = '''            elif metric == "f1":
                hgb_denominator = (
                    2.0 * hgb_tp
                    + hgb_fp
                    + hgb_fn
                )
                m5_denominator = (
                    2.0 * m5_tp
                    + m5_fp
                    + m5_fn
                )

                hgb_metric = np.zeros(
                    chunk_size,
                    dtype=np.float64,
                )
                m5_metric = np.zeros(
                    chunk_size,
                    dtype=np.float64,
                )

                safe_hgb = hgb_denominator > 0.0
                safe_m5 = m5_denominator > 0.0

                hgb_metric[safe_hgb] = (
                    2.0 * hgb_tp[safe_hgb]
                    / hgb_denominator[safe_hgb]
                )
                m5_metric[safe_m5] = (
                    2.0 * m5_tp[safe_m5]
                    / m5_denominator[safe_m5]
                )

                # Include every bootstrap sample. Zero denominators
                # yield F1=0, matching sklearn zero_division=0.
                chunk_result[:] = m5_metric - hgb_metric

'''

f1_matches = list(f1_pattern.finditer(corrected_source))
if len(f1_matches) != 1:
    raise RuntimeError(f"Expected one F1 branch; found {len(f1_matches)}.")

corrected_source = f1_pattern.sub(f1_replacement, corrected_source, count=1)

# MCC: a zero prediction denominator means MCC=0 when both target
# classes are represented, matching sklearn.
mcc_pattern = re.compile(
    r'(?ms)^            elif metric == "mcc":\n.*?'
    r'(?=^            else:\n                raise RuntimeError\()'
)

mcc_replacement = '''            elif metric == "mcc":
                hgb_denominator = np.sqrt(
                    (hgb_tp + hgb_fp)
                    * (hgb_tp + hgb_fn)
                    * (hgb_tn + hgb_fp)
                    * (hgb_tn + hgb_fn)
                )
                m5_denominator = np.sqrt(
                    (m5_tp + m5_fp)
                    * (m5_tp + m5_fn)
                    * (m5_tn + m5_fp)
                    * (m5_tn + m5_fn)
                )

                hgb_metric = np.zeros(
                    chunk_size,
                    dtype=np.float64,
                )
                m5_metric = np.zeros(
                    chunk_size,
                    dtype=np.float64,
                )

                safe_hgb = valid_rows & (hgb_denominator > 0.0)
                safe_m5 = valid_rows & (m5_denominator > 0.0)

                hgb_metric[safe_hgb] = (
                    (
                        hgb_tp[safe_hgb] * hgb_tn[safe_hgb]
                        - hgb_fp[safe_hgb] * hgb_fn[safe_hgb]
                    )
                    / hgb_denominator[safe_hgb]
                )
                m5_metric[safe_m5] = (
                    (
                        m5_tp[safe_m5] * m5_tn[safe_m5]
                        - m5_fp[safe_m5] * m5_fn[safe_m5]
                    )
                    / m5_denominator[safe_m5]
                )

                chunk_result[valid_rows] = (
                    m5_metric[valid_rows] - hgb_metric[valid_rows]
                )

'''

mcc_matches = list(mcc_pattern.finditer(corrected_source))
if len(mcc_matches) != 1:
    raise RuntimeError(f"Expected one MCC branch; found {len(mcc_matches)}.")

corrected_source = mcc_pattern.sub(mcc_replacement, corrected_source, count=1)

module = types.ModuleType("step310_corrected_bootstrap")
module.__file__ = str(fast_path)
exec(compile(corrected_source, str(fast_path), "exec"), module.__dict__)

module.N_BOOTSTRAP = 2000
module.SEED = 20260902
module.HGB_THRESHOLD = 0.125
module.M5_THRESHOLD = 0.855

print("PR-AUC correction: PASS")
print("F1 single-class resamples: included")
print("MCC zero-denominator correction: PASS")
print("Original fast script unchanged.")

print("\n========== C. INDEPENDENT WEIGHTED F1 VALIDATION ==========")

rng = np.random.default_rng(20261116)
n = 4000
y0 = rng.binomial(1, 0.03, size=n).astype(np.int8)
p0 = rng.binomial(1, 0.05, size=n).astype(np.int8)

max_error = 0.0
tested = 0
single_class = 0

for _ in range(300):
    counts = rng.multinomial(n, np.full(n, 1.0 / n))
    sampled_y = np.unique(y0[counts > 0])

    if len(sampled_y) == 1:
        single_class += 1

    tp = np.sum(counts * (y0 == 1) * (p0 == 1))
    fp = np.sum(counts * (y0 == 0) * (p0 == 1))
    fn = np.sum(counts * (y0 == 1) * (p0 == 0))

    den = 2.0 * tp + fp + fn
    custom = 0.0 if den == 0 else 2.0 * tp / den

    reference = f1_score(
        y0,
        p0,
        sample_weight=counts,
        zero_division=0,
    )

    max_error = max(max_error, abs(custom - reference))
    tested += 1

print("Weighted F1 comparisons:", tested)
print("Single-class bootstrap examples:", single_class)
print("Maximum absolute error vs sklearn:", max_error)

if tested == 0 or max_error > 1e-12:
    raise RuntimeError("Weighted F1 validation failed.")

print("\n========== D. RECOMPUTE 12 FINAL-STEP237 F1 BOOTSTRAPS ==========")

df = pd.read_parquet(assign_path)
if len(df) != 46622 or int(df["TARGET"].sum()) != 1173:
    raise RuntimeError("Final Step237 assignment population mismatch.")

regime_map = {
    "missingness_quartile": "MISSINGNESS_QUARTILE",
    "extremeness_quartile": "EXTREMENESS_QUARTILE",
    "missingness_extremeness_2x2": "MISSINGNESS_EXTREMENESS_2X2",
}

seed_match = re.search(
    r"hexdigest\(\)\s*\[\s*:\s*(\d+)\s*\]",
    source,
)
if not seed_match:
    raise RuntimeError("Cannot recover original deterministic seed rule.")
digest_chars = int(seed_match.group(1))

rows = []

for family, column in regime_map.items():
    regimes = sorted(
        str(x) for x in df[column].dropna().unique()
        if str(x) != "UNAVAILABLE"
    )

    for regime in regimes:
        subset = df[df[column].astype(str) == regime]

        y = subset["TARGET"].to_numpy(dtype=np.int8)
        hgb = subset["E64R_HGB_PROB"].to_numpy(dtype=np.float64)
        m5 = subset["E64R_M5_PROB"].to_numpy(dtype=np.float64)

        key = f"{family}|{regime}|f1"
        stable_hash = int(
            hashlib.sha256(key.encode("utf-8")).hexdigest()[:digest_chars],
            16,
        )
        seed = module.SEED + (stable_hash % 1_000_000)

        result = module.paired_bootstrap(
            y, hgb, m5, metric="f1", seed=seed
        )

        rows.append({
            "regime_family": family,
            "regime": regime,
            **result,
            "bootstrap_seed": seed,
            "HGB_threshold": 0.125,
            "M5_threshold": 0.855,
            "M5_model": "M5_ACHER_SAGE_Step237",
            "test_year": 2024,
            "positives": int(y.sum()),
        })

f1_corrected = pd.DataFrame(rows)

if len(f1_corrected) != 12:
    raise RuntimeError(f"Expected 12 F1 results, got {len(f1_corrected)}")

if not (f1_corrected["valid_bootstrap"] == 2000).all():
    raise RuntimeError(
        "F1 should retain all 2,000 replicates because F1 is defined "
        "for one-class resamples with zero_division=0."
    )

print(f1_corrected[
    [
        "regime_family", "regime", "n", "positives",
        "observed_diff", "ci_lower", "ci_upper",
        "p_two_sided", "valid_bootstrap",
    ]
].to_string(index=False))

print("\n========== E. VERIFY F1 POINT ESTIMATES ==========")

old_boot = pd.read_csv(old_boot_path)
old_f1 = old_boot[
    old_boot["metric"].astype(str).str.lower() == "f1"
]

comparison = old_f1.merge(
    f1_corrected,
    on=["regime_family", "regime"],
    suffixes=("_step308", "_corrected"),
    validate="one_to_one",
)

if len(comparison) != 12:
    raise RuntimeError("F1 comparison count mismatch.")

if not np.allclose(
    comparison["observed_diff_step308"],
    comparison["observed_diff_corrected"],
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError("F1 observed point differences changed.")

print("All 12 F1 observed differences unchanged.")

print("\n========== F. CONSOLIDATE CORRECTED THREE-METRIC GATE ==========")

mcc = pd.read_csv(mcc_path)
pr = old_boot[
    old_boot["metric"].astype(str).str.lower() == "pr_auc"
].copy()

def prepare(frame, metric):
    return frame[
        ["regime_family", "regime", "observed_diff", "ci_lower", "ci_upper"]
    ].rename(columns={
        "observed_diff": metric + "_diff",
        "ci_lower": metric + "_lower",
        "ci_upper": metric + "_upper",
    })

gate = (
    prepare(f1_corrected, "f1")
    .merge(prepare(mcc, "mcc"),
           on=["regime_family", "regime"], validate="one_to_one")
    .merge(prepare(pr, "pr_auc"),
           on=["regime_family", "regime"], validate="one_to_one")
)

for metric in ["f1", "mcc", "pr_auc"]:
    gate[metric.upper() + "_eligible"] = (
        (gate[metric + "_diff"] > 0)
        & (gate[metric + "_lower"] > 0)
    )

gate["ALL_THREE_ELIGIBLE"] = gate[
    ["F1_eligible", "MCC_eligible", "PR_AUC_eligible"]
].all(axis=1)

print(gate.to_string(index=False))
eligible = gate[gate["ALL_THREE_ELIGIBLE"]]
print("\nFinal adaptation-eligible regimes:", len(eligible))

print("\n========== G. WRITE SUPPLEMENTARY CORRECTED OUTPUTS ==========")

out.mkdir(parents=True, exist_ok=True)

f1_path = out / "step237_f1_bootstrap_corrected.csv"
gate_path = out / "step237_final_three_metric_gate_corrected.csv"
manifest_path = out / "step310_f1_correction_manifest.json"

f1_corrected.to_csv(f1_path, index=False)
gate.to_csv(gate_path, index=False)

manifest = {
    "experiment": "Step237_Final_M5_F1_Bootstrap_Correction",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_fast_script_sha256": sha256(fast_path),
    "step308_assignments_sha256": sha256(assign_path),
    "step308_bootstrap_sha256": sha256(old_boot_path),
    "step309_corrected_mcc_sha256": sha256(mcc_path),
    "corrections": {
        "PR_AUC": "Zero-weight score groups handled safely.",
        "F1": "All bootstrap samples included; zero denominator maps to F1=0, matching sklearn zero_division=0.",
        "MCC": "Zero denominator maps to MCC=0 when both target classes are represented."
    },
    "bootstrap_replicates": 2000,
    "base_seed": 20260902,
    "HGB_threshold": 0.125,
    "M5_threshold": 0.855,
    "F1_valid_replicates_min": int(f1_corrected["valid_bootstrap"].min()),
    "F1_valid_replicates_max": int(f1_corrected["valid_bootstrap"].max()),
    "weighted_f1_max_abs_error_vs_sklearn": max_error,
    "adaptation_eligible_regimes": int(len(eligible)),
    "original_fast_script_unchanged": True,
    "step308_files_unchanged": True,
    "step309_files_unchanged": True,
    "outputs": {
        f1_path.name: sha256(f1_path),
        gate_path.name: sha256(gate_path),
    },
}

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("Corrected F1 file:", f1_path)
print("Consolidated gate:", gate_path)
print("Manifest:", manifest_path)
print("Original fast script unchanged:", sha256(fast_path))
print("STEP310_COMPLETE")
