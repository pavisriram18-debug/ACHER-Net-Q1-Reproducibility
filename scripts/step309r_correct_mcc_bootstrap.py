import os, re, json, hashlib, types
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

root = Path(os.environ["S309_ROOT"])

fast_path = root / "scripts/279R_fast_acher_net_q1_e64r_conditional_adaptation_benefit_locked855.py"
step308 = root / "data/processed/validation/acher_net_q1_final/step237_final_m5_regime_evaluation_20261116"
assign_path = step308 / "step237_final_m5_regime_assignments.parquet"
bootstrap_path = step308 / "step237_pairwise_bootstrap.csv"
out = root / "data/processed/validation/acher_net_q1_final/step237_MCC_BOOTSTRAP_CORRECTED_20261116"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("========== A. INPUT HASHES ==========")
for p in [fast_path, assign_path, bootstrap_path]:
    if not p.exists():
        raise FileNotFoundError(p)
    print(p.name, sha256(p))

source = fast_path.read_text(encoding="utf-8-sig")

print("\n========== B. APPLY BOTH CORRECTIONS IN MEMORY ==========")

# Correction 1: prevent zero-weight score groups from causing NaN
# precision values and invalidating otherwise valid PR-AUC replicates.
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
    raise RuntimeError("Original PR-AUC block was not found exactly once.")

corrected_source = source.replace(old_pr, new_pr, 1)

# Correction 2: match sklearn MCC=0 for a zero denominator when
# both target classes occur in the bootstrap sample.
mcc_pattern = re.compile(
    r'(?ms)^            elif metric == "mcc":\n.*?'
    r'(?=^            else:\n                raise RuntimeError\()'
)

replacement = '''            elif metric == "mcc":
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

                # Match sklearn: MCC is zero when its denominator
                # is zero, provided both target classes are present.
                hgb_metric = np.zeros(chunk_size, dtype=np.float64)
                m5_metric = np.zeros(chunk_size, dtype=np.float64)

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

matches = list(mcc_pattern.finditer(corrected_source))
if len(matches) != 1:
    raise RuntimeError(f"Expected one MCC branch; found {len(matches)}.")

corrected_source = mcc_pattern.sub(replacement, corrected_source, count=1)

if new_pr not in corrected_source or replacement.strip() not in corrected_source:
    raise RuntimeError("One or both corrections failed verification.")

module = types.ModuleType("step309r_corrected_bootstrap")
module.__file__ = str(fast_path)
exec(compile(corrected_source, str(fast_path), "exec"), module.__dict__)

module.N_BOOTSTRAP = 2000
module.SEED = 20260902
module.HGB_THRESHOLD = 0.125
module.M5_THRESHOLD = 0.855

print("PR-AUC zero-weight correction: PASS")
print("MCC zero-denominator correction: PASS")
print("Original fast script remains unchanged.")

print("\n========== C. INDEPENDENT WEIGHTED MCC TEST ==========")
rng = np.random.default_rng(20261116)
n = 3000
y0 = rng.binomial(1, 0.08, size=n).astype(np.int8)
p0 = rng.binomial(1, 0.12, size=n).astype(np.int8)
max_error = 0.0
tested = 0

for _ in range(200):
    counts = rng.multinomial(n, np.full(n, 1/n))
    if len(np.unique(y0[counts > 0])) < 2:
        continue

    tp = np.sum(counts * (y0 == 1) * (p0 == 1))
    fn = np.sum(counts * (y0 == 1) * (p0 == 0))
    fp = np.sum(counts * (y0 == 0) * (p0 == 1))
    tn = np.sum(counts * (y0 == 0) * (p0 == 0))

    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    custom = 0.0 if den == 0 else (tp*tn-fp*fn)/den
    reference = matthews_corrcoef(y0, p0, sample_weight=counts)

    max_error = max(max_error, abs(custom-reference))
    tested += 1

print("Weighted MCC comparisons:", tested)
print("Maximum absolute error:", max_error)
if tested == 0 or max_error > 1e-12:
    raise RuntimeError("Independent MCC validation failed.")

print("\n========== D. RECOMPUTE 12 FINAL-STEP237 MCC BOOTSTRAPS ==========")

df = pd.read_parquet(assign_path)
old_boot = pd.read_csv(bootstrap_path)

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
    for regime in sorted(
        str(x) for x in df[column].dropna().unique()
        if str(x) != "UNAVAILABLE"
    ):
        s = df[df[column].astype(str) == regime]
        y = s["TARGET"].to_numpy(dtype=np.int8)
        hgb = s["E64R_HGB_PROB"].to_numpy(dtype=np.float64)
        m5 = s["E64R_M5_PROB"].to_numpy(dtype=np.float64)

        key = f"{family}|{regime}|mcc"
        stable_hash = int(
            hashlib.sha256(key.encode("utf-8")).hexdigest()[:digest_chars],
            16,
        )
        seed = module.SEED + (stable_hash % 1_000_000)

        result = module.paired_bootstrap(
            y, hgb, m5, metric="mcc", seed=seed
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

corrected = pd.DataFrame(rows)
if len(corrected) != 12:
    raise RuntimeError(f"Expected 12 MCC results; got {len(corrected)}.")

print(corrected[
    ["regime_family", "regime", "n", "positives",
     "observed_diff", "ci_lower", "ci_upper",
     "p_two_sided", "valid_bootstrap"]
].to_string(index=False))

print("\nValid-replicate range:",
      int(corrected["valid_bootstrap"].min()),
      "to", int(corrected["valid_bootstrap"].max()))

print("\n========== E. VERIFY POINT ESTIMATES ==========")
old_mcc = old_boot[
    old_boot["metric"].astype(str).str.lower() == "mcc"
]

check = old_mcc.merge(
    corrected,
    on=["regime_family", "regime"],
    suffixes=("_old", "_new"),
    validate="one_to_one",
)

if len(check) != 12:
    raise RuntimeError("MCC point-estimate comparison incomplete.")

if not np.allclose(
    check["observed_diff_old"],
    check["observed_diff_new"],
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError("MCC observed point differences changed.")

print("All 12 observed MCC point differences unchanged.")

print("\n========== F. REBUILD FINAL THREE-METRIC GATE ==========")

f1 = old_boot[
    old_boot["metric"].astype(str).str.lower() == "f1"
][["regime_family", "regime", "observed_diff", "ci_lower"]].copy()

pr = old_boot[
    old_boot["metric"].astype(str).str.lower() == "pr_auc"
][["regime_family", "regime", "observed_diff", "ci_lower"]].copy()

mcc_gate = corrected[
    ["regime_family", "regime", "observed_diff", "ci_lower"]
].copy()

def rename_metric(frame, metric):
    return frame.rename(columns={
        "observed_diff": metric + "_diff",
        "ci_lower": metric + "_lower",
    })

gate = (
    rename_metric(f1, "f1")
    .merge(
        rename_metric(mcc_gate, "mcc"),
        on=["regime_family", "regime"],
        validate="one_to_one",
    )
    .merge(
        rename_metric(pr, "pr_auc"),
        on=["regime_family", "regime"],
        validate="one_to_one",
    )
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
print("Final eligible regimes:", len(eligible))

print("\n========== G. WRITE SEPARATE CORRECTED OUTPUTS ==========")
out.mkdir(parents=True, exist_ok=True)

mcc_path = out / "step237_mcc_bootstrap_corrected.csv"
gate_path = out / "step237_final_three_metric_gate_corrected.csv"
manifest_path = out / "step309r_correction_manifest.json"

corrected.to_csv(mcc_path, index=False)
gate.to_csv(gate_path, index=False)

manifest = {
    "experiment": "Step237_Final_M5_MCC_Bootstrap_Correction",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_fast_script_sha256": sha256(fast_path),
    "step308_assignments_sha256": sha256(assign_path),
    "step308_bootstrap_sha256": sha256(bootstrap_path),
    "PR_AUC_correction": "Zero-weight cumulative score groups handled safely.",
    "MCC_correction": "Zero MCC denominator maps to 0 when both target classes are present, matching sklearn.",
    "bootstrap_replicates": 2000,
    "base_seed": 20260902,
    "HGB_threshold": 0.125,
    "M5_threshold": 0.855,
    "weighted_mcc_max_abs_error_vs_sklearn": max_error,
    "mcc_valid_min": int(corrected["valid_bootstrap"].min()),
    "mcc_valid_max": int(corrected["valid_bootstrap"].max()),
    "adaptation_eligible_regimes": int(len(eligible)),
    "original_source_unchanged": True,
    "step308_files_unchanged": True,
    "outputs": {
        mcc_path.name: sha256(mcc_path),
        gate_path.name: sha256(gate_path),
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Corrected MCC file:", mcc_path)
print("Final gate file:", gate_path)
print("Manifest:", manifest_path)
print("Original source SHA256:", sha256(fast_path))
print("STEP309R_COMPLETE")
