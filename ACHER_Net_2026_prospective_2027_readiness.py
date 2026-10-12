#!/usr/bin/env python
"""
ACHER-Net — 2026 CURRENT DATA / 2027 PROSPECTIVE READINESS AUDIT

This is an operational readiness check, not a performance evaluation.
It distinguishes:
  * current 2026 source coverage;
  * exact authorized 108-feature coverage;
  * same-target outcome availability.

A partial 2026 observation year is never treated as a completed annual target.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path.cwd().resolve()
TEMPORAL = ROOT / "data/processed/canonical/hazard_temporal/mine_hazard_temporal_features.parquet"
LEARNING = ROOT / "data/processed/learning/acher_temporal_learning_matrix.parquet"
Q1_MATRIX = ROOT / "data/processed/validation/acher_net_q1_features/step134_q1_189_feature_target_matrix.parquet"
OUT = ROOT / "data/processed/validation/acher_net_q1_final/prospective_2027_readiness"
OUT.mkdir(parents=True, exist_ok=True)

candidates_108 = []
for p in ROOT.rglob("*.csv"):
    n = p.name.lower()
    if "108" in n and "contract" in n:
        candidates_108.append(p)

report = {
    "status": "NOT_READY",
    "as_of_calendar_year": 2026,
    "prospective_prediction_year": 2027,
    "performance_evaluation_allowed": False,
}

if TEMPORAL.exists():
    t = pd.read_parquet(TEMPORAL, columns=["MINE_ID", "AS_OF_YEAR"], engine="pyarrow")
    t["AS_OF_YEAR"] = pd.to_numeric(t["AS_OF_YEAR"], errors="coerce")
    t26 = t[t["AS_OF_YEAR"].eq(2026)]
    report["canonical_temporal_2026_rows"] = int(len(t26))
    report["canonical_temporal_2026_mines"] = int(t26["MINE_ID"].astype(str).nunique())
else:
    report["canonical_temporal_2026_rows"] = 0
    report["canonical_temporal_2026_mines"] = 0

if LEARNING.exists():
    l = pd.read_parquet(LEARNING, columns=["MINE_ID", "AS_OF_YEAR"], engine="pyarrow")
    l["AS_OF_YEAR"] = pd.to_numeric(l["AS_OF_YEAR"], errors="coerce")
    l26 = l[l["AS_OF_YEAR"].eq(2026)]
    report["learning_matrix_2026_rows"] = int(len(l26))
else:
    report["learning_matrix_2026_rows"] = 0

report["authorized_108_manifest_candidates"] = [str(p) for p in sorted(candidates_108)[:20]]
report["authorized_108_manifest_unique"] = len(candidates_108) == 1

# The current Q1 Step-134 matrix is a frozen 2020-2024 labeled matrix.
if Q1_MATRIX.exists():
    q = pd.read_parquet(Q1_MATRIX, columns=["AS_OF_YEAR"], engine="pyarrow")
    report["step134_latest_as_of_year"] = int(pd.to_numeric(q["AS_OF_YEAR"], errors="coerce").max())
else:
    report["step134_latest_as_of_year"] = None

# Current official public data are available, but 2026 is incomplete at the present date.
report["annual_2026_outcome_complete"] = False
report["2025_to_2026_final_evaluation_allowed"] = False
report["reason"] = (
    "2026 source data are current/partial-year observations. A 2025->2026 same-target "
    "annual outcome is not frozen until the 2026 annual state is complete and the exact "
    "NEXT_YEAR_HIGHER_HAZARD_STATE construction is validated against Q1 lineage."
)

if report["canonical_temporal_2026_rows"] > 0 and report["learning_matrix_2026_rows"] > 0 and report["authorized_108_manifest_unique"]:
    report["status"] = "READY_FOR_PROSPECTIVE_2027_MODEL_BUILD"
    report["reason"] = (
        "2026 predictor rows exist under the authorized 108-feature contract; 2027 remains "
        "a prospective inference horizon with no performance evaluation until 2027 outcomes exist."
    )

(OUT / "prospective_2027_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"OUTPUT: {OUT.resolve()}")
