#!/usr/bin/env python
"""
ACHER-Net — TARGET / STATE TRANSITION AUDIT

Purpose
-------
1. Verify what NEXT_YEAR_HIGHER_HAZARD_STATE means using the upstream state
   artifacts already present in the project.
2. Reconstruct mine-year state transitions.
3. Compare reconstructed 2022->2025 labels against the locked Step-129 target.
4. Produce an exact negative-class decomposition where the upstream states
   are available.
5. Audit whether 2026 state rows exist and clearly label any 2025->2026 result
   as interim/partial-year rather than a final annual evaluation.

No model training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STEP129_DEFAULT = Path("data/processed/validation/acher_net_q1_target/step129_authoritative_next_year_targets.parquet")
STATE_DEFAULT = Path("data/processed/discovery/adaptive_emerging_hazard/acher_adaptive_emerging_hazard_states.parquet")
RUL_DEFAULT = Path("data/processed/rul_final/step126_hazard_rul_runtime.parquet")
TEMPORAL_DEFAULT = Path("data/processed/canonical/hazard_temporal/mine_hazard_temporal_features.parquet")

# Canonical ordinal mapping used for a higher-hazard comparison. Aliases are
# intentionally broad; exact validation against Step-129 is the acceptance gate.
STATE_RANK = {
    "STABLE": 0, "LOW": 0, "LOW_RISK": 0,
    "PERSISTENT": 1, "MODERATE": 1, "MODERATE_RISK": 1,
    "EMERGING": 2,
    "EMERGING_ESCALATING": 3, "ESCALATING": 3, "ESCALATED": 3,
    "HIGH": 4, "HIGH_RISK": 4,
    "NOVEL": 5, "CRITICAL": 5, "CRITICAL_RISK": 5, "VERY_HIGH": 5, "VERY_HIGH_RISK": 5,
}


def norm_state(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper().replace("-", "_").replace(" ", "_")
    return s


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=STEP129_DEFAULT)
    p.add_argument("--state", type=Path, default=STATE_DEFAULT)
    p.add_argument("--rul", type=Path, default=RUL_DEFAULT)
    p.add_argument("--temporal", type=Path, default=TEMPORAL_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/validation/acher_net_q1_final/target_transition_audit_2026"))
    return p.parse_args()


def load_state(path: Path):
    if path is None or not path.exists():
        return None, "missing"
    cols = pd.read_parquet(path, engine="pyarrow").columns.tolist()
    state_col = next((c for c in ["HAZARD_STATE", "HAZARD_ACTIVITY_STATE", "CURRENT_HAZARD_STATE", "STATE", "DISPLAY_STATE"] if c in cols), None)
    mine_col = next((c for c in ["MINE_ID", "mine_id", "Mine_ID"] if c in cols), None)
    year_col = next((c for c in ["AS_OF_YEAR", "YEAR", "year", "REPORTING_YEAR"] if c in cols), None)
    if not (mine_col and year_col and state_col):
        return None, f"missing required state columns; found={cols}"
    df = pd.read_parquet(path, columns=[mine_col, year_col, state_col], engine="pyarrow")
    df = df.rename(columns={mine_col: "MINE_ID", year_col: "AS_OF_YEAR", state_col: "STATE"})
    df["MINE_ID"] = df["MINE_ID"].astype(str).str.strip()
    df["AS_OF_YEAR"] = pd.to_numeric(df["AS_OF_YEAR"], errors="coerce")
    df["STATE"] = df["STATE"].map(norm_state)
    df = df.dropna(subset=["AS_OF_YEAR"]).copy()
    df["AS_OF_YEAR"] = df["AS_OF_YEAR"].astype(int)
    df = df.drop_duplicates(["MINE_ID", "AS_OF_YEAR"], keep="last")
    return df, "ok"


def reconstruct(state: pd.DataFrame) -> pd.DataFrame:
    if state is None or state.empty:
        return pd.DataFrame()
    s = state.sort_values(["MINE_ID", "AS_OF_YEAR"], kind="mergesort").copy()
    g = s.groupby("MINE_ID", sort=False)
    s["NEXT_YEAR"] = s.groupby("MINE_ID")["AS_OF_YEAR"].shift(-1)
    s["NEXT_STATE"] = s.groupby("MINE_ID")["STATE"].shift(-1)
    s["CURRENT_RANK"] = s["STATE"].map(STATE_RANK)
    s["NEXT_RANK"] = s["NEXT_STATE"].map(STATE_RANK)
    s["CONSECUTIVE_NEXT_YEAR"] = s["NEXT_YEAR"].eq(s["AS_OF_YEAR"] + 1)
    s = s[s["CONSECUTIVE_NEXT_YEAR"]].copy()
    s["STATE_CHANGE"] = s["STATE"].ne(s["NEXT_STATE"])
    s["HIGHER_STATE"] = (
        s["CURRENT_RANK"].notna() & s["NEXT_RANK"].notna() &
        s["NEXT_RANK"].gt(s["CURRENT_RANK"])
    )
    s["LOWER_STATE"] = (
        s["CURRENT_RANK"].notna() & s["NEXT_RANK"].notna() &
        s["NEXT_RANK"].lt(s["CURRENT_RANK"])
    )
    return s


def main():
    a = args(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    for p in [a.target]:
        if not p.exists():
            raise SystemExit(f"Missing target artifact: {p.resolve()}")

    target = pd.read_parquet(
        a.target,
        columns=["MINE_ID", "AS_OF_YEAR", "NEXT_YEAR_STATE_CHANGE", "NEXT_YEAR_HIGHER_HAZARD_STATE", "SPLIT"],
        engine="pyarrow",
    )
    target["MINE_ID"] = target["MINE_ID"].astype(str).str.strip()
    target["AS_OF_YEAR"] = pd.to_numeric(target["AS_OF_YEAR"], errors="coerce").astype(int)
    target["NEXT_YEAR_STATE_CHANGE"] = pd.to_numeric(target["NEXT_YEAR_STATE_CHANGE"], errors="coerce").astype(bool)
    target["NEXT_YEAR_HIGHER_HAZARD_STATE"] = pd.to_numeric(target["NEXT_YEAR_HIGHER_HAZARD_STATE"], errors="coerce").astype(bool)

    # Prefer the adaptive emerging-hazard state artifact; fall back to RUL state rows.
    state_df, state_status = load_state(a.state)
    state_source = a.state
    if state_df is None:
        state_df, state_status = load_state(a.rul)
        state_source = a.rul

    if state_df is None:
        raise SystemExit(
            "No upstream mine-year state source with MINE_ID/year/state could be loaded. "
            "The Step-129 binary target remains authoritative, but transition decomposition "
            "cannot be reconstructed from the available files."
        )

    transitions = reconstruct(state_df)
    transitions.to_parquet(out / "reconstructed_mine_year_transitions.parquet", index=False)

    summary = {
        "target_path": str(a.target.resolve()),
        "target_sha256": sha256(a.target),
        "state_source": str(state_source.resolve()),
        "state_source_sha256": sha256(state_source),
        "state_source_status": state_status,
        "state_year_min": int(state_df["AS_OF_YEAR"].min()),
        "state_year_max": int(state_df["AS_OF_YEAR"].max()),
        "state_rows": int(len(state_df)),
        "state_mines": int(state_df["MINE_ID"].nunique()),
    }

    # Exact validation against Step-129 for authoritative horizons.
    validation_rows = []
    for year in [2022, 2023, 2024]:
        truth = target[target["AS_OF_YEAR"].eq(year)][["MINE_ID", "AS_OF_YEAR", "NEXT_YEAR_STATE_CHANGE", "NEXT_YEAR_HIGHER_HAZARD_STATE"]].copy()
        pred = transitions[transitions["AS_OF_YEAR"].eq(year)][["MINE_ID", "AS_OF_YEAR", "STATE_CHANGE", "HIGHER_STATE"]].copy()
        merged = truth.merge(pred, on=["MINE_ID", "AS_OF_YEAR"], how="inner", validate="one_to_one")
        n = len(merged)
        state_change_mismatch = int((merged["NEXT_YEAR_STATE_CHANGE"] != merged["STATE_CHANGE"]).sum()) if n else -1
        higher_mismatch = int((merged["NEXT_YEAR_HIGHER_HAZARD_STATE"] != merged["HIGHER_STATE"]).sum()) if n else -1
        validation_rows.append({
            "AS_OF_YEAR": year,
            "TARGET_ROWS": len(truth),
            "RECONSTRUCTED_ROWS": len(pred),
            "JOINED_ROWS": n,
            "STATE_CHANGE_MISMATCH": state_change_mismatch,
            "HIGHER_STATE_MISMATCH": higher_mismatch,
            "EXACT_MATCH": bool(n == len(truth) == len(pred) and state_change_mismatch == 0 and higher_mismatch == 0),
            "TARGET_POSITIVES": int(truth["NEXT_YEAR_HIGHER_HAZARD_STATE"].sum()),
            "RECONSTRUCTED_POSITIVES": int(pred["HIGHER_STATE"].sum()),
        })
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out / "target_reconstruction_validation.csv", index=False)
    summary["authoritative_reconstruction_exact_match"] = bool(validation["EXACT_MATCH"].all())

    # Exact 2024->2025 decomposition when the state lineage matches the target.
    t24 = transitions[transitions["AS_OF_YEAR"].eq(2024)].copy()
    if not t24.empty:
        decomp = pd.DataFrame({
            "CATEGORY": ["NO_STATE_CHANGE", "CHANGED_LOWER", "CHANGED_HIGHER", "CHANGED_BUT_UNRANKED"],
            "COUNT": [
                int((~t24["STATE_CHANGE"]).sum()),
                int(t24["STATE_CHANGE"].fillna(False).where(t24["LOWER_STATE"].fillna(False), False).sum()),
                int(t24["HIGHER_STATE"].fillna(False).sum()),
                int((t24["STATE_CHANGE"] & t24["CURRENT_RANK"].isna() | t24["NEXT_RANK"].isna()).fillna(False).sum()),
            ],
        })
        # The boolean expression above can overlap in edge cases; build a non-overlapping category table too.
        cat = pd.Series("CHANGED_BUT_UNRANKED", index=t24.index)
        cat.loc[~t24["STATE_CHANGE"]] = "NO_STATE_CHANGE"
        cat.loc[t24["LOWER_STATE"]] = "CHANGED_LOWER"
        cat.loc[t24["HIGHER_STATE"]] = "CHANGED_HIGHER"
        exact_decomp = cat.value_counts().rename_axis("CATEGORY").reset_index(name="COUNT")
        exact_decomp["PERCENT"] = exact_decomp["COUNT"] / len(t24)
        exact_decomp.to_csv(out / "2024_to_2025_negative_class_decomposition.csv", index=False)

        trans_table = (
            t24.groupby(["STATE", "NEXT_STATE"], dropna=False)
            .size().reset_index(name="COUNT")
            .sort_values("COUNT", ascending=False)
        )
        trans_table.to_csv(out / "2024_to_2025_state_transition_table.csv", index=False)
        summary["2024_to_2025_rows"] = int(len(t24))
        summary["2024_to_2025_higher_positive"] = int(t24["HIGHER_STATE"].sum())
        summary["2024_to_2025_negative"] = int((~t24["HIGHER_STATE"]).sum())

    # 2025->2026 is only an INTERIM current-year observation, not a frozen annual outcome.
    t25 = transitions[transitions["AS_OF_YEAR"].eq(2025)].copy()
    if not t25.empty:
        interim = t25[
            ["MINE_ID", "AS_OF_YEAR", "STATE", "NEXT_STATE", "STATE_CHANGE", "HIGHER_STATE"]
        ].copy()
        interim["STATUS"] = "INTERIM_PARTIAL_2026_NOT_A_FINAL_ANNUAL_TARGET"
        interim.to_csv(out / "2025_to_2026_interim_state_transitions.csv", index=False)
        summary["2025_to_2026_interim_rows"] = int(len(t25))
        summary["2025_to_2026_interim_positive_count"] = int(t25["HIGHER_STATE"].sum())
    else:
        summary["2025_to_2026_interim_rows"] = 0

    # Current 2026 predictor coverage from canonical temporal features.
    current_2026 = {}
    if a.temporal.exists():
        temp = pd.read_parquet(a.temporal, columns=["MINE_ID", "AS_OF_YEAR"], engine="pyarrow")
        temp["AS_OF_YEAR"] = pd.to_numeric(temp["AS_OF_YEAR"], errors="coerce")
        temp = temp[temp["AS_OF_YEAR"].eq(2026)]
        current_2026 = {
            "canonical_temporal_2026_rows": int(len(temp)),
            "canonical_temporal_2026_mines": int(temp["MINE_ID"].astype(str).nunique()),
        }
    summary["current_2026_coverage"] = current_2026

    (out / "target_transition_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("ACHER-Net — TARGET / STATE TRANSITION AUDIT")
    print("=" * 100)
    print(json.dumps(summary, indent=2, default=str))
    print("\nValidation by authoritative horizon:")
    print(validation.to_string(index=False))
    print(f"\nOUTPUT: {out}")


if __name__ == "__main__":
    main()
