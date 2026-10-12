#!/usr/bin/env python
"""
ACHER-Net — AUTHORITATIVE Q1 SEQUENTIAL PREQUENTIAL CONTINUAL EVALUATION

Purpose
-------
Run the sequential continual-learning experiment on the SAME target used by
Q1: NEXT_YEAR_HIGHER_HAZARD_STATE.

Stages
------
2021 adaptation -> 2022 score -> 2023 outcome
2022 adaptation -> 2023 score -> 2024 outcome
2023 adaptation -> 2024 score -> 2025 outcome

The final stage is the locked frozen 2024->2025 Q1 horizon.

Important
---------
This script intentionally does NOT use EMERGING_HAZARD_NEXT_YEAR.
It does NOT evaluate 2025->2026 because the authoritative Q1 target artifact
contains no validated 2026 higher-hazard-state outcome.
It also does not claim 2026->2027 performance.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

STAGES = [
    (2021, 2022, 2023),
    (2022, 2023, 2024),
    (2023, 2024, 2025),
]
MODEL_NAMES = [
    "M0_FINE_TUNING",
    "M1_EWC",
    "M2_EWC_REPLAY",
    "M3_DRIFT_AWARE",
    "M4_VULNERABILITY_WEIGHTED",
    "M5_ACHER_SAGE",
]
HGB_PARAMS = dict(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=1.0,
)
Q1_MATRIX_DEFAULT = (
    Path("data") / "processed" / "validation" / "acher_net_q1_features"
    / "step134_q1_189_feature_target_matrix.parquet"
)
STEP150_DEFAULT = Path("scripts") / "146_acher_net_q1_step150_leakage_safe_prediction.py"
M5_DEFAULT = (
    Path("data") / "processed" / "validation" / "acher_net_q1_acher_sage"
    / "final_step237_acher_sage_optimization" / "step237_m5_inner_tuning_summary.csv"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, default=Q1_MATRIX_DEFAULT)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--step150", type=Path, default=STEP150_DEFAULT)
    p.add_argument("--m5-summary", type=Path, default=M5_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/validation/acher_net_q1_final/sequential_authoritative_q1"))
    p.add_argument("--seed", type=int, default=20260904)
    return p.parse_args()


def discover_manifest(root: Path) -> Path:
    candidates = []
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        if "108" in name and "contract" in name and "authoritative" in name:
            candidates.append(p)
    if not candidates:
        for p in root.rglob("*.csv"):
            name = p.name.lower()
            if "108" in name and "contract" in name:
                candidates.append(p)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        fail(
            "Could not uniquely discover the authoritative 108-feature manifest. "
            f"Found {len(candidates)} candidates:\n" + "\n".join(map(str, candidates[:20]))
        )
    return candidates[0]


def load_step150(path: Path):
    if not path.exists():
        fail(f"Step-150 not found: {path.resolve()}")
    ns = runpy.run_path(str(path), run_name="achernet_step150")
    required = [
        "HazardNet", "predict", "clone_state", "restore", "train_epochs",
        "fisher_information", "drift_score", "select_threshold", "calculate_metrics",
        "fit_semantic_criticality", "transform_semantic_criticality", "train_adaptive_branch",
        "M5_CANDIDATES", "INITIAL_EPOCHS", "FISHER_SAMPLES", "DEVICE", "SEED", "seed_all",
    ]
    missing = [x for x in required if x not in ns]
    if missing:
        fail(f"Step-150 missing symbols: {missing}")
    return ns


def selected_features(manifest_path: Path, frame: pd.DataFrame) -> list[str]:
    m = pd.read_csv(manifest_path, low_memory=False)
    candidates = [
        "FEATURE", "FEATURE_NAME", "feature", "feature_name", "COLUMN", "COLUMN_NAME",
        "SELECTED_FEATURE", "SELECTED_FEATURE_NAME",
    ]
    col = next((c for c in candidates if c in m.columns), None)
    if col is None and m.shape[1] == 1:
        col = m.columns[0]
    if col is None:
        fail(f"Cannot identify feature column in manifest: {list(m.columns)}")
    vals = m[col].dropna().astype(str).str.strip().tolist()
    out, seen = [], set()
    for x in vals:
        if x not in seen:
            out.append(x)
            seen.add(x)
    if len(out) != 108:
        fail(f"Expected exactly 108 selected features, found {len(out)}")
    missing = [x for x in out if x not in frame.columns]
    if missing:
        fail(f"Manifest features missing from Q1 matrix: {missing[:20]}")
    return out


def selected_m5_config(path: Path, step: dict):
    if not path.exists():
        fail(f"M5 tuning summary not found: {path.resolve()}")
    s = pd.read_csv(path, low_memory=False)
    if "CONFIG_ID" not in s.columns:
        fail(f"CONFIG_ID missing in {path}")
    cid = str(s.iloc[0]["CONFIG_ID"])
    hits = [copy.deepcopy(c) for c in step["M5_CANDIDATES"] if str(c.get("ID")) == cid]
    if len(hits) != 1:
        fail(f"Frozen M5 CONFIG_ID {cid!r} does not map uniquely to Step-150 candidates")
    return hits[0], cid


def idx_year(years: np.ndarray, year: int) -> np.ndarray:
    return np.flatnonzero(years == int(year)).astype(np.int64)


def hgb_stage(x_raw, y, years, adapt_year, score_year, seed, step):
    dev = np.flatnonzero((years >= 2020) & (years <= adapt_year - 1)).astype(np.int64)
    adapt = idx_year(years, adapt_year)
    score = idx_year(years, score_year)
    if min(map(len, (dev, adapt, score))) == 0:
        fail(f"Empty HGB population at adaptation={adapt_year}, score={score_year}")

    imp0 = SimpleImputer(strategy="median")
    xd = imp0.fit_transform(x_raw[dev])
    xa = imp0.transform(x_raw[adapt])
    pre = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=seed + score_year)
    pre.fit(xd, y[dev])
    pa = pre.predict_proba(xa)[:, 1]
    th, _ = step["select_threshold"](y[adapt], pa)
    th = float(th)

    fit = np.concatenate([dev, adapt])
    imp = SimpleImputer(strategy="median")
    xf = imp.fit_transform(x_raw[fit])
    xs = imp.transform(x_raw[score])
    model = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=seed + score_year)
    model.fit(xf, y[fit])
    ps = model.predict_proba(xs)[:, 1]
    met = step["calculate_metrics"](y[score], ps, th, "HGB")
    return met, ps, th, score


def main() -> None:
    a = parse_args()
    root = Path.cwd().resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    matrix = a.matrix.resolve()
    step150 = a.step150.resolve()
    m5_summary = a.m5_summary.resolve()
    manifest = a.manifest.resolve() if a.manifest else discover_manifest(root)

    for path in [matrix, step150, m5_summary, manifest]:
        if not path.exists():
            fail(f"Required file not found: {path}")

    step = load_step150(step150)
    full = pd.read_parquet(matrix, engine="pyarrow")

    required = {"MINE_ID", "AS_OF_YEAR", "NEXT_YEAR_HIGHER_HAZARD_STATE"}
    missing = sorted(required - set(full.columns))
    if missing:
        fail(f"Q1 matrix is missing authoritative target columns: {missing}")

    # Explicitly protect against the known wrong target.
    if "EMERGING_HAZARD_NEXT_YEAR" in full.columns:
        print("[INFO] Wrong-target column EMERGING_HAZARD_NEXT_YEAR is present but will NOT be used.")

    selected = selected_features(manifest, full)
    frame = full[full["AS_OF_YEAR"].between(2020, 2024)].copy()
    frame["AS_OF_YEAR"] = pd.to_numeric(frame["AS_OF_YEAR"], errors="coerce")
    frame["NEXT_YEAR_HIGHER_HAZARD_STATE"] = pd.to_numeric(
        frame["NEXT_YEAR_HIGHER_HAZARD_STATE"], errors="coerce"
    )
    if frame["NEXT_YEAR_HIGHER_HAZARD_STATE"].isna().any():
        fail("Authoritative Q1 target contains missing labels in 2020–2024.")
    frame = frame.sort_values(["AS_OF_YEAR", "MINE_ID"], kind="mergesort").reset_index(drop=True)

    years = frame["AS_OF_YEAR"].astype(int).to_numpy()
    y = frame["NEXT_YEAR_HIGHER_HAZARD_STATE"].astype(np.int8).to_numpy()
    x_raw = frame[selected].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)

    # We only evaluate the three score horizons whose targets are authoritative.
    expected = {
        2022: 1104,
        2023: 1061,
        2024: 1173,
    }
    coverage_rows = []
    for year, expected_positive in expected.items():
        idx = idx_year(years, year)
        observed_positive = int(np.sum(y[idx] == 1))
        coverage_rows.append({
            "AS_OF_YEAR": year,
            "PREDICTION_YEAR": year + 1,
            "ROWS": int(len(idx)),
            "POSITIVES": observed_positive,
            "NEGATIVES": int(len(idx) - observed_positive),
            "EXPECTED_Q1_POSITIVES": expected_positive,
            "TARGET_COUNT_CHECK": bool(observed_positive == expected_positive),
        })
    coverage = pd.DataFrame(coverage_rows)
    if not coverage["TARGET_COUNT_CHECK"].all():
        fail("Step-134 target counts do not match locked Q1 counts.\n" + coverage.to_string(index=False))
    coverage.to_csv(out / "authoritative_target_coverage.csv", index=False)

    # Initial preprocessing uses only 2020, matching the first sequential stage.
    initial = idx_year(years, 2020)
    if len(initial) == 0:
        fail("No 2020 initial development rows.")
    pos = int(np.sum(y[initial] == 1)); neg = int(np.sum(y[initial] == 0))
    if pos == 0:
        fail("2020 initial development contains no positive cases.")
    pos_weight = neg / max(pos, 1)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed_initial = imputer.fit_transform(x_raw[initial])
    scaler.fit(imputed_initial)
    x_all = scaler.transform(imputer.transform(x_raw)).astype(np.float32)

    crit_fit = step["fit_semantic_criticality"](x_raw[initial], selected)
    criticality = step["transform_semantic_criticality"](x_raw, crit_fit).astype(np.float32)

    cfg, cfg_id = selected_m5_config(m5_summary, step)
    common = {
        "H1": int(cfg["H1"]), "H2": int(cfg["H2"]), "DROPOUT": float(cfg["DROPOUT"]),
        "LR": float(cfg["LR"]), "WEIGHT_DECAY": float(cfg["WEIGHT_DECAY"]),
        "REPLAY_RATIO": float(cfg["REPLAY_RATIO"]), "EWC_MIN": float(cfg["EWC_MIN"]),
        "EWC_MAX": float(cfg["EWC_MAX"]), "ADAPTATION_EPOCHS": int(cfg["ADAPTATION_EPOCHS"]),
    }

    step["seed_all"](a.seed)
    base = step["HazardNet"](len(selected), common["H1"], common["H2"], common["DROPOUT"]).to(step["DEVICE"])
    step["train_epochs"](
        base, x_all[initial], y[initial], pos_weight,
        step["INITIAL_EPOCHS"], None, None, None, 0.0,
        common["LR"], common["WEIGHT_DECAY"], "SEQUENTIAL_COMMON_INITIAL",
    )
    init_state = step["clone_state"](base)
    del base
    gc.collect()
    states = {name: copy.deepcopy(init_state) for name in MODEL_NAMES}

    result_rows, prediction_rows, evolution_rows = [], [], []

    for stage_no, (adapt_year, score_year, prediction_year) in enumerate(STAGES, start=1):
        dev = np.flatnonzero((years >= 2020) & (years <= adapt_year - 1)).astype(np.int64)
        adapt = idx_year(years, adapt_year)
        score = idx_year(years, score_year)
        if min(map(len, (dev, adapt, score))) == 0:
            fail(f"Empty sequential stage {adapt_year}->{score_year}")

        drift = float(step["drift_score"](x_all[dev], x_all[adapt]))

        hm, hp, ht, hs = hgb_stage(x_raw, y, years, adapt_year, score_year, a.seed, step)
        result_rows.append({
            "STAGE": stage_no,
            "ADAPTATION_AS_OF_YEAR": adapt_year,
            "SCORE_AS_OF_YEAR": score_year,
            "PREDICTION_YEAR": prediction_year,
            "MODEL": "HGB",
            "THRESHOLD": float(ht),
            "DRIFT_SCORE": drift,
            "N_SCORE": len(hs),
            "N_POSITIVE": int(np.sum(y[hs] == 1)),
            "BALANCED_ACCURACY": hm["BALANCED_ACCURACY"],
            "F1": hm["F1"],
            "MCC": hm["MCC"],
            "PR_AUC": hm["PR_AUC"],
            "ROC_AUC": hm["ROC_AUC"],
        })
        for j, idx in enumerate(hs):
            prediction_rows.append({
                "STAGE": stage_no, "MINE_ID": str(frame.iloc[idx]["MINE_ID"]),
                "AS_OF_YEAR": int(years[idx]), "PREDICTION_YEAR": prediction_year,
                "Y_TRUE": int(y[idx]), "PREDICTION_PROBABILITY": float(hp[j]),
                "PREDICTION": int(hp[j] >= ht), "MODEL": "HGB", "THRESHOLD": float(ht),
            })

        # Prevent a subtle source of accidental leakage: Fisher is sampled only from dev.
        fn = min(int(step["FISHER_SAMPLES"]), len(dev))
        frng = np.random.default_rng(a.seed + stage_no * 9001)
        find = np.sort(frng.choice(dev, size=fn, replace=False)).astype(np.int64) if fn < len(dev) else dev

        for mi, name in enumerate(MODEL_NAMES, start=1):
            pre = step["HazardNet"](len(selected), common["H1"], common["H2"], common["DROPOUT"]).to(step["DEVICE"])
            step["restore"](pre, states[name])
            p_adapt = step["predict"](pre, x_all[adapt])
            th, _ = step["select_threshold"](y[adapt], p_adapt)
            th = float(th)
            fisher = {} if name == "M0_FINE_TUNING" else step["fisher_information"](
                pre, x_all, y, pos_weight, find
            )
            model = step["train_adaptive_branch"](
                name=name,
                initial_state=states[name],
                fisher=fisher,
                X=x_all,
                y=y,
                years=years,
                development_indices=dev,
                adaptation_indices=adapt,
                pos_weight=pos_weight,
                criticality=criticality,
                drift=drift,
                config=common,
                model_i=mi + stage_no * 100,
            )
            ps = step["predict"](model, x_all[score])
            met = step["calculate_metrics"](y[score], ps, th, name)
            result_rows.append({
                "STAGE": stage_no,
                "ADAPTATION_AS_OF_YEAR": adapt_year,
                "SCORE_AS_OF_YEAR": score_year,
                "PREDICTION_YEAR": prediction_year,
                "MODEL": name,
                "THRESHOLD": float(th),
                "DRIFT_SCORE": drift,
                "N_SCORE": len(score),
                "N_POSITIVE": int(np.sum(y[score] == 1)),
                "BALANCED_ACCURACY": met["BALANCED_ACCURACY"],
                "F1": met["F1"],
                "MCC": met["MCC"],
                "PR_AUC": met["PR_AUC"],
                "ROC_AUC": met["ROC_AUC"],
            })
            for j, idx in enumerate(score):
                prediction_rows.append({
                    "STAGE": stage_no, "MINE_ID": str(frame.iloc[idx]["MINE_ID"]),
                    "AS_OF_YEAR": int(years[idx]), "PREDICTION_YEAR": prediction_year,
                    "Y_TRUE": int(y[idx]), "PREDICTION_PROBABILITY": float(ps[j]),
                    "PREDICTION": int(ps[j] >= th), "MODEL": name, "THRESHOLD": float(th),
                })
            states[name] = step["clone_state"](model)
            evolution_rows.append({
                "STAGE": stage_no, "MODEL": name,
                "ADAPTATION_AS_OF_YEAR": adapt_year, "SCORE_AS_OF_YEAR": score_year,
                "PREDICTION_YEAR": prediction_year, "N_DEVELOPMENT": len(dev),
                "N_ADAPTATION": len(adapt), "N_SCORE": len(score), "DRIFT_SCORE": drift,
                "THRESHOLD_SOURCE": f"{adapt_year}_PRE_ADAPTATION",
                "STATE_UPDATED_AFTER_SCORE": True,
            })
            del pre, model, fisher, p_adapt, ps
            gc.collect()
            try:
                if step.get("torch") is not None and step["torch"].cuda.is_available():
                    step["torch"].cuda.empty_cache()
            except Exception:
                pass

    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    evolution = pd.DataFrame(evolution_rows)
    pivot = results.pivot_table(
        index=["STAGE", "SCORE_AS_OF_YEAR", "PREDICTION_YEAR"],
        columns="MODEL",
        values=["BALANCED_ACCURACY", "F1", "MCC", "PR_AUC", "ROC_AUC"],
        aggfunc="first",
    )
    pivot.columns = ["__".join(map(str, c)) for c in pivot.columns.to_flat_index()]
    comparison = pivot.reset_index()
    for metric in ["BALANCED_ACCURACY", "F1", "MCC", "PR_AUC", "ROC_AUC"]:
        a1 = f"{metric}__M5_ACHER_SAGE"; a2 = f"{metric}__HGB"
        if a1 in comparison.columns and a2 in comparison.columns:
            comparison[f"M5_MINUS_HGB__{metric}"] = comparison[a1] - comparison[a2]

    results.to_csv(out / "sequential_stage_metrics.csv", index=False)
    predictions.to_csv(out / "sequential_stage_predictions.csv", index=False)
    evolution.to_csv(out / "sequential_model_evolution_log.csv", index=False)
    comparison.to_csv(out / "sequential_stage_comparison.csv", index=False)

    manifest_data = {
        "experiment": "ACHER-Net authoritative sequential prequential continual evaluation",
        "target": "NEXT_YEAR_HIGHER_HAZARD_STATE",
        "target_source": str(matrix),
        "target_validation": "Step-134 counts match locked Q1 positives for 2022/2023/2024",
        "wrong_target_excluded": "EMERGING_HAZARD_NEXT_YEAR",
        "stages": [
            {"adaptation_as_of_year": a, "score_as_of_year": s, "prediction_year": p}
            for a, s, p in STAGES
        ],
        "feature_manifest": str(manifest),
        "feature_count": 108,
        "initial_preprocessing_fit_year": 2020,
        "step150": str(step150),
        "m5_selected_config_id": cfg_id,
        "thresholding": "stage-specific threshold from adaptation year before adaptation; score-year threshold tuning prohibited",
        "2025_to_2026_status": "NOT_INCLUDED_NO_AUTHORITATIVE_SAME_TARGET_OUTCOME",
        "2026_to_2027_status": "PROSPECTIVE_ONLY_NO_OUTCOME_EVALUATION",
        "seed": a.seed,
        "source_sha256": sha256_file(matrix),
    }
    (out / "sequential_run_manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    print("\nAUTHORITATIVE SEQUENTIAL Q1 COMPLETE")
    print(results.to_string(index=False))
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
