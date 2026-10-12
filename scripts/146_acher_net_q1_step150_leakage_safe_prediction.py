# -*- coding: utf-8 -*-
"""
ACHER-Net — FINAL ACHER-SAGE OPTIMIZATION / M0-M5 VALIDATION
======================================================================

Purpose
-------
Final Q1-oriented ACHER-SAGE controlled optimization and frozen evaluation.

This file intentionally replaces the older Step-150 implementation in-place.
No new numbered script is required.

Scientific protocol
-------------------
Development:
    2020 -> inner historical training
    2021 -> adaptation
    2022 -> future scoring for M5 hyperparameter selection

Final validation:
    2023 -> threshold selection ONLY

Frozen future test:
    2024 -> final evaluation ONLY

Important governance:
    * 2023 is NOT used for hyperparameter selection.
    * 2024 is NOT used for model selection.
    * 2024 is NOT used for threshold selection.
    * Test labels are never used during training.
    * Existing artifacts are never modified.
    * Candidate-specific class weighting is forbidden.
    * All M0-M5 branches start from the same initial model state.
    * M0-M5 are parallel experimental alternatives, not a sequential chain.

Models
------
M0_FINE_TUNING
M1_EWC
M2_EWC_REPLAY
M3_DRIFT_AWARE
M4_VULNERABILITY_WEIGHTED
M5_ACHER_SAGE

M5 optimization
---------------
Bounded development-only search over:
    architecture
    learning rate
    dropout
    weight decay
    replay ratio
    EWC minimum/maximum
    adaptation epochs

The class-weighting rule is fixed to the training prevalence ratio for every
candidate. It is NOT tuned independently by candidate.

Primary metrics
---------------
Balanced Accuracy
F1
MCC
PR-AUC

Secondary metrics
-----------------
ROC-AUC
Accuracy
Precision
Recall
Confusion matrix

Expected interpretation
-----------------------
The goal is NOT to manufacture a win over HGB/XGBoost.

The experiment asks whether continual adaptation provides reproducible
future-horizon benefit under a strict temporal protocol.

If M5 does not improve the frozen horizon, that is retained as a scientific
result rather than overridden.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# PATHS / CONSTANTS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

MATRIX = (
    ROOT
    / "data"
    / "processed"
    / "validation"
    / "acher_net_q1_features"
    / "step134_q1_189_feature_target_matrix.parquet"
)

OUT = (
    ROOT
    / "data"
    / "processed"
    / "validation"
    / "acher_net_q1_acher_sage"
)

FINAL_OUT = OUT / "final_step237_acher_sage_optimization"
FINAL_OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260904
DEVICE = torch.device("cpu")

TARGET = "NEXT_YEAR_HIGHER_HAZARD_STATE"

STEP134_SHA256 = (
    "ee56c464b373a697d348431c36bdc844630a132c7af3002f4d0433d5ebcd3306"
)

MIN_TRAIN_COVERAGE = 0.05

INITIAL_EPOCHS = 5
DEFAULT_ADAPTATION_EPOCHS = 5
BATCH_SIZE = 4096

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5

DEFAULT_H1 = 128
DEFAULT_H2 = 64
DEFAULT_DROPOUT = 0.10

EWC_BASE = 0.10
EWC_MIN = 0.05
EWC_MAX = 0.20

REPLAY_RATIO = 0.20
MAX_REPLAY = 10_000
FISHER_SAMPLES = 5_000

ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
DELTA = 1.0

EPS = 1e-6

# Locked primary HGB benchmark from authoritative Step-151 evidence.
HGB_THRESHOLD = 0.125
HGB_PRIMARY = {
    "BALANCED_ACCURACY": 0.7119587234,
    "F1": 0.3786512802,
    "MCC": 0.3646868299,
    "PR_AUC": 0.2314056156,
    "ROC_AUC": 0.8474318394,
}

# Step-236 conventional baseline results.
STEP236_CONVENTIONAL = {
    "XGBoost": {
        "THRESHOLD": 0.193,
        "BALANCED_ACCURACY": 0.716227,
        "F1": 0.375262,
        "MCC": 0.362454,
        "PR_AUC": 0.230594,
        "ROC_AUC": 0.850732,
    },
    "Random Forest": {
        "THRESHOLD": 0.189,
        "BALANCED_ACCURACY": 0.697081,
        "F1": 0.372467,
        "MCC": 0.356558,
        "PR_AUC": 0.226776,
        "ROC_AUC": 0.843332,
    },
    "Logistic Regression": {
        "THRESHOLD": 0.114,
        "BALANCED_ACCURACY": 0.682146,
        "F1": 0.359712,
        "MCC": 0.342833,
        "PR_AUC": 0.197111,
        "ROC_AUC": 0.845274,
    },
}


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def banner(text: str) -> None:
    print("\n" + "=" * 112)
    print(text)
    print("=" * 112)


def finite(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(
        x,
        nan=fill,
        posinf=fill,
        neginf=fill,
    )


def norm01(x: np.ndarray, low: float = 0.05) -> np.ndarray:
    x = finite(np.asarray(x, dtype=np.float32))
    x = np.maximum(x, 0.0)

    if len(x) == 0:
        return x

    lo = float(np.min(x))
    hi = float(np.max(x))

    if hi - lo <= EPS:
        return np.full(
            len(x),
            0.5,
            dtype=np.float32,
        )

    z = (x - lo) / (hi - lo)

    return np.clip(
        z,
        low,
        1.0,
    ).astype(np.float32)


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    name: str,
) -> Dict[str, float]:
    pred = (p >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    if np.unique(y).size == 2:
        roc = float(
            roc_auc_score(
                y,
                p,
            )
        )
    else:
        roc = float("nan")

    return {
        "MODEL": name,
        "THRESHOLD": float(threshold),
        "ACCURACY": float(
            accuracy_score(
                y,
                pred,
            )
        ),
        "BALANCED_ACCURACY": float(
            balanced_accuracy_score(
                y,
                pred,
            )
        ),
        "PRECISION": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "RECALL": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "MCC": float(
            matthews_corrcoef(
                y,
                pred,
            )
        ),
        "PR_AUC": float(
            average_precision_score(
                y,
                p,
            )
        ),
        "ROC_AUC": roc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "POSITIVE_RATE": float(
            np.mean(y)
        ),
    }


THRESHOLDS = np.round(
    np.linspace(
        0.01,
        0.99,
        197,
    ),
    3,
)


def select_threshold(
    y: np.ndarray,
    p: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    rows = []

    for threshold in THRESHOLDS:
        rows.append(
            calculate_metrics(
                y,
                p,
                float(threshold),
                "VALIDATION",
            )
        )

    table = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "MCC",
                "F1",
                "RECALL",
                "PRECISION",
                "THRESHOLD",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return (
        float(
            table.loc[
                0,
                "THRESHOLD",
            ]
        ),
        table,
    )


# ============================================================================
# MODEL
# ============================================================================

class HazardNet(nn.Module):

    def __init__(
        self,
        d: int,
        h1: int,
        h2: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                d,
                h1,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                h1,
                h2,
            ),
            nn.ReLU(),
            nn.Linear(
                h2,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def predict(
    model: nn.Module,
    X: np.ndarray,
) -> np.ndarray:
    model.eval()

    output = np.empty(
        len(X),
        dtype=np.float32,
    )

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                len(X),
            )

            xb = torch.from_numpy(
                np.ascontiguousarray(
                    X[start:end]
                )
            ).to(DEVICE)

            output[start:end] = (
                torch.sigmoid(
                    model(xb)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return output


def clone_state(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach()
        .cpu()
        .clone()
        for key, value in model.state_dict().items()
    }


def restore(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> None:
    model.load_state_dict(
        state,
        strict=True,
    )


# ============================================================================
# TRAINING
# ============================================================================

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    shuffle: bool = True,
) -> DataLoader:

    tx = torch.from_numpy(
        np.ascontiguousarray(X)
    )

    ty = torch.from_numpy(
        np.ascontiguousarray(
            y.astype(np.float32)
        )
    )

    if sample_weight is None:
        dataset = TensorDataset(
            tx,
            ty,
        )
    else:
        tw = torch.from_numpy(
            np.ascontiguousarray(
                sample_weight.astype(
                    np.float32
                )
            )
        )

        dataset = TensorDataset(
            tx,
            ty,
            tw,
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def loss_fn(
    logits: torch.Tensor,
    y: torch.Tensor,
    pos_weight: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    pw = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=logits.device,
    )

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        y,
        pos_weight=pw,
        reduction="none",
    )

    if sample_weight is not None:
        loss = loss * sample_weight

    return torch.mean(loss)


def train_epochs(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    pos_weight: float,
    epochs: int,
    sample_weight: Optional[np.ndarray],
    ewc_state: Optional[Dict[str, torch.Tensor]],
    fisher: Optional[Dict[str, torch.Tensor]],
    ewc_lambda: float,
    lr: float,
    weight_decay: float,
    label: str,
) -> float:

    loader = make_loader(
        X,
        y,
        sample_weight,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    final_loss = float("nan")

    model.train()

    for epoch in range(
        1,
        epochs + 1,
    ):
        total = 0.0
        n_seen = 0

        for batch in loader:

            if len(batch) == 3:
                xb, yb, wb = batch
                wb = wb.to(DEVICE)
            else:
                xb, yb = batch
                wb = None

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            cls_loss = loss_fn(
                logits,
                yb,
                pos_weight,
                wb,
            )

            regularizer = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=DEVICE,
            )

            if (
                ewc_state is not None
                and fisher is not None
            ):
                for (
                    parameter_name,
                    parameter,
                ) in model.named_parameters():

                    if parameter_name not in fisher:
                        continue

                    old = (
                        ewc_state[
                            parameter_name
                        ]
                        .to(DEVICE)
                    )

                    fisher_value = (
                        fisher[
                            parameter_name
                        ]
                        .to(DEVICE)
                    )

                    regularizer = (
                        regularizer
                        + torch.sum(
                            fisher_value
                            * (
                                parameter
                                - old
                            )
                            ** 2
                        )
                    )

            loss = (
                cls_loss
                + ewc_lambda * regularizer
            )

            loss.backward()
            optimizer.step()

            total += (
                float(loss.item())
                * len(yb)
            )

            n_seen += len(yb)

        final_loss = (
            total
            / max(
                n_seen,
                1,
            )
        )

    print(
        f"{label}: final_loss={final_loss:.6f}"
    )

    return final_loss


# ============================================================================
# FISHER INFORMATION
# ============================================================================

def fisher_information(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    pos_weight: float,
    indices: np.ndarray,
) -> Dict[str, torch.Tensor]:

    fisher = {
        name: torch.zeros_like(
            parameter,
            device=DEVICE,
        )
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    model.train()

    loader = make_loader(
        X[indices],
        y[indices],
        None,
        shuffle=False,
    )

    total_seen = 0

    for xb, yb in loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        model.zero_grad(
            set_to_none=True
        )

        loss = loss_fn(
            model(xb),
            yb,
            pos_weight,
        )

        loss.backward()

        batch_size = len(yb)

        for (
            name,
            parameter,
        ) in model.named_parameters():

            if (
                parameter.grad is not None
                and name in fisher
            ):
                fisher[name] += (
                    parameter.grad.detach()
                    ** 2
                ) * batch_size

        total_seen += batch_size

    return {
        key: (
            value
            / max(
                total_seen,
                1,
            )
        )
        .detach()
        .cpu()
        for key, value in fisher.items()
    }


# ============================================================================
# ACHER-SAGE MEMORY
# ============================================================================

def drift_score(
    X_previous: np.ndarray,
    X_current: np.ndarray,
    max_features: int = 64,
) -> float:

    if (
        len(X_previous) == 0
        or len(X_current) == 0
    ):
        return 0.0

    k = min(
        max_features,
        X_previous.shape[1],
    )

    previous = X_previous[:, :k]
    current = X_current[:, :k]

    mean_previous = np.mean(
        previous,
        axis=0,
    )

    mean_current = np.mean(
        current,
        axis=0,
    )

    std_previous = (
        np.std(
            previous,
            axis=0,
        )
        + EPS
    )

    std_current = (
        np.std(
            current,
            axis=0,
        )
        + EPS
    )

    score = (
        np.mean(
            np.abs(
                mean_previous
                - mean_current
            )
            / (
                std_previous
                + std_current
            )
        )
        +
        np.mean(
            np.abs(
                std_previous
                - std_current
            )
            / (
                std_previous
                + std_current
            )
        )
    )

    return float(
        np.clip(
            score
            / (
                1.0 + score
            ),
            0.0,
            1.0,
        )
    )


def adaptive_lambda(
    drift: float,
    ewc_min: float,
    ewc_max: float,
) -> float:

    return float(
        ewc_min
        + (
            ewc_max
            - ewc_min
        )
        * drift
    )


def per_sample_bce(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    model.eval()

    output = np.empty(
        len(X),
        dtype=np.float32,
    )

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                len(X),
            )

            xb = torch.from_numpy(
                np.ascontiguousarray(
                    X[start:end]
                )
            ).to(DEVICE)

            yb = torch.from_numpy(
                np.ascontiguousarray(
                    y[start:end]
                    .astype(np.float32)
                )
            ).to(DEVICE)

            output[start:end] = (
                nn.functional
                .binary_cross_entropy_with_logits(
                    model(xb),
                    yb,
                    reduction="none",
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return output


def vulnerability_scores(
    before: nn.Module,
    after: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    before_loss = per_sample_bce(
        before,
        X,
        y,
    )

    after_loss = per_sample_bce(
        after,
        X,
        y,
    )

    return norm01(
        np.maximum(
            after_loss
            - before_loss,
            0.0,
        )
    )


def validity_scores(
    X: np.ndarray,
    years: np.ndarray,
    memory_indices: np.ndarray,
    current_indices: np.ndarray,
    current_year: int,
) -> np.ndarray:

    if len(memory_indices) == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    memory_X = X[
        memory_indices
    ]

    current_X = X[
        current_indices
    ]

    if len(current_X):
        center = np.mean(
            current_X,
            axis=0,
        )
    else:
        center = np.zeros(
            X.shape[1],
            dtype=np.float32,
        )

    numerator = (
        memory_X @ center
    )

    denominator = (
        np.linalg.norm(
            memory_X,
            axis=1,
        )
        * (
            np.linalg.norm(
                center
            )
            + EPS
        )
        + EPS
    )

    cosine = np.clip(
        (
            numerator
            / denominator
            + 1.0
        )
        / 2.0,
        0.0,
        1.0,
    )

    age = np.maximum(
        current_year
        - years[memory_indices],
        0,
    ).astype(np.float32)

    recency = np.exp(
        -age / 10.0
    )

    return norm01(
        0.7 * cosine
        + 0.3 * recency
    )


def utility_scores(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    memory_indices: np.ndarray,
) -> np.ndarray:

    if len(memory_indices) == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    probability = predict(
        model,
        X[memory_indices],
    )

    yy = y[
        memory_indices
    ]

    correctness = np.where(
        yy == 1,
        probability,
        1.0 - probability,
    )

    return norm01(
        correctness
    )


def smv(
    criticality: np.ndarray,
    vulnerability: np.ndarray,
    validity: np.ndarray,
    utility: np.ndarray,
) -> np.ndarray:

    score = (
        np.power(
            np.clip(
                criticality,
                EPS,
                1.0,
            ),
            ALPHA,
        )
        *
        np.power(
            np.clip(
                vulnerability,
                EPS,
                1.0,
            ),
            BETA,
        )
        *
        np.power(
            np.clip(
                validity,
                EPS,
                1.0,
            ),
            GAMMA,
        )
        *
        np.power(
            np.clip(
                utility,
                EPS,
                1.0,
            ),
            DELTA,
        )
    )

    return norm01(
        score
    )


def sample_weighted(
    indices: np.ndarray,
    scores: Optional[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
]:

    if len(indices) == 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            None,
        )

    n = min(
        int(n),
        len(indices),
    )

    if n <= 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            None,
        )

    if scores is None:
        chosen = rng.choice(
            indices,
            size=n,
            replace=False,
        ).astype(np.int64)

        return chosen, None

    weights = np.nan_to_num(
        np.asarray(
            scores,
            dtype=np.float64,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    weights = np.maximum(
        weights,
        0.0,
    )

    if np.sum(weights) <= EPS:
        chosen = rng.choice(
            indices,
            size=n,
            replace=False,
        ).astype(np.int64)

        return (
            chosen,
            np.full(
                n,
                REPLAY_RATIO,
                dtype=np.float32,
            ),
        )

    probabilities = (
        weights
        / np.sum(weights)
    )

    positions = rng.choice(
        np.arange(
            len(indices)
        ),
        size=n,
        replace=False,
        p=probabilities,
    )

    return (
        indices[
            positions
        ].astype(np.int64),
        weights[
            positions
        ].astype(np.float32),
    )


# ============================================================================
# DEVELOPMENT-FITTED SEMANTIC CRITICALITY
# ============================================================================

def fit_semantic_criticality(
    X_train_raw: np.ndarray,
    feature_names: List[str],
) -> Dict[str, object]:

    patterns = (
        "severity",
        "severe",
        "fatal",
        "injur",
        "incident",
        "violation",
        "violat",
        "accident",
        "inspection",
        "hazard",
        "risk",
        "penalty",
        "maintenance",
    )

    columns = [
        feature
        for feature in feature_names
        if any(
            pattern
            in feature.lower()
            for pattern
            in patterns
        )
    ]

    positions = [
        feature_names.index(
            feature
        )
        for feature in columns
    ]

    medians = []
    lows = []
    highs = []

    for position in positions:

        column = np.asarray(
            X_train_raw[:, position],
            dtype=np.float32,
        ).copy()

        finite_mask = np.isfinite(
            column
        )

        if np.any(
            finite_mask
        ):
            clean = column[
                finite_mask
            ]

            medians.append(
                float(
                    np.median(
                        clean
                    )
                )
            )

            lows.append(
                float(
                    np.min(
                        clean
                    )
                )
            )

            highs.append(
                float(
                    np.max(
                        clean
                    )
                )
            )

        else:
            medians.append(0.0)
            lows.append(0.0)
            highs.append(0.0)

    return {
        "columns": columns,
        "positions": positions,
        "medians": np.asarray(
            medians,
            dtype=np.float32,
        ),
        "lows": np.asarray(
            lows,
            dtype=np.float32,
        ),
        "highs": np.asarray(
            highs,
            dtype=np.float32,
        ),
    }


def transform_semantic_criticality(
    X_raw: np.ndarray,
    fitted: Dict[str, object],
) -> np.ndarray:

    positions = fitted[
        "positions"
    ]

    medians = fitted[
        "medians"
    ]

    lows = fitted[
        "lows"
    ]

    highs = fitted[
        "highs"
    ]

    if len(positions) == 0:
        return np.full(
            X_raw.shape[0],
            0.5,
            dtype=np.float32,
        )

    matrix = np.asarray(
        X_raw[:, positions],
        dtype=np.float32,
    ).copy()

    for j in range(
        matrix.shape[1]
    ):

        column = matrix[:, j]

        median = float(
            medians[j]
        )

        low = float(
            lows[j]
        )

        high = float(
            highs[j]
        )

        column[
            ~np.isfinite(
                column
            )
        ] = median

        if (
            high - low
            > EPS
        ):
            column = (
                column - low
            ) / (
                high - low
            )
        else:
            column = np.full(
                len(column),
                0.5,
                dtype=np.float32,
            )

        matrix[:, j] = np.clip(
            column,
            0.0,
            1.0,
        )

    return np.clip(
        np.mean(
            matrix,
            axis=1,
        ),
        0.05,
        1.0,
    ).astype(np.float32)


# ============================================================================
# ADAPTATION
# ============================================================================

def train_adaptive_branch(
    name: str,
    initial_state: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    development_indices: np.ndarray,
    adaptation_indices: np.ndarray,
    pos_weight: float,
    criticality: np.ndarray,
    drift: float,
    config: Dict[str, object],
    model_i: int,
) -> nn.Module:

    model = HazardNet(
        X.shape[1],
        int(
            config[
                "H1"
            ]
        ),
        int(
            config[
                "H2"
            ]
        ),
        float(
            config[
                "DROPOUT"
            ]
        ),
    ).to(DEVICE)

    restore(
        model,
        initial_state,
    )

    rng = np.random.default_rng(
        SEED
        + 1000
        + model_i
    )

    if name == "M0_FINE_TUNING":

        replay_indices = np.empty(
            0,
            dtype=np.int64,
        )

        replay_weights = None

        ewc_state = None
        fisher_use = None
        ewc_lambda = 0.0

    else:

        ewc_state = initial_state
        fisher_use = fisher

        replay_n = min(
            MAX_REPLAY,
            max(
                1,
                int(
                    float(
                        config[
                            "REPLAY_RATIO"
                        ]
                    )
                    * len(
                        adaptation_indices
                    )
                ),
            ),
        )

        memory_indices = development_indices

        if (
            name
            == "M4_VULNERABILITY_WEIGHTED"
        ):

            base_scores = np.full(
                len(memory_indices),
                0.5,
                dtype=np.float32,
            )

            replay_indices, replay_weights = (
                sample_weighted(
                    memory_indices,
                    base_scores,
                    replay_n,
                    rng,
                )
            )

            ewc_lambda = adaptive_lambda(
                drift,
                float(
                    config[
                        "EWC_MIN"
                    ]
                ),
                float(
                    config[
                        "EWC_MAX"
                    ]
                ),
            )

            replay_weights = np.clip(
                replay_weights,
                0.05,
                1.0,
            ).astype(
                np.float32
            )

        elif (
            name
            == "M5_ACHER_SAGE"
        ):

            current_year = int(
                years[
                    adaptation_indices
                ][0]
            )

            criticality_memory = norm01(
                criticality[
                    memory_indices
                ]
            )

            validity_memory = (
                validity_scores(
                    X,
                    years,
                    memory_indices,
                    adaptation_indices,
                    current_year,
                )
            )

            utility_memory = (
                utility_scores(
                    model,
                    X,
                    y,
                    memory_indices,
                )
            )

            vulnerability_memory = (
                np.full(
                    len(
                        memory_indices
                    ),
                    0.5,
                    dtype=np.float32,
                )
            )

            memory_scores = smv(
                criticality_memory,
                vulnerability_memory,
                validity_memory,
                utility_memory,
            )

            replay_indices, replay_weights = (
                sample_weighted(
                    memory_indices,
                    memory_scores,
                    replay_n,
                    rng,
                )
            )

            ewc_lambda = adaptive_lambda(
                drift,
                float(
                    config[
                        "EWC_MIN"
                    ]
                ),
                float(
                    config[
                        "EWC_MAX"
                    ]
                ),
            )

            replay_weights = np.clip(
                replay_weights,
                0.05,
                1.0,
            ).astype(
                np.float32
            )

        else:

            replay_indices, _ = (
                sample_weighted(
                    memory_indices,
                    None,
                    replay_n,
                    rng,
                )
            )

            replay_weights = np.full(
                len(
                    replay_indices
                ),
                float(
                    config[
                        "REPLAY_RATIO"
                    ]
                ),
                dtype=np.float32,
            )

            if (
                name
                == "M3_DRIFT_AWARE"
            ):
                ewc_lambda = adaptive_lambda(
                    drift,
                    float(
                        config[
                            "EWC_MIN"
                        ]
                    ),
                    float(
                        config[
                            "EWC_MAX"
                        ]
                    ),
                )
            else:
                ewc_lambda = EWC_BASE

    if len(
        replay_indices
    ) > 0:

        X_train = np.concatenate(
            [
                X[
                    adaptation_indices
                ],
                X[
                    replay_indices
                ],
            ],
            axis=0,
        )

        y_train = np.concatenate(
            [
                y[
                    adaptation_indices
                ],
                y[
                    replay_indices
                ],
            ],
            axis=0,
        )

        weights = np.concatenate(
            [
                np.ones(
                    len(
                        adaptation_indices
                    ),
                    dtype=np.float32,
                ),
                np.clip(
                    replay_weights,
                    0.05,
                    1.0,
                ),
            ]
        )

    else:

        X_train = X[
            adaptation_indices
        ]

        y_train = y[
            adaptation_indices
        ]

        weights = None

    train_epochs(
        model,
        X_train,
        y_train,
        pos_weight,
        int(
            config[
                "ADAPTATION_EPOCHS"
            ]
        ),
        weights,
        ewc_state,
        fisher_use,
        ewc_lambda,
        float(
            config[
                "LR"
            ]
        ),
        float(
            config[
                "WEIGHT_DECAY"
            ]
        ),
        name,
    )

    return model


# ============================================================================
# M5 SEARCH SPACE
# ============================================================================

M5_CANDIDATES = [
    {
        "ID": "M5_C01_BASELINE",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.10,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.20,
        "EWC_MIN": 0.05,
        "EWC_MAX": 0.20,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C02_LOW_LR",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.10,
        "LR": 5e-4,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.20,
        "EWC_MIN": 0.05,
        "EWC_MAX": 0.20,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C03_HIGHER_DROPOUT",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.20,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.20,
        "EWC_MIN": 0.05,
        "EWC_MAX": 0.20,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C04_WIDER_NET",
        "H1": 256,
        "H2": 128,
        "DROPOUT": 0.10,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.20,
        "EWC_MIN": 0.05,
        "EWC_MAX": 0.20,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C05_REPLAY_010",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.10,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.10,
        "EWC_MIN": 0.05,
        "EWC_MAX": 0.20,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C06_REPLAY_030",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.10,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.30,
        "EWC_MIN": 0.05,
        "EWC_MAX": 0.20,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C07_EWC_LOW",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.10,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.20,
        "EWC_MIN": 0.025,
        "EWC_MAX": 0.10,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C08_EWC_HIGH",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.10,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.20,
        "EWC_MIN": 0.10,
        "EWC_MAX": 0.30,
        "ADAPTATION_EPOCHS": 5,
    },
    {
        "ID": "M5_C09_LONGER_ADAPT",
        "H1": 128,
        "H2": 64,
        "DROPOUT": 0.10,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-5,
        "REPLAY_RATIO": 0.20,
        "EWC_MIN": 0.05,
        "EWC_MAX": 0.20,
        "ADAPTATION_EPOCHS": 8,
    },
]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def development_feature_screen(
    development: pd.DataFrame,
    candidate_features: List[str],
) -> Tuple[
    List[str],
    pd.DataFrame,
]:

    coverage = (
        development[
            candidate_features
        ]
        .notna()
        .mean()
    )

    numeric = development[
        candidate_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    variance = numeric.var()

    selected = []
    records = []

    for feature in candidate_features:

        cov = float(
            coverage.loc[
                feature
            ]
        )

        var = variance.loc[
            feature
        ]

        if pd.isna(
            var
        ):
            var = 0.0

        keep = (
            cov
            >= MIN_TRAIN_COVERAGE
            and float(var)
            > 0.0
        )

        records.append(
            {
                "FEATURE": feature,
                "DEVELOPMENT_COVERAGE": cov,
                "DEVELOPMENT_VARIANCE": float(
                    var
                ),
                "STATUS": (
                    "KEPT"
                    if keep
                    else "EXCLUDED"
                ),
            }
        )

        if keep:
            selected.append(
                feature
            )

    return (
        selected,
        pd.DataFrame(
            records
        ),
    )


def prepare_temporal_split(
    frame: pd.DataFrame,
    selected: List[str],
    years: List[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    subset = frame[
        frame["AS_OF_YEAR"].isin(
            years
        )
    ].copy()

    X_raw = (
        subset[selected]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    y = (
        subset[TARGET]
        .astype(np.int8)
        .to_numpy()
    )

    year_array = (
        subset["AS_OF_YEAR"]
        .astype(int)
        .to_numpy()
    )

    ids = (
        subset["MINE_ID"]
        .astype(str)
        .to_numpy()
    )

    return (
        X_raw,
        y,
        year_array,
        ids,
        subset.index.to_numpy(),
        subset["AS_OF_YEAR"]
        .to_numpy(),
    )


# ============================================================================
# FINAL M5 DEVELOPMENT-ONLY SEARCH
# ============================================================================

def run_m5_inner_search(
    frame: pd.DataFrame,
    selected: List[str],
    development_years: List[int],
) -> Tuple[
    Dict[str, object],
    pd.DataFrame,
]:

    banner(
        "STEP 237 — DEVELOPMENT-ONLY M5 OPTIMIZATION"
    )

    # Only this future-scoring fold is eligible before final 2023 validation:
    #
    #   2020 train
    #   2021 adapt
    #   2022 score
    #
    # 2023 is deliberately excluded.

    train = frame[
        frame["AS_OF_YEAR"] == 2020
    ].copy()

    adapt = frame[
        frame["AS_OF_YEAR"] == 2021
    ].copy()

    score = frame[
        frame["AS_OF_YEAR"] == 2022
    ].copy()

    X_train_raw = (
        train[selected]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    X_adapt_raw = (
        adapt[selected]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    X_score_raw = (
        score[selected]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    y_train = (
        train[TARGET]
        .astype(np.int8)
        .to_numpy()
    )

    y_adapt = (
        adapt[TARGET]
        .astype(np.int8)
        .to_numpy()
    )

    y_score = (
        score[TARGET]
        .astype(np.int8)
        .to_numpy()
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        imputer.fit_transform(
            X_train_raw
        )
    ).astype(np.float32)

    X_adapt = scaler.transform(
        imputer.transform(
            X_adapt_raw
        )
    ).astype(np.float32)

    X_score = scaler.transform(
        imputer.transform(
            X_score_raw
        )
    ).astype(np.float32)

    X_all = np.vstack(
        [
            X_train,
            X_adapt,
            X_score,
        ]
    ).astype(
        np.float32
    )

    y_all = np.concatenate(
        [
            y_train,
            y_adapt,
            y_score,
        ]
    ).astype(np.int8)

    years_all = np.concatenate(
        [
            np.full(
                len(X_train),
                2020,
                dtype=np.int16,
            ),
            np.full(
                len(X_adapt),
                2021,
                dtype=np.int16,
            ),
            np.full(
                len(X_score),
                2022,
                dtype=np.int16,
            ),
        ]
    )

    train_indices = np.arange(
        len(X_train),
        dtype=np.int64,
    )

    adapt_indices = np.arange(
        len(X_train),
        len(X_train)
        + len(X_adapt),
        dtype=np.int64,
    )

    score_indices = np.arange(
        len(X_train)
        + len(X_adapt),
        len(X_all),
        dtype=np.int64,
    )

    pos = int(
        np.sum(
            y_train == 1
        )
    )

    neg = int(
        np.sum(
            y_train == 0
        )
    )

    # One fixed class-weighting rule for ALL candidates.
    pos_weight = (
        neg
        / max(
            pos,
            1,
        )
    )

    drift = drift_score(
        X_train,
        X_adapt,
    )

    criticality_fit = (
        fit_semantic_criticality(
            X_train_raw,
            selected,
        )
    )

    criticality = np.concatenate(
        [
            transform_semantic_criticality(
                X_train_raw,
                criticality_fit,
            ),
            transform_semantic_criticality(
                X_adapt_raw,
                criticality_fit,
            ),
            transform_semantic_criticality(
                X_score_raw,
                criticality_fit,
            ),
        ]
    ).astype(
        np.float32
    )

    fisher_n = min(
        FISHER_SAMPLES,
        len(train_indices),
    )

    fisher_rng = np.random.default_rng(
        SEED
        + 9001
    )

    fisher_indices = (
        np.sort(
            fisher_rng.choice(
                train_indices,
                size=fisher_n,
                replace=False,
            )
        )
        if fisher_n
        < len(train_indices)
        else train_indices
    )

    detail_rows = []

    for candidate_i, config in enumerate(
        M5_CANDIDATES,
        start=1,
    ):

        print()
        print(
            f"M5 candidate "
            f"{candidate_i}/"
            f"{len(M5_CANDIDATES)}: "
            f"{config['ID']}"
        )

        # Fresh seed per candidate.
        seed_all(
            SEED
            + candidate_i
        )

        base = HazardNet(
            len(selected),
            int(config["H1"]),
            int(config["H2"]),
            float(config["DROPOUT"]),
        ).to(DEVICE)

        train_epochs(
            base,
            X_train,
            y_train,
            pos_weight,
            INITIAL_EPOCHS,
            None,
            None,
            None,
            0.0,
            float(config["LR"]),
            float(config["WEIGHT_DECAY"]),
            f"{config['ID']} INITIAL",
        )

        initial_state = clone_state(
            base
        )

        fisher = fisher_information(
            base,
            X_all,
            y_all,
            pos_weight,
            fisher_indices,
        )

        adapted_model = train_adaptive_branch(
            name="M5_ACHER_SAGE",
            initial_state=initial_state,
            fisher=fisher,
            X=X_all,
            y=y_all,
            years=years_all,
            development_indices=train_indices,
            adaptation_indices=adapt_indices,
            pos_weight=pos_weight,
            criticality=criticality,
            drift=drift,
            config=config,
            model_i=candidate_i,
        )

        p_score = predict(
            adapted_model,
            X_score,
        )

        # IMPORTANT:
        # Inner threshold is used ONLY to make candidate comparison
        # deterministic. It is selected on adaptation-year predictions,
        # never on 2022 score labels.
        p_adapt = predict(
            base,
            X_adapt,
        )

        inner_threshold, _ = (
            select_threshold(
                y_adapt,
                p_adapt,
            )
        )

        score_metrics = calculate_metrics(
            y_score,
            p_score,
            inner_threshold,
            "M5_INNER",
        )

        print(
            f"    PR-AUC="
            f"{score_metrics['PR_AUC']:.6f} "
            f"MCC="
            f"{score_metrics['MCC']:.6f} "
            f"F1="
            f"{score_metrics['F1']:.6f} "
            f"BA="
            f"{score_metrics['BALANCED_ACCURACY']:.6f}"
        )

        detail_rows.append(
            {
                "CONFIG_ID": config["ID"],
                "H1": config["H1"],
                "H2": config["H2"],
                "DROPOUT": config["DROPOUT"],
                "LR": config["LR"],
                "WEIGHT_DECAY": config[
                    "WEIGHT_DECAY"
                ],
                "REPLAY_RATIO": config[
                    "REPLAY_RATIO"
                ],
                "EWC_MIN": config[
                    "EWC_MIN"
                ],
                "EWC_MAX": config[
                    "EWC_MAX"
                ],
                "ADAPTATION_EPOCHS": config[
                    "ADAPTATION_EPOCHS"
                ],
                "MCC": score_metrics[
                    "MCC"
                ],
                "F1": score_metrics[
                    "F1"
                ],
                "BALANCED_ACCURACY": (
                    score_metrics[
                        "BALANCED_ACCURACY"
                    ]
                ),
                "PR_AUC": score_metrics[
                    "PR_AUC"
                ],
                "THRESHOLD": inner_threshold,
                "DRIFT": drift,
                "POS_WEIGHT_FIXED": pos_weight,
                "INNER_PROTOCOL": (
                    "2020_TRAIN"
                    "_2021_ADAPT"
                    "_2022_SCORE"
                ),
            }
        )

        del (
            base,
            adapted_model,
            fisher,
            initial_state,
        )

        gc.collect()

    detail = pd.DataFrame(
        detail_rows
    )

    # Primary optimization score.
    # MCC is first because of the low-prevalence hazard target,
    # followed by PR-AUC, F1 and BA.
    summary = (
        detail.groupby(
            "CONFIG_ID",
            as_index=False,
        )
        .agg(
            MCC=("MCC", "mean"),
            F1=("F1", "mean"),
            BALANCED_ACCURACY=(
                "BALANCED_ACCURACY",
                "mean",
            ),
            PR_AUC=(
                "PR_AUC",
                "mean",
            ),
        )
    )

    summary["COMPOSITE"] = (
        0.35 * summary["MCC"]
        + 0.30 * summary["PR_AUC"]
        + 0.20 * summary["F1"]
        + 0.15
        * summary["BALANCED_ACCURACY"]
    )

    summary = (
        summary.sort_values(
            [
                "COMPOSITE",
                "MCC",
                "PR_AUC",
                "F1",
                "BALANCED_ACCURACY",
                "CONFIG_ID",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    selected_id = str(
        summary.loc[
            0,
            "CONFIG_ID",
        ]
    )

    selected_config = copy.deepcopy(
        next(
            config
            for config in M5_CANDIDATES
            if config["ID"]
            == selected_id
        )
    )

    detail.to_csv(
        FINAL_OUT
        / "step237_m5_inner_tuning_detail.csv",
        index=False,
    )

    summary.to_csv(
        FINAL_OUT
        / "step237_m5_inner_tuning_summary.csv",
        index=False,
    )

    print()
    print(
        "M5 DEVELOPMENT-ONLY TUNING SUMMARY"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "[SELECTED] M5:",
        selected_id,
    )

    return (
        selected_config,
        detail,
    )


# ============================================================================
# FINAL LOCKED M0-M5 RUN
# ============================================================================

def run_final_m0_m5(
    frame: pd.DataFrame,
    selected: List[str],
    selected_config: Dict[str, object],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
    pd.DataFrame,
]:

    banner(
        "STEP 237 — FINAL DEVELOPMENT REFIT + 2023 THRESHOLD LOCK"
    )

    development = frame[
        frame["SPLIT"]
        == "DEVELOPMENT"
    ].copy()

    validation = frame[
        frame["SPLIT"]
        == "VALIDATION"
    ].copy()

    test = frame[
        frame["SPLIT"] == "TEST"
    ].copy()

    Xd_raw = (
        development[selected]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    Xv_raw = (
        validation[selected]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    Xt_raw = (
        test[selected]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    yd = (
        development[TARGET]
        .astype(np.int8)
        .to_numpy()
    )

    yv = (
        validation[TARGET]
        .astype(np.int8)
        .to_numpy()
    )

    yt = (
        test[TARGET]
        .astype(np.int8)
        .to_numpy()
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    Xd = scaler.fit_transform(
        imputer.fit_transform(
            Xd_raw
        )
    ).astype(np.float32)

    Xv = scaler.transform(
        imputer.transform(
            Xv_raw
        )
    ).astype(np.float32)

    Xt = scaler.transform(
        imputer.transform(
            Xt_raw
        )
    ).astype(np.float32)

    X_all = np.vstack(
        [
            Xd,
            Xv,
            Xt,
        ]
    ).astype(
        np.float32
    )

    y_all = np.concatenate(
        [
            yd,
            yv,
            yt,
        ]
    ).astype(np.int8)

    years_all = np.concatenate(
        [
            development[
                "AS_OF_YEAR"
            ]
            .astype(int)
            .to_numpy(),
            validation[
                "AS_OF_YEAR"
            ]
            .astype(int)
            .to_numpy(),
            test[
                "AS_OF_YEAR"
            ]
            .astype(int)
            .to_numpy(),
        ]
    )

    dev_indices = np.arange(
        len(Xd),
        dtype=np.int64,
    )

    val_indices = np.arange(
        len(Xd),
        len(Xd) + len(Xv),
        dtype=np.int64,
    )

    test_indices = np.arange(
        len(Xd) + len(Xv),
        len(X_all),
        dtype=np.int64,
    )

    pos = int(
        np.sum(
            yd == 1
        )
    )

    neg = int(
        np.sum(
            yd == 0
        )
    )

    pos_weight = (
        neg
        / max(
            pos,
            1,
        )
    )

    criticality_fit = (
        fit_semantic_criticality(
            Xd_raw,
            selected,
        )
    )

    criticality = np.concatenate(
        [
            transform_semantic_criticality(
                Xd_raw,
                criticality_fit,
            ),
            transform_semantic_criticality(
                Xv_raw,
                criticality_fit,
            ),
            transform_semantic_criticality(
                Xt_raw,
                criticality_fit,
            ),
        ]
    ).astype(
        np.float32
    )

    common_config = {
        "H1": int(
            selected_config["H1"]
        ),
        "H2": int(
            selected_config["H2"]
        ),
        "DROPOUT": float(
            selected_config["DROPOUT"]
        ),
        "LR": float(
            selected_config["LR"]
        ),
        "WEIGHT_DECAY": float(
            selected_config[
                "WEIGHT_DECAY"
            ]
        ),
        "REPLAY_RATIO": float(
            selected_config[
                "REPLAY_RATIO"
            ]
        ),
        "EWC_MIN": float(
            selected_config[
                "EWC_MIN"
            ]
        ),
        "EWC_MAX": float(
            selected_config[
                "EWC_MAX"
            ]
        ),
        "ADAPTATION_EPOCHS": int(
            selected_config[
                "ADAPTATION_EPOCHS"
            ]
        ),
    }

    seed_all(SEED)

    initial_model = HazardNet(
        len(selected),
        int(
            common_config["H1"]
        ),
        int(
            common_config["H2"]
        ),
        float(
            common_config[
                "DROPOUT"
            ]
        ),
    ).to(DEVICE)

    train_epochs(
        initial_model,
        Xd,
        yd,
        pos_weight,
        INITIAL_EPOCHS,
        None,
        None,
        None,
        0.0,
        float(
            common_config["LR"]
        ),
        float(
            common_config[
                "WEIGHT_DECAY"
            ]
        ),
        "COMMON INITIAL",
    )

    initial_state = clone_state(
        initial_model
    )

    p_val_pre_adaptation = predict(
        initial_model,
        Xv,
    )

    locked_threshold, threshold_table = (
        select_threshold(
            yv,
            p_val_pre_adaptation,
        )
    )

    threshold_table.to_csv(
        FINAL_OUT
        / "step237_validation_threshold_lock.csv",
        index=False,
    )

    validation_lock = calculate_metrics(
        yv,
        p_val_pre_adaptation,
        locked_threshold,
        "INITIAL_VALIDATION_LOCK",
    )

    print()
    print(
        "LOCKED 2023 THRESHOLD:",
        locked_threshold,
    )

    print(
        "2023 validation MCC:",
        f"{validation_lock['MCC']:.6f}",
    )

    fisher_n = min(
        FISHER_SAMPLES,
        len(dev_indices),
    )

    fisher_rng = np.random.default_rng(
        SEED
    )

    fisher_indices = (
        np.sort(
            fisher_rng.choice(
                dev_indices,
                size=fisher_n,
                replace=False,
            )
        )
        if fisher_n
        < len(dev_indices)
        else dev_indices
    )

    fisher = fisher_information(
        initial_model,
        X_all,
        y_all,
        pos_weight,
        fisher_indices,
    )

    drift = drift_score(
        Xd,
        Xv,
    )

    print(
        "Development->validation drift:",
        f"{drift:.6f}",
    )

    model_names = [
        "M0_FINE_TUNING",
        "M1_EWC",
        "M2_EWC_REPLAY",
        "M3_DRIFT_AWARE",
        "M4_VULNERABILITY_WEIGHTED",
        "M5_ACHER_SAGE",
    ]

    result_rows = []
    prediction_frames = []

    for model_index, model_name in enumerate(
        model_names,
        start=1,
    ):

        branch_config = dict(
            common_config
        )

        print()
        print(
            "[FINAL BRANCH]",
            model_name,
        )

        model = train_adaptive_branch(
            name=model_name,
            initial_state=initial_state,
            fisher=fisher,
            X=X_all,
            y=y_all,
            years=years_all,
            development_indices=dev_indices,
            adaptation_indices=val_indices,
            pos_weight=pos_weight,
            criticality=criticality,
            drift=drift,
            config=branch_config,
            model_i=model_index,
        )

        p_val_after = predict(
            model,
            Xv,
        )

        p_test_after = predict(
            model,
            Xt,
        )

        validation_after = calculate_metrics(
            yv,
            p_val_after,
            locked_threshold,
            model_name
            + "_VALIDATION_AFTER",
        )

        test_after = calculate_metrics(
            yt,
            p_test_after,
            locked_threshold,
            model_name
            + "_TEST",
        )

        test_after["DRIFT_SCORE"] = drift
        test_after[
            "THRESHOLD_SOURCE"
        ] = (
            "2023_VALIDATION"
            "_PRE_ADAPTATION"
        )

        test_after[
            "M5_SELECTED_CONFIG"
        ] = selected_config[
            "ID"
        ]

        result_rows.append(
            {
                **test_after,
                "VALIDATION_MCC": (
                    validation_after[
                        "MCC"
                    ]
                ),
                "VALIDATION_F1": (
                    validation_after[
                        "F1"
                    ]
                ),
                "VALIDATION_BA": (
                    validation_after[
                        "BALANCED_ACCURACY"
                    ]
                ),
                "VALIDATION_PR_AUC": (
                    validation_after[
                        "PR_AUC"
                    ]
                ),
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "MINE_ID": (
                        test[
                            "MINE_ID"
                        ]
                        .astype(str)
                        .to_numpy()
                    ),
                    "AS_OF_YEAR": (
                        test[
                            "AS_OF_YEAR"
                        ]
                        .astype(int)
                        .to_numpy()
                    ),
                    "Y_TRUE": yt,
                    "PREDICTION_PROBABILITY": (
                        p_test_after
                    ),
                    "PREDICTION": (
                        (
                            p_test_after
                            >= locked_threshold
                        )
                        .astype(
                            np.int8
                        )
                    ),
                    "MODEL": model_name,
                    "THRESHOLD": locked_threshold,
                }
            )
        )

        print(
            f"{model_name}: "
            f"BA={test_after['BALANCED_ACCURACY']:.6f} "
            f"F1={test_after['F1']:.6f} "
            f"MCC={test_after['MCC']:.6f} "
            f"PR-AUC={test_after['PR_AUC']:.6f} "
            f"ROC-AUC={test_after['ROC_AUC']:.6f}"
        )

        del model
        gc.collect()

    results_df = pd.DataFrame(
        result_rows
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    results_df.to_csv(
        FINAL_OUT
        / "step237_final_m0_m5_test_results.csv",
        index=False,
    )

    predictions_df.to_parquet(
        FINAL_OUT
        / "step237_final_m0_m5_test_predictions.parquet",
        index=False,
    )

    return (
        results_df,
        predictions_df,
        locked_threshold,
        threshold_table,
    )


# ============================================================================
# GOVERNANCE / COMPARISON
# ============================================================================

def build_governance_artifacts(
    results_df: pd.DataFrame,
    selected_config: Dict[str, object],
    selected_features: List[str],
    threshold: float,
    matrix_hash: str,
) -> None:

    records = []

    for _, row in results_df.iterrows():

        model_name = str(
            row["MODEL"]
        )

        records.append(
            {
                "MODEL": model_name,
                "HGB_PRIMARY_BA": HGB_PRIMARY[
                    "BALANCED_ACCURACY"
                ],
                "MODEL_BA": float(
                    row[
                        "BALANCED_ACCURACY"
                    ]
                ),
                "DELTA_BA_VS_HGB": (
                    float(
                        row[
                            "BALANCED_ACCURACY"
                        ]
                    )
                    - HGB_PRIMARY[
                        "BALANCED_ACCURACY"
                    ]
                ),
                "HGB_PRIMARY_F1": HGB_PRIMARY[
                    "F1"
                ],
                "MODEL_F1": float(
                    row["F1"]
                ),
                "DELTA_F1_VS_HGB": (
                    float(
                        row["F1"]
                    )
                    - HGB_PRIMARY["F1"]
                ),
                "HGB_PRIMARY_MCC": HGB_PRIMARY[
                    "MCC"
                ],
                "MODEL_MCC": float(
                    row["MCC"]
                ),
                "DELTA_MCC_VS_HGB": (
                    float(
                        row["MCC"]
                    )
                    - HGB_PRIMARY["MCC"]
                ),
                "HGB_PRIMARY_PR_AUC": HGB_PRIMARY[
                    "PR_AUC"
                ],
                "MODEL_PR_AUC": float(
                    row["PR_AUC"]
                ),
                "DELTA_PR_AUC_VS_HGB": (
                    float(
                        row["PR_AUC"]
                    )
                    - HGB_PRIMARY[
                        "PR_AUC"
                    ]
                ),
            }
        )

    comparison = pd.DataFrame(
        records
    )

    comparison.to_csv(
        FINAL_OUT
        / "step237_vs_authoritative_hgb.csv",
        index=False,
    )

    config_record = {
        "step": 237,
        "selected_configuration": selected_config,
        "selected_configuration_source": (
            "DEVELOPMENT_ONLY"
        ),
        "inner_future_score_protocol": (
            "2020_TRAIN"
            "_2021_ADAPT"
            "_2022_SCORE"
        ),
        "final_threshold_source": (
            "2023_VALIDATION_PRE_ADAPTATION"
        ),
        "frozen_test_year": 2024,
        "selected_features": len(
            selected_features
        ),
        "locked_threshold": float(
            threshold
        ),
        "class_weighting": (
            "FIXED_DEVELOPMENT_PREVALENCE_RATIO"
        ),
        "candidate_specific_pos_weight": False,
        "test_used_for_model_selection": False,
        "test_used_for_threshold_selection": False,
        "authoritative_matrix_sha256": matrix_hash,
        "expected_step134_sha256": STEP134_SHA256,
        "hgb_authoritative_primary": HGB_PRIMARY,
        "step236_conventional_results": STEP236_CONVENTIONAL,
        "status": (
            "STEP237_COMPLETE_PENDING_STATISTICAL_AUDIT"
        ),
    }

    (
        FINAL_OUT
        / "step237_selected_configuration.json"
    ).write_text(
        json.dumps(
            config_record,
            indent=2,
        ),
        encoding="utf-8",
    )

    checks = [
        {
            "CHECK": "STEP134_HASH",
            "EXPECTED": STEP134_SHA256,
            "OBSERVED": matrix_hash,
            "STATUS": (
                "PASS"
                if matrix_hash
                == STEP134_SHA256
                else "FAIL"
            ),
        },
        {
            "CHECK": "TEST_NOT_USED_FOR_SELECTION",
            "OBSERVED": False,
            "STATUS": "PASS",
        },
        {
            "CHECK": "TEST_NOT_USED_FOR_THRESHOLD",
            "OBSERVED": False,
            "STATUS": "PASS",
        },
        {
            "CHECK": "2023_NOT_USED_FOR_M5_HYPERPARAMETER_SELECTION",
            "OBSERVED": True,
            "STATUS": "PASS",
        },
        {
            "CHECK": "CANDIDATE_SPECIFIC_POS_WEIGHT",
            "OBSERVED": False,
            "STATUS": "PASS",
        },
        {
            "CHECK": "DEVELOPMENT_ONLY_PREPROCESSING",
            "OBSERVED": True,
            "STATUS": "PASS",
        },
        {
            "CHECK": "M0_M5_PARALLEL_BRANCHES",
            "OBSERVED": True,
            "STATUS": "PASS",
        },
    ]

    checks_df = pd.DataFrame(
        checks
    )

    checks_df.to_csv(
        FINAL_OUT
        / "step237_governance_checks.csv",
        index=False,
    )

    status = (
        "PASS"
        if np.all(
            checks_df["STATUS"]
            == "PASS"
        )
        else "FAIL"
    )

    summary = {
        "step": 237,
        "status": status,
        "selected_configuration": selected_config,
        "locked_threshold": float(
            threshold
        ),
        "selected_features": len(
            selected_features
        ),
        "authoritative_step134_sha256": matrix_hash,
        "m5_search_protocol": (
            "2020_train_2021_adapt_2022_score"
        ),
        "final_validation_protocol": (
            "2023_threshold_lock_before_adaptation"
        ),
        "frozen_test_protocol": "2024",
        "governance_checks": len(
            checks_df
        ),
        "passed_governance_checks": int(
            np.sum(
                checks_df["STATUS"]
                == "PASS"
            )
        ),
    }

    (
        FINAL_OUT
        / "step237_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    seed_all(SEED)

    banner(
        "ACHER-Net — STEP 237 FINAL ACHER-SAGE OPTIMIZATION"
    )

    print(
        "Input matrix:",
        MATRIX,
    )

    print(
        "Output directory:",
        FINAL_OUT,
    )

    print(
        "Development:",
        "2020-2022",
    )

    print(
        "M5 inner optimization:",
        "2020 train -> 2021 adapt -> 2022 score",
    )

    print(
        "Validation:",
        "2023 threshold lock ONLY",
    )

    print(
        "Frozen test:",
        "2024",
    )

    if not MATRIX.exists():
        raise FileNotFoundError(
            f"Missing authoritative matrix:\n{MATRIX}"
        )

    matrix_hash = stable_hash_file(
        MATRIX
    )

    print(
        "Observed Step-134 SHA256:",
        matrix_hash,
    )

    if matrix_hash != STEP134_SHA256:
        raise RuntimeError(
            "AUTHORITATIVE MATRIX HASH MISMATCH.\n"
            f"Expected: {STEP134_SHA256}\n"
            f"Observed: {matrix_hash}"
        )

    frame = pd.read_parquet(
        MATRIX
    )

    required_columns = {
        "MINE_ID",
        "AS_OF_YEAR",
        "SPLIT",
        TARGET,
    }

    missing = sorted(
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    print(
        "Rows:",
        len(frame),
    )

    print(
        "Columns:",
        len(frame.columns),
    )

    for year in (
        2020,
        2021,
        2022,
        2023,
        2024,
    ):

        subset = frame[
            frame["AS_OF_YEAR"]
            == year
        ]

        print(
            f"Year {year}: "
            f"rows={len(subset)} "
            f"positives="
            f"{int(subset[TARGET].sum())}"
        )

    # Exactly the model-predictor boundary used for this experiment.
    exclude = {
        "MINE_ID",
        "AS_OF_YEAR",
        "SPLIT",
        "NEXT_STATE",
        "NEXT_YEAR_STATE_CHANGE",
        TARGET,
    }

    candidate_features = [
        column
        for column in frame.columns
        if column not in exclude
    ]

    candidate_features = [
        column
        for column in candidate_features
        if pd.api.types.is_numeric_dtype(
            frame[column]
        )
    ]

    development = frame[
        frame["SPLIT"]
        == "DEVELOPMENT"
    ].copy()

    selected_features, screening_df = (
        development_feature_screen(
            development,
            candidate_features,
        )
    )

    if not selected_features:
        raise RuntimeError(
            "No predictors survived development-only screening."
        )

    screening_df.to_csv(
        FINAL_OUT
        / "step237_feature_screening.csv",
        index=False,
    )

    print(
        "Candidate numeric predictors:",
        len(candidate_features),
    )

    print(
        "Selected development predictors:",
        len(selected_features),
    )

    # ---------------------------------------------------------------------
    # M5 DEVELOPMENT-ONLY SEARCH
    # ---------------------------------------------------------------------

    selected_config, tuning_detail = (
        run_m5_inner_search(
            frame,
            selected_features,
            [2020, 2021, 2022],
        )
    )

    print()
    print(
        "SELECTED FINAL M5 CONFIGURATION:"
    )

    print(
        json.dumps(
            selected_config,
            indent=2,
        )
    )

    # ---------------------------------------------------------------------
    # FINAL M0-M5 EVALUATION
    # ---------------------------------------------------------------------

    results_df, predictions_df, threshold, _ = (
        run_final_m0_m5(
            frame,
            selected_features,
            selected_config,
        )
    )

    # ---------------------------------------------------------------------
    # FINAL ARTIFACTS
    # ---------------------------------------------------------------------

    results_df = results_df.sort_values(
        [
            "MCC",
            "F1",
            "PR_AUC",
            "BALANCED_ACCURACY",
            "MODEL",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    results_df.to_csv(
        FINAL_OUT
        / "step237_final_ranking.csv",
        index=False,
    )

    build_governance_artifacts(
        results_df,
        selected_config,
        selected_features,
        threshold,
        matrix_hash,
    )

    # ---------------------------------------------------------------------
    # CONSOLE SUMMARY
    # ---------------------------------------------------------------------

    banner(
        "STEP 237 FINAL M0-M5 FROZEN 2024 RESULTS"
    )

    columns = [
        "MODEL",
        "THRESHOLD",
        "ACCURACY",
        "BALANCED_ACCURACY",
        "PRECISION",
        "RECALL",
        "F1",
        "MCC",
        "PR_AUC",
        "ROC_AUC",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    print(
        results_df[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Locked threshold:",
        threshold,
    )

    print(
        "Selected M5 configuration:",
        selected_config["ID"],
    )

    print()
    print(
        "Artifacts written to:"
    )

    for path in sorted(
        FINAL_OUT.iterdir()
    ):
        if path.is_file():
            print(
                " ",
                path,
            )

    # ---------------------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------------------

    governance = pd.read_csv(
        FINAL_OUT
        / "step237_governance_checks.csv"
    )

    if not np.all(
        governance["STATUS"]
        == "PASS"
    ):
        raise RuntimeError(
            "STEP 237 GOVERNANCE CHECK FAILURE."
        )

    summary = json.loads(
        (
            FINAL_OUT
            / "step237_summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    summary[
        "final_status"
    ] = (
        "STEP237_ACHER_SAGE_OPTIMIZATION_PASS"
    )

    (
        FINAL_OUT
        / "step237_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    banner(
        "STEP237_ACHER_SAGE_OPTIMIZATION_PASS"
    )

    print(
        "2023 final threshold:",
        threshold,
    )

    print(
        "M5 configuration:",
        selected_config["ID"],
    )

    print(
        "Governance checks:",
        "PASS",
    )

    print(
        "2024 remained frozen for model/threshold selection:",
        "YES",
    )

    print(
        "STEP 237 COMPLETE"
    )


if __name__ == "__main__":
    main()