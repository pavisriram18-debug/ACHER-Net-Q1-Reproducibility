# -*- coding: utf-8 -*-

"""
ACHER-Net Q1
E6.4R â€” Conditional Adaptation Benefit Analysis
================================================

Purpose
-------
Test whether ACHER-SAGE (M5) provides context-specific benefit over
the authoritative HGB benchmark under different reliability regimes.

AUTHORITATIVE INPUTS
--------------------
Step-134:
    data/processed/validation/acher_net_q1_features/
    step134_q1_189_feature_target_matrix.parquet

Step-151:
    data/processed/validation/acher_net_q1_statistical/
    step151_paired_hgb_m5_test_predictions.parquet

Step-169R:
    data/processed/validation/acher_net_q1_final/
    step169R_corrected_missingness_provenance/
    step169R_authoritative_selected_feature_missingness.csv

SCIENTIFIC SAFEGUARDS
---------------------
* No model retraining.
* No threshold tuning.
* No test-derived threshold selection.
* No test labels used to define reliability cutpoints.
* Existing ACHER-Net artifacts are not modified.
* Existing E5 artifacts are not modified.
* Existing manuscript is not modified.
* Step-230 reconstructed HGB predictions are not used.
* Step-233 reconstructed HGB predictions are not used.

CRITICAL ROW-ALIGNMENT RULE
----------------------------
The authoritative Step-151 prediction artifact may contain repeated
mine-year keys. These are NOT duplicates to be removed.

Alignment is performed using:

    MINE_ID
    AS_OF_YEAR
    Y_TRUE
    OCCURRENCE

where OCCURRENCE is a deterministic cumcount within each exact
mine/year/target key.

This follows the already validated Step-225 methodology.

TEMPORAL PROTOCOL
------------------
Development : 2020-2022
Validation  : 2023
Frozen test : 2024 -> 2025

Frozen test:
    46,622 rows
    1,173 positives

FEATURE CONTRACT
----------------
189 candidate predictors
108 selected predictors

REGIME VARIABLES
----------------
1. Row-wise missingness over 108 selected predictors.

2. Covariate extremeness:
       empirical percentile q within development
       tail exposure = 1 - 2 * min(q, 1-q)

   Row extremeness is the mean exposure across observed selected
   predictors.

REGIME CUTPOINTS
----------------
Computed using development years ONLY.

Evaluated regimes:
    A. Missingness quartiles
    B. Extremeness quartiles
    C. Missingness x Extremeness 2x2

MODELS
------
HGB threshold = 0.125
M5  threshold = 0.83

PRIMARY COMPARISON
------------------
M5 - HGB

Metrics:
    Balanced Accuracy
    F1
    MCC
    PR-AUC

Inference:
    Paired bootstrap
    2,000 replicates
    Seed 20260902

OUTPUT
------
data/processed/validation/acher_net_q1_final/
stepE64R_conditional_adaptation_benefit/

    e64r_regime_metrics.csv
    e64r_pairwise_bootstrap.csv
    e64r_regime_definition.csv
    e64r_feature_extremeness_summary.csv
    e64r_test_regime_assignments.csv
    e64r_summary.json
    e64r_run_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)


# ============================================================
# 1. AUTHORITATIVE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROC = (
    ROOT
    / "data"
    / "processed"
)

VAL = (
    PROC
    / "validation"
)

FINAL = (
    VAL
    / "acher_net_q1_final"
)

MATRIX = (
    VAL
    / "acher_net_q1_features"
    / "step134_q1_189_feature_target_matrix.parquet"
)

PAIRED = (
    VAL
    / "acher_net_q1_statistical"
    / "step151_paired_hgb_m5_test_predictions.parquet"
)

FEATURE_MANIFEST = (
    FINAL
    / "step169R_corrected_missingness_provenance"
    / "step169R_authoritative_selected_feature_missingness.csv"
)

OUT = (
    FINAL
    / "stepE64R_conditional_adaptation_benefit"
)


# ============================================================
# 2. SCIENTIFIC CONSTANTS
# ============================================================

SEED = 20260902

N_BOOTSTRAP = 2000

HGB_THRESHOLD = 0.125

M5_THRESHOLD = 0.855

DEVELOPMENT_YEARS = (
    2020,
    2021,
    2022,
)

VALIDATION_YEAR = 2023

TEST_YEAR = 2024

EXPECTED_MATRIX_ROWS = 233128

EXPECTED_MATRIX_COLUMNS = 195

EXPECTED_CANDIDATES = 189

EXPECTED_SELECTED = 108

EXPECTED_TEST_ROWS = 46622

EXPECTED_TEST_POSITIVES = 1173


# ============================================================
# 3. DISPLAY
# ============================================================

def banner(
    text: str,
) -> None:

    print()

    print(
        "=" * 108
    )

    print(
        text
    )

    print(
        "=" * 108
    )


# ============================================================
# 4. FILE HASH
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# 5. COLUMN RESOLUTION
# ============================================================

def resolve_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:

    exact = {
        str(column):
            str(column)
        for column
        in df.columns
    }

    for candidate in candidates:

        if candidate in exact:
            return exact[
                candidate
            ]

    lowered = {
        str(column)
        .strip()
        .lower():
            str(column)
        for column
        in df.columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in lowered:
            return lowered[
                key
            ]

    return None


def require_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    description: str,
) -> str:

    column = resolve_column(
        df,
        candidates,
    )

    if column is None:

        raise RuntimeError(
            f"Could not resolve {description}.\n"
            f"Candidates={list(candidates)}\n"
            f"Available={list(df.columns)}"
        )

    return column


# ============================================================
# 6. NORMALIZATION
# ============================================================

def normalize_mine_id(
    series: pd.Series,
) -> pd.Series:

    result = (
        series
        .astype(str)
        .str.strip()
    )

    result = (
        result
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    result = result.replace(
        {
            "nan": "",
            "NaN": "",
            "None": "",
        }
    )

    return result


def normalize_target(
    series: pd.Series,
    description: str,
) -> np.ndarray:

    result = pd.to_numeric(
        series,
        errors="coerce",
    )

    if result.isna().any():

        raise RuntimeError(
            f"{description} contains "
            "NaN/non-numeric values."
        )

    unique = set(
        result.astype(
            int
        ).unique()
    )

    if not unique.issubset(
        {0, 1}
    ):

        raise RuntimeError(
            f"{description} is not binary 0/1.\n"
            f"Observed={sorted(unique)}"
        )

    return (
        result
        .astype(
            np.int8
        )
        .to_numpy()
    )


def normalize_probability(
    series: pd.Series,
    description: str,
) -> np.ndarray:

    result = pd.to_numeric(
        series,
        errors="coerce",
    )

    if result.isna().any():

        raise RuntimeError(
            f"{description} contains NaN values."
        )

    values = result.to_numpy(
        dtype=float
    )

    if np.isinf(
        values
    ).any():

        raise RuntimeError(
            f"{description} contains infinite values."
        )

    if (
        values.min() < 0.0
        or values.max() > 1.0
    ):

        raise RuntimeError(
            f"{description} is outside [0,1]. "
            f"Range={values.min()}..{values.max()}"
        )

    return values


# ============================================================
# 7. LOAD 108-FEATURE MANIFEST
# ============================================================

def load_features() -> List[str]:

    if not FEATURE_MANIFEST.exists():

        raise FileNotFoundError(
            "Authoritative Step-169R feature manifest not found:\n"
            f"{FEATURE_MANIFEST}"
        )

    manifest = pd.read_csv(
        FEATURE_MANIFEST
    )

    feature_column = resolve_column(
        manifest,
        [
            "FEATURE",
            "feature",
            "feature_name",
            "FEATURE_NAME",
            "COLUMN",
            "COLUMN_NAME",
        ],
    )

    if feature_column is None:

        raise RuntimeError(
            "No feature-name column found in "
            "Step-169R manifest."
        )

    features = (
        manifest[
            feature_column
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    features = list(
        dict.fromkeys(
            features
        )
    )

    if len(features) != EXPECTED_SELECTED:

        raise RuntimeError(
            "Authoritative selected-feature count mismatch.\n"
            f"Expected={EXPECTED_SELECTED}\n"
            f"Observed={len(features)}"
        )

    return features


# ============================================================
# 8. LOAD STEP-134
# ============================================================

def load_matrix(
    features: Sequence[str],
) -> pd.DataFrame:

    if not MATRIX.exists():

        raise FileNotFoundError(
            f"Missing authoritative Step-134 matrix:\n"
            f"{MATRIX}"
        )

    matrix = pd.read_parquet(
        MATRIX,
        engine="pyarrow",
    )

    if len(matrix) != EXPECTED_MATRIX_ROWS:

        raise RuntimeError(
            "Step-134 row count mismatch.\n"
            f"Expected={EXPECTED_MATRIX_ROWS:,}\n"
            f"Observed={len(matrix):,}"
        )

    if matrix.shape[1] != EXPECTED_MATRIX_COLUMNS:

        raise RuntimeError(
            "Step-134 column count mismatch.\n"
            f"Expected={EXPECTED_MATRIX_COLUMNS:,}\n"
            f"Observed={matrix.shape[1]:,}"
        )

    mine_column = require_column(
        matrix,
        [
            "MINE_ID",
            "mine_id",
        ],
        "MINE_ID",
    )

    year_column = require_column(
        matrix,
        [
            "AS_OF_YEAR",
            "YEAR",
            "year",
        ],
        "AS_OF_YEAR",
    )

    split_column = require_column(
        matrix,
        [
            "SPLIT",
            "split",
        ],
        "SPLIT",
    )

    target_column = require_column(
        matrix,
        [
            "NEXT_YEAR_HIGHER_HAZARD_STATE",
            "EMERGING_HAZARD_NEXT_YEAR",
            "TARGET",
        ],
        "target",
    )

    missing_features = [
        feature
        for feature
        in features
        if feature
        not in matrix.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Selected features absent from Step-134:\n"
            + "\n".join(
                missing_features
            )
        )

    matrix[
        "E64R_MINE_ID"
    ] = normalize_mine_id(
        matrix[
            mine_column
        ]
    )

    matrix[
        "E64R_YEAR"
    ] = pd.to_numeric(
        matrix[
            year_column
        ],
        errors="coerce",
    )

    if matrix[
        "E64R_YEAR"
    ].isna().any():

        raise RuntimeError(
            "Invalid year values in Step-134."
        )

    matrix[
        "E64R_YEAR"
    ] = (
        matrix[
            "E64R_YEAR"
        ]
        .astype(
            np.int16
        )
    )

    matrix[
        "E64R_SPLIT"
    ] = (
        matrix[
            split_column
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    matrix[
        "E64R_TARGET"
    ] = normalize_target(
        matrix[
            target_column
        ],
        target_column,
    )

    expected_years = {
        2020,
        2021,
        2022,
        2023,
        2024,
    }

    observed_years = set(
        matrix[
            "E64R_YEAR"
        ].unique()
    )

    if observed_years != expected_years:

        raise RuntimeError(
            "Unexpected Step-134 years.\n"
            f"Expected={sorted(expected_years)}\n"
            f"Observed={sorted(observed_years)}"
        )

    return matrix


# ============================================================
# 9. VALIDATE TEMPORAL POPULATION
# ============================================================

def validate_population(
    matrix: pd.DataFrame,
) -> None:

    expected_counts = {
        2020: 46632,
        2021: 46627,
        2022: 46628,
        2023: 46619,
        2024: 46622,
    }

    print()

    print(
        "YEAR COUNTS"
    )

    for year, expected in (
        expected_counts.items()
    ):

        observed = int(
            (
                matrix[
                    "E64R_YEAR"
                ] == year
            ).sum()
        )

        print(
            f"  {year}: {observed:,}"
        )

        if observed != expected:

            raise RuntimeError(
                f"Year {year} count mismatch."
            )

    # Use the authoritative TEST split for the frozen test.
    test_mask = (
        matrix[
            "E64R_SPLIT"
        ].eq(
            "TEST"
        )
        & matrix[
            "E64R_YEAR"
        ].eq(
            TEST_YEAR
        )
    )

    test_rows = int(
        test_mask.sum()
    )

    test_positives = int(
        matrix.loc[
            test_mask,
            "E64R_TARGET",
        ].sum()
    )

    if test_rows != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Frozen TEST population row count mismatch.\n"
            f"Expected={EXPECTED_TEST_ROWS:,}\n"
            f"Observed={test_rows:,}"
        )

    if test_positives != EXPECTED_TEST_POSITIVES:

        raise RuntimeError(
            "Frozen TEST positive count mismatch.\n"
            f"Expected={EXPECTED_TEST_POSITIVES:,}\n"
            f"Observed={test_positives:,}"
        )

    print()

    print(
        f"Frozen TEST rows      : {test_rows:,}"
    )

    print(
        f"Frozen TEST positives : {test_positives:,}"
    )

    print(
        f"Frozen TEST rate      : "
        f"{test_positives / test_rows:.8f}"
    )


# ============================================================
# 10. MISSINGNESS
# ============================================================

def compute_missingness(
    matrix: pd.DataFrame,
    features: Sequence[str],
) -> pd.Series:

    return (
        matrix[
            list(features)
        ]
        .isna()
        .mean(
            axis=1
        )
        .astype(float)
    )


# ============================================================
# 11. DEVELOPMENT EMPIRICAL REFERENCES
# ============================================================

def fit_development_references(
    matrix: pd.DataFrame,
    features: Sequence[str],
) -> Dict[
    str,
    np.ndarray,
]:

    development = matrix[
        matrix[
            "E64R_YEAR"
        ].isin(
            DEVELOPMENT_YEARS
        )
    ]

    references = {}

    for feature in features:

        values = pd.to_numeric(
            development[
                feature
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        if len(values) == 0:

            raise RuntimeError(
                "No development observations for feature:\n"
                f"{feature}"
            )

        references[
            feature
        ] = np.sort(
            values
        )

    return references


# ============================================================
# 12. COVARIATE EXTREMENESS
# ============================================================

def compute_extremeness(
    matrix: pd.DataFrame,
    features: Sequence[str],
    references: Dict[
        str,
        np.ndarray,
    ],
) -> pd.Series:

    row_count = len(
        matrix
    )

    total = np.zeros(
        row_count,
        dtype=np.float64,
    )

    counts = np.zeros(
        row_count,
        dtype=np.int16,
    )

    for feature in features:

        reference = references[
            feature
        ]

        values = pd.to_numeric(
            matrix[
                feature
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        observed = np.isfinite(
            values
        )

        if not observed.any():
            continue

        x = values[
            observed
        ]

        positions = np.searchsorted(
            reference,
            x,
            side="right",
        )

        q = (
            positions
            / float(
                len(reference)
            )
        )

        q = np.clip(
            q,
            0.0,
            1.0,
        )

        exposure = (
            1.0
            - 2.0
            * np.minimum(
                q,
                1.0 - q,
            )
        )

        exposure = np.clip(
            exposure,
            0.0,
            1.0,
        )

        total[
            observed
        ] += exposure

        counts[
            observed
        ] += 1

    output = np.full(
        row_count,
        np.nan,
        dtype=float,
    )

    valid = (
        counts > 0
    )

    output[
        valid
    ] = (
        total[
            valid
        ]
        / counts[
            valid
        ]
    )

    return pd.Series(
        output,
        index=matrix.index,
        name="COVARIATE_EXTREMENESS",
    )


# ============================================================
# 13. DEVELOPMENT CUTPOINTS
# ============================================================

def development_quantiles(
    matrix: pd.DataFrame,
    series: pd.Series,
) -> Dict[str, float]:

    development_mask = (
        matrix[
            "E64R_YEAR"
        ].isin(
            DEVELOPMENT_YEARS
        )
    )

    values = pd.to_numeric(
        series.loc[
            development_mask
        ],
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    if len(values) == 0:

        raise RuntimeError(
            "No valid development values."
        )

    return {
        "q25":
            float(
                np.quantile(
                    values,
                    0.25,
                )
            ),

        "q50":
            float(
                np.quantile(
                    values,
                    0.50,
                )
            ),

        "q75":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),
    }


def make_quartile_regime(
    series: pd.Series,
    cuts: Dict[str, float],
    prefix: str,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = np.full(
        len(values),
        "UNAVAILABLE",
        dtype=object,
    )

    valid = values.notna()

    result[
        valid
        & (
            values
            <= cuts["q25"]
        )
    ] = (
        f"{prefix}_Q1"
    )

    result[
        valid
        & (
            values
            > cuts["q25"]
        )
        & (
            values
            <= cuts["q50"]
        )
    ] = (
        f"{prefix}_Q2"
    )

    result[
        valid
        & (
            values
            > cuts["q50"]
        )
        & (
            values
            <= cuts["q75"]
        )
    ] = (
        f"{prefix}_Q3"
    )

    result[
        valid
        & (
            values
            > cuts["q75"]
        )
    ] = (
        f"{prefix}_Q4"
    )

    return pd.Series(
        result,
        index=series.index,
    )


def make_binary_regime(
    series: pd.Series,
    median: float,
    prefix: str,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = np.full(
        len(values),
        "UNAVAILABLE",
        dtype=object,
    )

    valid = values.notna()

    result[
        valid
        & (
            values
            <= median
        )
    ] = (
        f"{prefix}_LOW"
    )

    result[
        valid
        & (
            values
            > median
        )
    ] = (
        f"{prefix}_HIGH"
    )

    return pd.Series(
        result,
        index=series.index,
    )


def make_2x2_regime(
    missingness_binary: pd.Series,
    extremeness_binary: pd.Series,
) -> pd.Series:

    result = np.full(
        len(
            missingness_binary
        ),
        "UNAVAILABLE",
        dtype=object,
    )

    miss_low = (
        missingness_binary
        == "MISS_LOW"
    )

    miss_high = (
        missingness_binary
        == "MISS_HIGH"
    )

    ext_low = (
        extremeness_binary
        == "EXT_LOW"
    )

    ext_high = (
        extremeness_binary
        == "EXT_HIGH"
    )

    result[
        miss_low
        & ext_low
    ] = (
        "MISS_LOW_EXT_LOW"
    )

    result[
        miss_low
        & ext_high
    ] = (
        "MISS_LOW_EXT_HIGH"
    )

    result[
        miss_high
        & ext_low
    ] = (
        "MISS_HIGH_EXT_LOW"
    )

    result[
        miss_high
        & ext_high
    ] = (
        "MISS_HIGH_EXT_HIGH"
    )

    return pd.Series(
        result,
        index=missingness_binary.index,
    )


# ============================================================
# 14. LOAD STEP-151
# ============================================================

def load_step151() -> pd.DataFrame:

    if not PAIRED.exists():

        raise FileNotFoundError(
            "Authoritative Step-151 prediction artifact not found:\n"
            f"{PAIRED}"
        )

    paired = pd.read_parquet(
        PAIRED,
        engine="pyarrow",
    )

    required = {
        "MINE_ID",
        "AS_OF_YEAR",
        "Y_TRUE",
        "HGB_PROBABILITY",
        "HGB_PREDICTION",
        "M5_PROBABILITY",
        "M5_PREDICTION",
    }

    missing = sorted(
        required
        - set(
            paired.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step-151 is missing required fields:\n"
            + "\n".join(
                missing
            )
        )

    if len(paired) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Step-151 row count mismatch.\n"
            f"Expected={EXPECTED_TEST_ROWS:,}\n"
            f"Observed={len(paired):,}"
        )

    paired[
        "E64R_MINE_ID"
    ] = normalize_mine_id(
        paired[
            "MINE_ID"
        ]
    )

    paired[
        "E64R_YEAR"
    ] = pd.to_numeric(
        paired[
            "AS_OF_YEAR"
        ],
        errors="coerce",
    )

    if paired[
        "E64R_YEAR"
    ].isna().any():

        raise RuntimeError(
            "Step-151 contains invalid AS_OF_YEAR values."
        )

    paired[
        "E64R_YEAR"
    ] = (
        paired[
            "E64R_YEAR"
        ].astype(
            np.int16
        )
    )

    paired[
        "E64R_Y_TRUE"
    ] = normalize_target(
        paired[
            "Y_TRUE"
        ],
        "Step-151 Y_TRUE",
    )

    paired[
        "E64R_HGB_PROB"
    ] = normalize_probability(
        paired[
            "HGB_PROBABILITY"
        ],
        "Step-151 HGB_PROBABILITY",
    )

    paired[
        "E64R_M5_PROB"
    ] = normalize_probability(
        paired[
            "M5_PROBABILITY"
        ],
        "Step-151 M5_PROBABILITY",
    )

    paired[
        "E64R_HGB_PRED"
    ] = pd.to_numeric(
        paired[
            "HGB_PREDICTION"
        ],
        errors="coerce",
    )

    paired[
        "E64R_M5_PRED"
    ] = pd.to_numeric(
        paired[
            "M5_PREDICTION"
        ],
        errors="coerce",
    )

    hgb_pred_unique = set(
        paired[
            "E64R_HGB_PRED"
        ]
        .astype(int)
        .unique()
    )

    m5_pred_unique = set(
        paired[
            "E64R_M5_PRED"
        ]
        .astype(int)
        .unique()
    )

    if not hgb_pred_unique.issubset(
        {0, 1}
    ):

        raise RuntimeError(
            "Unexpected HGB prediction values."
        )

    if not m5_pred_unique.issubset(
        {0, 1}
    ):

        raise RuntimeError(
            "Unexpected M5 prediction values."
        )

    return paired


# ============================================================
# 15. CRITICAL OCCURRENCE-BASED ALIGNMENT
# ============================================================

def align_authoritative_predictions(
    matrix: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align Step-151 predictions to the frozen Step-134 TEST
    population using exact key multiplicity + occurrence.

    IMPORTANT:
    Repeated mine-year rows are preserved.

    Exact key:
        E64R_MINE_ID
        E64R_YEAR
        E64R_TARGET / E64R_Y_TRUE

    OCCURRENCE:
        deterministic cumcount within the exact key.
    """

    banner(
        "STEP-134 / STEP-151 OCCURRENCE-BASED ALIGNMENT"
    )

    source = matrix[
        (
            matrix[
                "E64R_SPLIT"
            ] == "TEST"
        )
        & (
            matrix[
                "E64R_YEAR"
            ] == TEST_YEAR
        )
    ].copy()

    if len(source) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Step-134 frozen TEST population mismatch "
            "before prediction alignment."
        )

    source["E64R_KEY_TARGET"] = (
        source[
            "E64R_TARGET"
        ].astype(
            np.int8
        )
    )

    prediction = paired[
        paired[
            "E64R_YEAR"
        ] == TEST_YEAR
    ].copy()

    if len(prediction) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Step-151 frozen prediction population mismatch."
        )

    prediction["E64R_KEY_TARGET"] = (
        prediction[
            "E64R_Y_TRUE"
        ].astype(
            np.int8
        )
    )

    key_columns = [
        "E64R_MINE_ID",
        "E64R_YEAR",
        "E64R_KEY_TARGET",
    ]

    source_counts = (
        source
        .groupby(
            key_columns,
            sort=False,
        )
        .size()
        .rename(
            "SOURCE_ROWS"
        )
        .reset_index()
    )

    prediction_counts = (
        prediction
        .groupby(
            key_columns,
            sort=False,
        )
        .size()
        .rename(
            "PREDICTION_ROWS"
        )
        .reset_index()
    )

    coverage = (
        source_counts
        .merge(
            prediction_counts,
            on=key_columns,
            how="outer",
            indicator=True,
        )
    )

    coverage[
        "COUNT_MATCH"
    ] = (
        (
            coverage[
                "_merge"
            ] == "both"
        )
        & (
            coverage[
                "SOURCE_ROWS"
            ]
            ==
            coverage[
                "PREDICTION_ROWS"
            ]
        )
    )

    bad = coverage[
        ~coverage[
            "COUNT_MATCH"
        ]
    ]

    if not bad.empty:

        raise RuntimeError(
            "Step-134 / Step-151 multiplicity mismatch.\n"
            f"Bad key cells={len(bad):,}"
        )

    # Deterministic occurrence assignment.
    source[
        "OCCURRENCE"
    ] = (
        source
        .groupby(
            key_columns,
            sort=False,
        )
        .cumcount()
    )

    prediction[
        "OCCURRENCE"
    ] = (
        prediction
        .groupby(
            key_columns,
            sort=False,
        )
        .cumcount()
    )

    alignment_columns = (
        key_columns
        + [
            "OCCURRENCE",
        ]
    )

    aligned = (
        prediction
        .merge(
            source[
                alignment_columns
                + [
                    "ROW_MISSING_FRACTION",
                    "COVARIATE_EXTREMENESS",
                    "MISSINGNESS_QUARTILE",
                    "EXTREMENESS_QUARTILE",
                    "MISSINGNESS_EXTREMENESS_2X2",
                ]
            ],
            on=alignment_columns,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
    )

    if (
        aligned[
            "_merge"
        ] != "both"
    ).any():

        missing_rows = int(
            (
                aligned[
                    "_merge"
                ] != "both"
            ).sum()
        )

        raise RuntimeError(
            "Occurrence-based row alignment failed.\n"
            f"Unmatched prediction rows={missing_rows:,}"
        )

    if len(aligned) != EXPECTED_TEST_ROWS:

        raise RuntimeError(
            "Aligned population size changed.\n"
            f"Expected={EXPECTED_TEST_ROWS:,}\n"
            f"Observed={len(aligned):,}"
        )

    # Verify targets are identical.
    source_y = (
        source[
            key_columns
            + [
                "OCCURRENCE",
            ]
        ]
        .merge(
            aligned[
                alignment_columns
                + [
                    "E64R_Y_TRUE",
                ]
            ],
            on=alignment_columns,
            how="left",
            validate="one_to_one",
        )
    )

    source_target = (
        source[
            "E64R_TARGET"
        ].to_numpy(
            dtype=np.int8
        )
    )

    prediction_target = (
        aligned[
            "E64R_Y_TRUE"
        ].to_numpy(
            dtype=np.int8
        )
    )

    if not np.array_equal(
        np.sort(
            source_target
        ),
        np.sort(
            prediction_target
        ),
    ):

        raise RuntimeError(
            "Aligned target populations do not match."
        )

    aligned[
        "TARGET"
    ] = aligned[
        "E64R_Y_TRUE"
    ].astype(
        np.int8
    )

    print(
        f"Source rows       : {len(source):,}"
    )

    print(
        f"Prediction rows   : {len(prediction):,}"
    )

    print(
        f"Matched rows      : {len(aligned):,}"
    )

    print(
        f"Repeated key cells: "
        f"{int((source_counts['SOURCE_ROWS'] > 1).sum()):,}"
    )

    print(
        "[PASS] Multiplicity equivalence."
    )

    print(
        "[PASS] Occurrence-based one-to-one alignment."
    )

    print(
        "[PASS] Y_TRUE population preserved."
    )

    return aligned


# ============================================================
# 16. METRIC CALCULATION
# ============================================================

def metric_value(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    metric: str,
) -> float:

    prediction = (
        score
        >= threshold
    ).astype(
        np.int8
    )

    if metric == "balanced_accuracy":

        if len(
            np.unique(
                y_true
            )
        ) < 2:

            return float(
                "nan"
            )

        return float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        )

    if metric == "f1":

        return float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        )

    if metric == "mcc":

        if len(
            np.unique(
                y_true
            )
        ) < 2:

            return float(
                "nan"
            )

        return float(
            matthews_corrcoef(
                y_true,
                prediction,
            )
        )

    if metric == "pr_auc":

        if len(
            np.unique(
                y_true
            )
        ) < 2:

            return float(
                "nan"
            )

        return float(
            average_precision_score(
                y_true,
                score,
            )
        )

    raise RuntimeError(
        f"Unsupported metric={metric}"
    )


# ============================================================
# 17. PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(
    y_true: np.ndarray,
    hgb_score: np.ndarray,
    m5_score: np.ndarray,
    metric: str,
    seed: int,
) -> Dict[str, float]:

    n = len(
        y_true
    )

    observed_hgb = metric_value(
        y_true,
        hgb_score,
        HGB_THRESHOLD,
        metric,
    )

    observed_m5 = metric_value(
        y_true,
        m5_score,
        M5_THRESHOLD,
        metric,
    )

    observed_difference = (
        observed_m5
        - observed_hgb
    )

    rng = np.random.default_rng(
        seed
    )

    differences = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    valid = 0

    for index in range(
        N_BOOTSTRAP
    ):

        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )

        y = y_true[
            sample_indices
        ]

        hgb = hgb_score[
            sample_indices
        ]

        m5 = m5_score[
            sample_indices
        ]

        hgb_metric = metric_value(
            y,
            hgb,
            HGB_THRESHOLD,
            metric,
        )

        m5_metric = metric_value(
            y,
            m5,
            M5_THRESHOLD,
            metric,
        )

        difference = (
            m5_metric
            - hgb_metric
        )

        if np.isfinite(
            difference
        ):

            differences[
                valid
            ] = difference

            valid += 1

    if valid == 0:

        return {
            "n":
                int(n),

            "observed_diff":
                float(
                    observed_difference
                ),

            "bootstrap_mean_diff":
                float("nan"),

            "ci_lower":
                float("nan"),

            "ci_upper":
                float("nan"),

            "p_two_sided":
                float("nan"),

            "valid_bootstrap":
                0,
        }

    differences = (
        differences[
            :valid
        ]
    )

    ci_lower = float(
        np.quantile(
            differences,
            0.025,
        )
    )

    ci_upper = float(
        np.quantile(
            differences,
            0.975,
        )
    )

    lower_tail = float(
        np.mean(
            differences
            <= 0.0
        )
    )

    upper_tail = float(
        np.mean(
            differences
            >= 0.0
        )
    )

    p_value = float(
        min(
            1.0,
            2.0
            * min(
                lower_tail,
                upper_tail,
            ),
        )
    )

    return {
        "n":
            int(n),

        "observed_diff":
            float(
                observed_difference
            ),

        "bootstrap_mean_diff":
            float(
                differences.mean()
            ),

        "ci_lower":
            ci_lower,

        "ci_upper":
            ci_upper,

        "p_two_sided":
            p_value,

        "valid_bootstrap":
            int(valid),
    }


# ============================================================
# 18. EVALUATE REGIMES
# ============================================================

METRICS = [
    "balanced_accuracy",
    "f1",
    "mcc",
    "pr_auc",
]


def evaluate_family(
    test_df: pd.DataFrame,
    family: str,
    regime_column: str,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
]:

    metric_rows = []

    bootstrap_rows = []

    regimes = sorted(
        [
            str(value)
            for value
            in test_df[
                regime_column
            ].unique()
            if str(value)
            != "UNAVAILABLE"
        ]
    )

    if not regimes:

        raise RuntimeError(
            f"No usable regimes for {family}."
        )

    for regime in regimes:

        subset = (
            test_df[
                test_df[
                    regime_column
                ].astype(str)
                == regime
            ]
        )

        if subset.empty:

            raise RuntimeError(
                f"Empty regime: {family}/{regime}"
            )

        y_true = subset[
            "TARGET"
        ].to_numpy(
            dtype=np.int8
        )

        hgb = subset[
            "E64R_HGB_PROB"
        ].to_numpy(
            dtype=float
        )

        m5 = subset[
            "E64R_M5_PROB"
        ].to_numpy(
            dtype=float
        )

        for metric in METRICS:

            hgb_metric = metric_value(
                y_true,
                hgb,
                HGB_THRESHOLD,
                metric,
            )

            m5_metric = metric_value(
                y_true,
                m5,
                M5_THRESHOLD,
                metric,
            )

            delta = float(
                m5_metric
                - hgb_metric
            )

            metric_rows.append(
                {
                    "regime_family":
                        family,

                    "regime":
                        regime,

                    "n":
                        int(
                            len(subset)
                        ),

                    "positives":
                        int(
                            y_true.sum()
                        ),

                    "positive_rate":
                        float(
                            y_true.mean()
                        ),

                    "metric":
                        metric,

                    "HGB":
                        float(
                            hgb_metric
                        ),

                    "M5":
                        float(
                            m5_metric
                        ),

                    "delta_M5_minus_HGB":
                        delta,

                    "HGB_threshold":
                        HGB_THRESHOLD,

                    "M5_threshold":
                        M5_THRESHOLD,

                    "test_year":
                        TEST_YEAR,
                }
            )

            stable_key = (
                f"{family}|"
                f"{regime}|"
                f"{metric}"
            )

            stable_hash = int(
                hashlib.sha256(
                    stable_key.encode(
                        "utf-8"
                    )
                ).hexdigest()[
                    :8
                ],
                16,
            )

            bootstrap_result = (
                paired_bootstrap(
                    y_true,
                    hgb,
                    m5,
                    metric,
                    SEED
                    + (
                        stable_hash
                        % 1_000_000
                    ),
                )
            )

            bootstrap_rows.append(
                {
                    "regime_family":
                        family,

                    "regime":
                        regime,

                    "metric":
                        metric,

                    **bootstrap_result,

                    "HGB_threshold":
                        HGB_THRESHOLD,

                    "M5_threshold":
                        M5_THRESHOLD,

                    "test_year":
                        TEST_YEAR,
                }
            )

    return (
        metric_rows,
        bootstrap_rows,
    )


# ============================================================
# 19. MAIN
# ============================================================

def main() -> int:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    banner(
        "ACHER-Net Q1 â€” E6.4R"
    )

    print(
        "Conditional Adaptation Benefit"
    )

    print(
        f"Seed                 : {SEED}"
    )

    print(
        f"Bootstrap replicates : {N_BOOTSTRAP:,}"
    )

    print(
        f"HGB threshold        : {HGB_THRESHOLD}"
    )

    print(
        f"M5 threshold         : {M5_THRESHOLD}"
    )

    print(
        f"Development          : {DEVELOPMENT_YEARS}"
    )

    print(
        f"Validation           : {VALIDATION_YEAR}"
    )

    print(
        f"Frozen test          : {TEST_YEAR}"
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    banner(
        "AUTHORITATIVE INPUT PROVENANCE"
    )

    print(
        f"Step-134:\n{MATRIX}"
    )

    print(
        f"Step-151:\n{PAIRED}"
    )

    print(
        f"Step-169R:\n{FEATURE_MANIFEST}"
    )

    for path in [
        MATRIX,
        PAIRED,
        FEATURE_MANIFEST,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact does not exist:\n"
                f"{path}"
            )

    print(
        "[PASS] Required authoritative artifacts exist."
    )

    # --------------------------------------------------------
    # Feature contract
    # --------------------------------------------------------

    banner(
        "AUTHORITATIVE 108-FEATURE CONTRACT"
    )

    features = (
        load_features()
    )

    print(
        f"Candidate predictors : "
        f"{EXPECTED_CANDIDATES}"
    )

    print(
        f"Selected predictors  : "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Matrix
    # --------------------------------------------------------

    banner(
        "LOAD STEP-134"
    )

    matrix = load_matrix(
        features
    )

    print(
        f"Rows    : {len(matrix):,}"
    )

    print(
        f"Columns : {matrix.shape[1]:,}"
    )

    validate_population(
        matrix
    )

    # --------------------------------------------------------
    # Reliability variables
    # --------------------------------------------------------

    banner(
        "RELIABILITY VARIABLE CONSTRUCTION"
    )

    matrix[
        "ROW_MISSING_FRACTION"
    ] = compute_missingness(
        matrix,
        features,
    )

    print(
        "Row-wise missingness computed over "
        f"{len(features)} selected predictors."
    )

    references = (
        fit_development_references(
            matrix,
            features,
        )
    )

    print(
        f"Development feature references: "
        f"{len(references):,}"
    )

    matrix[
        "COVARIATE_EXTREMENESS"
    ] = compute_extremeness(
        matrix,
        features,
        references,
    )

    # --------------------------------------------------------
    # Development-only cutpoints
    # --------------------------------------------------------

    banner(
        "DEVELOPMENT-ONLY REGIME CUTPOINTS"
    )

    missingness_cuts = (
        development_quantiles(
            matrix,
            matrix[
                "ROW_MISSING_FRACTION"
            ],
        )
    )

    extremeness_cuts = (
        development_quantiles(
            matrix,
            matrix[
                "COVARIATE_EXTREMENESS"
            ],
        )
    )

    print(
        "Missingness:"
    )

    print(
        f"  Q25 = "
        f"{missingness_cuts['q25']:.12f}"
    )

    print(
        f"  Q50 = "
        f"{missingness_cuts['q50']:.12f}"
    )

    print(
        f"  Q75 = "
        f"{missingness_cuts['q75']:.12f}"
    )

    print(
        "Covariate extremeness:"
    )

    print(
        f"  Q25 = "
        f"{extremeness_cuts['q25']:.12f}"
    )

    print(
        f"  Q50 = "
        f"{extremeness_cuts['q50']:.12f}"
    )

    print(
        f"  Q75 = "
        f"{extremeness_cuts['q75']:.12f}"
    )

    # --------------------------------------------------------
    # Regimes
    # --------------------------------------------------------

    matrix[
        "MISSINGNESS_QUARTILE"
    ] = make_quartile_regime(
        matrix[
            "ROW_MISSING_FRACTION"
        ],
        missingness_cuts,
        "MISS",
    )

    matrix[
        "EXTREMENESS_QUARTILE"
    ] = make_quartile_regime(
        matrix[
            "COVARIATE_EXTREMENESS"
        ],
        extremeness_cuts,
        "EXT",
    )

    missingness_binary = (
        make_binary_regime(
            matrix[
                "ROW_MISSING_FRACTION"
            ],
            missingness_cuts[
                "q50"
            ],
            "MISS",
        )
    )

    extremeness_binary = (
        make_binary_regime(
            matrix[
                "COVARIATE_EXTREMENESS"
            ],
            extremeness_cuts[
                "q50"
            ],
            "EXT",
        )
    )

    matrix[
        "MISSINGNESS_EXTREMENESS_2X2"
    ] = make_2x2_regime(
        missingness_binary,
        extremeness_binary,
    )

    print(
        "[PASS] Development-defined regimes."
    )

    # --------------------------------------------------------
    # Load Step-151
    # --------------------------------------------------------

    banner(
        "LOAD AUTHORITATIVE STEP-151"
    )

    paired = (
        load_step151()
    )

    print(
        f"Step-151 rows: "
        f"{len(paired):,}"
    )

    # --------------------------------------------------------
    # Occurrence-based alignment
    # --------------------------------------------------------

    aligned = (
        align_authoritative_predictions(
            matrix,
            paired,
        )
    )

    # --------------------------------------------------------
    # Build analysis dataframe
    # --------------------------------------------------------

    test_df = aligned.copy()

    # --------------------------------------------------------
    # Print reliability distributions
    # --------------------------------------------------------

    banner(
        "FROZEN TEST RELIABILITY REGIMES"
    )

    regime_map = {
        "missingness_quartile":
            "MISSINGNESS_QUARTILE",

        "extremeness_quartile":
            "EXTREMENESS_QUARTILE",

        "missingness_extremeness_2x2":
            "MISSINGNESS_EXTREMENESS_2X2",
    }

    for family, column in (
        regime_map.items()
    ):

        print()
        print(
            f"[{family}]"
        )

        counts = (
            test_df[
                column
            ]
            .value_counts()
            .sort_index()
        )

        for regime, count in (
            counts.items()
        ):

            regime_text = str(
                regime
            )

            if (
                regime_text
                == "UNAVAILABLE"
            ):

                continue

            subset = test_df[
                test_df[
                    column
                ]
                == regime
            ]

            positives = int(
                subset[
                    "TARGET"
                ].sum()
            )

            print(
                f"  {regime_text}: "
                f"n={len(subset):,}; "
                f"positives={positives:,}; "
                f"rate="
                f"{positives / len(subset):.6f}"
            )

    # --------------------------------------------------------
    # Conditional evaluation
    # --------------------------------------------------------

    banner(
        "CONDITIONAL ADAPTATION BENEFIT"
    )

    all_metrics = []

    all_bootstraps = []

    for family, column in (
        regime_map.items()
    ):

        metrics, bootstraps = (
            evaluate_family(
                test_df,
                family,
                column,
            )
        )

        all_metrics.extend(
            metrics
        )

        all_bootstraps.extend(
            bootstraps
        )

    metric_df = pd.DataFrame(
        all_metrics
    )

    bootstrap_df = pd.DataFrame(
        all_bootstraps
    )

    if metric_df.empty:

        raise RuntimeError(
            "No conditional metric results were produced."
        )

    if bootstrap_df.empty:

        raise RuntimeError(
            "No bootstrap results were produced."
        )

    # --------------------------------------------------------
    # Results display
    # --------------------------------------------------------

    print()

    print(
        metric_df[
            [
                "regime_family",
                "regime",
                "n",
                "positives",
                "metric",
                "HGB",
                "M5",
                "delta_M5_minus_HGB",
            ]
        ].to_string(
            index=False
        )
    )

    banner(
        "PAIRED BOOTSTRAP â€” M5 - HGB"
    )

    print(
        bootstrap_df[
            [
                "regime_family",
                "regime",
                "metric",
                "n",
                "observed_diff",
                "ci_lower",
                "ci_upper",
                "p_two_sided",
                "valid_bootstrap",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Metrics CSV
    # --------------------------------------------------------

    metric_path = (
        OUT
        / "e64r_regime_metrics.csv"
    )

    metric_df.to_csv(
        metric_path,
        index=False,
    )

    # --------------------------------------------------------
    # Bootstrap CSV
    # --------------------------------------------------------

    bootstrap_path = (
        OUT
        / "e64r_pairwise_bootstrap.csv"
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    # --------------------------------------------------------
    # Regime definition CSV
    # --------------------------------------------------------

    regime_definition = pd.DataFrame(
        [
            {
                "regime_family":
                    "missingness_quartile",
                "cutpoint":
                    "Q25",
                "value":
                    missingness_cuts["q25"],
                "source":
                    "development_2020_2022_only",
            },
            {
                "regime_family":
                    "missingness_quartile",
                "cutpoint":
                    "Q50",
                "value":
                    missingness_cuts["q50"],
                "source":
                    "development_2020_2022_only",
            },
            {
                "regime_family":
                    "missingness_quartile",
                "cutpoint":
                    "Q75",
                "value":
                    missingness_cuts["q75"],
                "source":
                    "development_2020_2022_only",
            },
            {
                "regime_family":
                    "extremeness_quartile",
                "cutpoint":
                    "Q25",
                "value":
                    extremeness_cuts["q25"],
                "source":
                    "development_2020_2022_only",
            },
            {
                "regime_family":
                    "extremeness_quartile",
                "cutpoint":
                    "Q50",
                "value":
                    extremeness_cuts["q50"],
                "source":
                    "development_2020_2022_only",
            },
            {
                "regime_family":
                    "extremeness_quartile",
                "cutpoint":
                    "Q75",
                "value":
                    extremeness_cuts["q75"],
                "source":
                    "development_2020_2022_only",
            },
        ]
    )

    regime_definition_path = (
        OUT
        / "e64r_regime_definition.csv"
    )

    regime_definition.to_csv(
        regime_definition_path,
        index=False,
    )

    # --------------------------------------------------------
    # Feature extremeness summary
    # --------------------------------------------------------

    feature_summary_rows = []

    for feature in features:

        values = references[
            feature
        ]

        feature_summary_rows.append(
            {
                "feature":
                    feature,

                "development_observed_n":
                    int(
                        len(values)
                    ),

                "development_min":
                    float(
                        values.min()
                    ),

                "development_q25":
                    float(
                        np.quantile(
                            values,
                            0.25,
                        )
                    ),

                "development_median":
                    float(
                        np.quantile(
                            values,
                            0.50,
                        )
                    ),

                "development_q75":
                    float(
                        np.quantile(
                            values,
                            0.75,
                        )
                    ),

                "development_max":
                    float(
                        values.max()
                    ),
            }
        )

    feature_summary_df = (
        pd.DataFrame(
            feature_summary_rows
        )
    )

    feature_summary_path = (
        OUT
        / "e64r_feature_extremeness_summary.csv"
    )

    feature_summary_df.to_csv(
        feature_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Test assignments
    # --------------------------------------------------------

    assignment_df = test_df[
        [
            "E64R_MINE_ID",
            "E64R_YEAR",
            "TARGET",
            "E64R_HGB_PROB",
            "E64R_M5_PROB",
            "ROW_MISSING_FRACTION",
            "COVARIATE_EXTREMENESS",
            "MISSINGNESS_QUARTILE",
            "EXTREMENESS_QUARTILE",
            "MISSINGNESS_EXTREMENESS_2X2",
        ]
    ].copy()

    assignment_df = (
        assignment_df.rename(
            columns={
                "E64R_MINE_ID":
                    "MINE_ID",

                "E64R_YEAR":
                    "AS_OF_YEAR",

                "E64R_HGB_PROB":
                    "HGB_PROBABILITY",

                "E64R_M5_PROB":
                    "M5_PROBABILITY",
            }
        )
    )

    assignment_path = (
        OUT
        / "e64r_test_regime_assignments.csv"
    )

    assignment_df.to_csv(
        assignment_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    m5_point_favorable = int(
        (
            metric_df[
                "delta_M5_minus_HGB"
            ]
            > 0
        )
        .fillna(False)
        .sum()
    )

    hgb_point_favorable = int(
        (
            metric_df[
                "delta_M5_minus_HGB"
            ]
            < 0
        )
        .fillna(False)
        .sum()
    )

    summary = {
        "experiment":
            "E6.4R",

        "description":
            "Conditional adaptation benefit analysis",

        "seed":
            SEED,

        "bootstrap_replicates":
            N_BOOTSTRAP,

        "thresholds":
            {
                "HGB":
                    HGB_THRESHOLD,

                "M5":
                    M5_THRESHOLD,
            },

        "temporal_protocol":
            {
                "development_years":
                    list(
                        DEVELOPMENT_YEARS
                    ),

                "validation_year":
                    VALIDATION_YEAR,

                "frozen_test_reference_year":
                    TEST_YEAR,

                "frozen_test_horizon":
                    "2024_to_2025",
            },

        "matrix_contract":
            {
                "rows":
                    EXPECTED_MATRIX_ROWS,

                "columns":
                    EXPECTED_MATRIX_COLUMNS,

                "candidate_features":
                    EXPECTED_CANDIDATES,

                "selected_features":
                    EXPECTED_SELECTED,
            },

        "frozen_test_contract":
            {
                "rows":
                    EXPECTED_TEST_ROWS,

                "positives":
                    EXPECTED_TEST_POSITIVES,

                "positive_rate":
                    EXPECTED_TEST_POSITIVES
                    / EXPECTED_TEST_ROWS,
            },

        "regime_cutpoints":
            {
                "missingness":
                    missingness_cuts,

                "covariate_extremeness":
                    extremeness_cuts,
            },

        "conditional_results":
            {
                "metric_rows":
                    int(
                        len(metric_df)
                    ),

                "bootstrap_rows":
                    int(
                        len(
                            bootstrap_df
                        )
                    ),

                "metric_cells_point_favoring_M5":
                    m5_point_favorable,

                "metric_cells_point_favoring_HGB":
                    hgb_point_favorable,
            },

        "alignment":
            {
                "method":
                    (
                        "MINE_ID + AS_OF_YEAR + "
                        "Y_TRUE + OCCURRENCE"
                    ),

                "occurrence_based":
                    True,

                "deduplication_performed":
                    False,

                "duplicate_mine_year_rows_removed":
                    0,
            },

        "scientific_safeguards":
            {
                "test_labels_used_for_regime_cutpoints":
                    False,

                "threshold_tuning":
                    False,

                "model_retraining":
                    False,

                "step230_reconstructed_HGB_used":
                    False,

                "step233_reconstructed_HGB_used":
                    False,

                "existing_acher_net_modified":
                    False,

                "E5_modified":
                    False,

                "manuscript_modified":
                    False,
            },

        "interpretation":
            (
                "E6.4R evaluates context-specific M5-minus-HGB "
                "performance differences across reliability regimes. "
                "A positive regime-specific difference is not treated "
                "as evidence of universal ACHER-SAGE superiority."
            ),
    }

    summary_path = (
        OUT
        / "e64r_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    output_paths = [
        metric_path,
        bootstrap_path,
        regime_definition_path,
        feature_summary_path,
        assignment_path,
        summary_path,
    ]

    manifest = {
        "experiment":
            "E6.4R",

        "inputs":
            {
                "step134_matrix":
                    {
                        "path":
                            str(MATRIX),

                        "sha256":
                            sha256_file(
                                MATRIX
                            ),

                        "bytes":
                            MATRIX.stat().st_size,
                    },

                "step151_paired_predictions":
                    {
                        "path":
                            str(PAIRED),

                        "sha256":
                            sha256_file(
                                PAIRED
                            ),

                        "bytes":
                            PAIRED.stat().st_size,
                    },

                "step169R_feature_manifest":
                    {
                        "path":
                            str(FEATURE_MANIFEST),

                        "sha256":
                            sha256_file(
                                FEATURE_MANIFEST
                            ),

                        "bytes":
                            FEATURE_MANIFEST.stat().st_size,
                    },
            },

        "configuration":
            {
                "seed":
                    SEED,

                "bootstrap_replicates":
                    N_BOOTSTRAP,

                "HGB_threshold":
                    HGB_THRESHOLD,

                "M5_threshold":
                    M5_THRESHOLD,

                "development_years":
                    list(
                        DEVELOPMENT_YEARS
                    ),

                "validation_year":
                    VALIDATION_YEAR,

                "test_year":
                    TEST_YEAR,
            },

        "alignment_protocol":
            {
                "key":
                    [
                        "E64R_MINE_ID",
                        "E64R_YEAR",
                        "E64R_KEY_TARGET",
                        "OCCURRENCE",
                    ],

                "deduplicate":
                    False,
            },

        "outputs":
            {},
    }

    for path in output_paths:

        manifest[
            "outputs"
        ][
            path.name
        ] = {
            "path":
                str(path),

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                path.stat().st_size,
        }

    manifest_path = (
        OUT
        / "e64r_run_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    banner(
        "E6.4R COMPLETE â€” STATUS=PASS"
    )

    print(
        f"Output directory:\n{OUT}"
    )

    print()

    print(
        "[PASS] Step-134 authoritative matrix."
    )

    print(
        "[PASS] Step-169R authoritative 108-feature contract."
    )

    print(
        "[PASS] Step-151 authoritative predictions."
    )

    print(
        "[PASS] Repeated mine-year rows preserved."
    )

    print(
        "[PASS] Occurrence-based alignment."
    )

    print(
        "[PASS] Frozen 46,622-row test population."
    )

    print(
        "[PASS] Development-only regime cutpoints."
    )

    print(
        "[PASS] Locked HGB/M5 thresholds."
    )

    print(
        "[PASS] No test-derived regime tuning."
    )

    print(
        "[PASS] No retraining."
    )

    print(
        "[PASS] Existing ACHER-Net artifacts untouched."
    )

    print(
        "[PASS] E6.4R outputs written."
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nE6.4R interrupted by user.",
            file=sys.stderr,
        )

        raise

    except Exception as exc:

        print(
            "\n"
            + "=" * 108,
            file=sys.stderr,
        )

        print(
            "E6.4R FAILED",
            file=sys.stderr,
        )

        print(
            "=" * 108,
            file=sys.stderr,
        )

        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise
