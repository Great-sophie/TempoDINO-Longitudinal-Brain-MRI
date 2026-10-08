#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_evaluate_tempodino_v7_oof.py

Locked OOF aggregation for TempoDINO-v7 unified developmental axis.

PRIMARY TASK
------------
Longitudinal developmental representation learning.

Primary representation metrics
------------------------------
1. Developmental score / chronological-age correlation
2. Within-subject developmental order accuracy
3. Delta developmental score / Delta-time correlation
4. Subject-level developmental monotonicity
5. Subject-level consecutive-order accuracy
6. Global relative temporal-spacing accuracy
7. Local relative temporal-spacing accuracy
8. Spacing distance / Delta-time correlation
9. Cross-subject developmental-order accuracy
10. Small / medium / large age-gap cross-subject accuracy

Secondary probes
----------------
1. Age MAE
2. Age RMSE
3. Age R2
4. Age Pearson
5. Age Spearman
6. Age-probe longitudinal order
7. Delta-age MAE / correlation
8. Effective rank

IMPORTANT
---------
Age prediction is a downstream probe, NOT the training objective.

Locked checkpoint policy:
    v7_final for every fold.

No per-fold post-hoc checkpoint selection is allowed.

Debug mode:
    --allow_partial

allows aggregation of currently available folds, but the resulting
summary is explicitly marked as incomplete and MUST NOT be treated
as the final locked OOF result.
"""

# ============================================================
# IMPORTS
# ============================================================

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    wilcoxon,
)


# ============================================================
# PATHS
# ============================================================

# Existing MRI research directory (contains metadata/, checkpoints/, results/).
# Override via TEMPODINO_PROJECT_DIR; never hard-code a personal machine path.
PROJECT_DIR = Path(os.environ.get(
    "TEMPODINO_PROJECT_DIR", Path(__file__).resolve().parents[1]
)).expanduser().resolve()

# Writes go to an independent release output directory to protect historical runs.
RUN_ROOT = Path(os.environ.get(
    "TEMPODINO_RUN_ROOT", Path(__file__).resolve().parents[1] / "local_runs"
)).expanduser().resolve()

INPUT_ROOT = (
    PROJECT_DIR
    / "results"
    / "tempodino_v7_unified_axis_probe"
)

OUTPUT_ROOT = (
    RUN_ROOT
    / "results"
    / "tempodino_v7_oof"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "TempoDINO-v7 locked OOF aggregation"
    )
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="v7_final",
    help=(
        "Locked checkpoint name. "
        "Default: v7_final"
    ),
)

parser.add_argument(
    "--folds",
    type=int,
    nargs="+",
    default=[
        0,
        1,
        2,
        3,
        4,
    ],
)

parser.add_argument(
    "--expected_mri",
    type=int,
    default=114,
)

parser.add_argument(
    "--expected_subjects",
    type=int,
    default=23,
)

parser.add_argument(
    "--allow_partial",
    action="store_true",
    help=(
        "Allow aggregation of currently available folds. "
        "DEBUG ONLY. Result is not a locked complete OOF."
    ),
)

args = parser.parse_args()


CHECKPOINT_NAME = (
    args.checkpoint
)


# ============================================================
# OUTPUT FILES
# ============================================================

OOF_PREDICTIONS_PATH = (
    OUTPUT_ROOT
    / "v7_oof_predictions.csv"
)

OOF_WITHIN_PATH = (
    OUTPUT_ROOT
    / "v7_oof_within_pairs.csv"
)

OOF_TRIPLETS_PATH = (
    OUTPUT_ROOT
    / "v7_oof_triplets.csv"
)

OOF_CROSS_PATH = (
    OUTPUT_ROOT
    / "v7_oof_cross_pairs.csv"
)

OOF_SUBJECT_GEOMETRY_PATH = (
    OUTPUT_ROOT
    / "v7_oof_subject_geometry.csv"
)

FOLD_METRICS_PATH = (
    OUTPUT_ROOT
    / "v7_fold_metrics.csv"
)

OOF_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "v7_oof_summary.csv"
)

OOF_SUMMARY_JSON_PATH = (
    OUTPUT_ROOT
    / "v7_oof_summary.json"
)

TABLE1_PATH = (
    OUTPUT_ROOT
    / "table1_tempodino_v7_locked.csv"
)

TABLE2_PATH = (
    OUTPUT_ROOT
    / "table2_secondary_age_probe_locked.csv"
)

AUDIT_PATH = (
    OUTPUT_ROOT
    / "oof_audit.json"
)


# ============================================================
# HELPERS
# ============================================================

def safe_pearson(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if len(x) < 2:

        return np.nan

    if (
        np.std(x) <= 0
        or
        np.std(y) <= 0
    ):

        return np.nan

    return float(
        pearsonr(
            x,
            y,
        ).statistic
    )


def safe_spearman(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if len(x) < 2:

        return np.nan

    if (
        np.std(x) <= 0
        or
        np.std(y) <= 0
    ):

        return np.nan

    return float(
        spearmanr(
            x,
            y,
        ).statistic
    )


def mean_or_nan(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def std_or_nan(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) <= 1:

        return np.nan

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def sem_or_nan(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) <= 1:

        return np.nan

    return float(
        np.std(
            values,
            ddof=1,
        )
        /
        np.sqrt(
            len(values)
        )
    )


def json_safe_float(
    value,
):

    if value is None:

        return None

    try:

        value = float(
            value
        )

    except Exception:

        return None

    if not np.isfinite(
        value
    ):

        return None

    return value


def require_columns(
    dataframe,
    required_columns,
    label,
    fold,
):

    missing = [
        column

        for column
        in required_columns

        if column
        not in dataframe.columns
    ]

    if missing:

        raise RuntimeError(
            f"Fold {fold}: {label} missing columns: "
            f"{missing}"
        )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("TEMPODINO V7 LOCKED OOF AGGREGATION")
print("=" * 80)

print(
    f"\nRequested checkpoint: "
    f"{CHECKPOINT_NAME}"
)

print(
    f"Requested folds:      "
    f"{args.folds}"
)

print(
    f"Allow partial:        "
    f"{args.allow_partial}"
)


# ============================================================
# LOAD AVAILABLE FOLDS
# ============================================================

all_predictions = []
all_within = []
all_triplets = []
all_cross = []
all_subject_geometry = []
fold_summary_rows = []

loaded_folds = []
missing_folds = []

missing_files_by_fold = {}


for fold in args.folds:

    fold_dir = (
        INPUT_ROOT
        / f"fold_{fold}"
    )

    prefix = (
        fold_dir
        / CHECKPOINT_NAME
    )

    prediction_path = Path(
        str(
            prefix
        )
        +
        "_predictions.csv"
    )

    within_path = Path(
        str(
            prefix
        )
        +
        "_within_pairs.csv"
    )

    triplet_path = Path(
        str(
            prefix
        )
        +
        "_triplets.csv"
    )

    cross_path = Path(
        str(
            prefix
        )
        +
        "_cross_pairs.csv"
    )

    subject_geometry_path = Path(
        str(
            prefix
        )
        +
        "_subject_geometry.csv"
    )

    summary_path = Path(
        str(
            prefix
        )
        +
        "_summary.csv"
    )


    required_files = {

        "predictions":
            prediction_path,

        "within_pairs":
            within_path,

        "triplets":
            triplet_path,

        "cross_pairs":
            cross_path,

        "subject_geometry":
            subject_geometry_path,

        "summary":
            summary_path,
    }


    missing = {

        name:
            str(
                path
            )

        for name, path
        in required_files.items()

        if not path.exists()
    }


    if missing:

        missing_folds.append(
            fold
        )

        missing_files_by_fold[
            str(
                fold
            )
        ] = missing

        print(
            f"\nFold {fold}: INCOMPLETE"
        )

        for name, path in missing.items():

            print(
                f"  MISSING {name}: "
                f"{path}"
            )


        if args.allow_partial:

            print(
                f"  -> Skipping fold {fold}"
            )

            continue


        raise FileNotFoundError(

            "\n"
            f"Fold {fold} is incomplete.\n\n"

            "Missing files:\n"

            +

            "\n".join(

                f"  {name}: {path}"

                for name, path
                in missing.items()
            )

            +

            "\n\nGenerate this fold first with:\n"

            f"python "
            f"19b_probe_tempodino_v7_unified_axis.py "
            f"--fold {fold} "
            f"--checkpoint v7_final "
            f"--batch_size 8 "
            f"--ridge_alpha 10"
        )


    # ========================================================
    # READ COMPLETE FOLD
    # ========================================================

    pred_df_fold = pd.read_csv(
        prediction_path,
        dtype={
            "subject_id":
                str
        },
    )

    within_df_fold = pd.read_csv(
        within_path,
        dtype={
            "subject_id":
                str
        },
    )

    triplet_df_fold = pd.read_csv(
        triplet_path,
        dtype={
            "subject_id":
                str
        },
    )

    cross_df_fold = pd.read_csv(
        cross_path
    )

    subject_df_fold = pd.read_csv(
        subject_geometry_path,
        dtype={
            "subject_id":
                str
        },
    )

    fold_summary = pd.read_csv(
        summary_path
    )


    # ========================================================
    # BASIC FILE AUDIT
    # ========================================================

    if len(
        pred_df_fold
    ) == 0:

        raise RuntimeError(
            f"Fold {fold}: "
            "prediction file is empty."
        )


    if len(
        within_df_fold
    ) == 0:

        raise RuntimeError(
            f"Fold {fold}: "
            "within-pair file is empty."
        )


    if len(
        triplet_df_fold
    ) == 0:

        raise RuntimeError(
            f"Fold {fold}: "
            "triplet file is empty."
        )


    if len(
        cross_df_fold
    ) == 0:

        raise RuntimeError(
            f"Fold {fold}: "
            "cross-pair file is empty."
        )


    if len(
        subject_df_fold
    ) == 0:

        raise RuntimeError(
            f"Fold {fold}: "
            "subject geometry file is empty."
        )


    if len(
        fold_summary
    ) != 1:

        raise RuntimeError(
            f"Fold {fold}: expected exactly "
            f"one summary row, "
            f"found {len(fold_summary)}."
        )


    # ========================================================
    # COLUMN AUDIT
    # ========================================================

    require_columns(

        pred_df_fold,

        [
            "subject_id",
            "age_month",
            "developmental_score",
            "age_pred",
        ],

        "predictions",

        fold,
    )


    require_columns(

        within_df_fold,

        [
            "subject_id",
            "delta_age_true",
            "delta_score",
            "axis_order_correct",
            "age_probe_order_correct",
            "delta_age_pred",
            "spacing_distance",
        ],

        "within pairs",

        fold,
    )


    require_columns(

        triplet_df_fold,

        [
            "subject_id",
            "delta12",
            "delta23",
            "d12",
            "d23",
            "d13",
        ],

        "triplets",

        fold,
    )


    require_columns(

        cross_df_fold,

        [
            "gap_bin",
            "correct",
        ],

        "cross pairs",

        fold,
    )


    require_columns(

        subject_df_fold,

        [
            "subject_id",
            "n_scans",
            "developmental_monotonicity",
            "developmental_consecutive_accuracy",
            "age_probe_monotonicity",
            "age_probe_consecutive_accuracy",
        ],

        "subject geometry",

        fold,
    )


    # ========================================================
    # NORMALIZE SUBJECT ID
    # ========================================================

    for dataframe in [

        pred_df_fold,
        within_df_fold,
        triplet_df_fold,
        subject_df_fold,

    ]:

        dataframe[
            "subject_id"
        ] = (
            dataframe[
                "subject_id"
            ]
            .astype(str)
            .str.zfill(3)
        )


    # ========================================================
    # ADD FOLD LABEL
    # ========================================================

    pred_df_fold[
        "fold"
    ] = fold

    within_df_fold[
        "fold"
    ] = fold

    triplet_df_fold[
        "fold"
    ] = fold

    cross_df_fold[
        "fold"
    ] = fold

    subject_df_fold[
        "fold"
    ] = fold


    # ========================================================
    # STORE
    # ========================================================

    all_predictions.append(
        pred_df_fold
    )

    all_within.append(
        within_df_fold
    )

    all_triplets.append(
        triplet_df_fold
    )

    all_cross.append(
        cross_df_fold
    )

    all_subject_geometry.append(
        subject_df_fold
    )


    summary_row = (
        fold_summary
        .iloc[0]
        .to_dict()
    )

    summary_row[
        "fold"
    ] = fold

    fold_summary_rows.append(
        summary_row
    )


    loaded_folds.append(
        fold
    )


    print(
        f"Fold {fold}: OK | "
        f"MRI={len(pred_df_fold)}, "
        f"subjects="
        f"{pred_df_fold['subject_id'].nunique()}, "
        f"within_pairs={len(within_df_fold)}, "
        f"triplets={len(triplet_df_fold)}, "
        f"cross_pairs={len(cross_df_fold)}"
    )


# ============================================================
# NO DATA GUARD
# ============================================================

if len(
    loaded_folds
) == 0:

    raise RuntimeError(
        "No complete folds are available."
    )


# ============================================================
# FOLD STATUS
# ============================================================

locked_complete = (

    set(
        loaded_folds
    )
    ==
    set(
        args.folds
    )
)


print("\n" + "=" * 80)
print("FOLD AVAILABILITY")
print("=" * 80)

print(
    f"Loaded folds:  "
    f"{loaded_folds}"
)

print(
    f"Missing folds: "
    f"{missing_folds}"
)

print(
    f"LOCKED_COMPLETE: "
    f"{locked_complete}"
)


if (
    not locked_complete
    and
    not args.allow_partial
):

    raise RuntimeError(
        "Locked OOF requires all requested folds."
    )


if (
    not locked_complete
    and
    args.allow_partial
):

    print(
        "\nWARNING:"
    )

    print(
        "This is PARTIAL DEBUG aggregation only."
    )

    print(
        "Do NOT use these values as final OOF results."
    )


# ============================================================
# CONCATENATE
# ============================================================

pred_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

within_df = pd.concat(
    all_within,
    ignore_index=True,
)

triplet_df = pd.concat(
    all_triplets,
    ignore_index=True,
)

cross_df = pd.concat(
    all_cross,
    ignore_index=True,
)

subject_df = pd.concat(
    all_subject_geometry,
    ignore_index=True,
)

fold_metrics_df = pd.DataFrame(
    fold_summary_rows
)


# ============================================================
# GLOBAL OOF AUDIT
# ============================================================

print("\n" + "=" * 80)
print("OOF AUDIT")
print("=" * 80)


n_mri = len(
    pred_df
)

n_subjects = int(
    pred_df[
        "subject_id"
    ].nunique()
)

n_folds = int(
    pred_df[
        "fold"
    ].nunique()
)


print(
    f"MRI:      "
    f"{n_mri}"
)

print(
    f"Subjects: "
    f"{n_subjects}"
)

print(
    f"Folds:    "
    f"{sorted(pred_df['fold'].unique().tolist())}"
)


print(
    "\nMRI per fold:"
)

print(
    pred_df[
        "fold"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# COMPLETE EXPECTED TOTAL AUDIT
# ============================================================

if locked_complete:

    if n_mri != args.expected_mri:

        raise RuntimeError(
            f"Expected "
            f"{args.expected_mri} MRI "
            f"for complete OOF, "
            f"but observed {n_mri}."
        )


    if n_subjects != args.expected_subjects:

        raise RuntimeError(
            f"Expected "
            f"{args.expected_subjects} subjects "
            f"for complete OOF, "
            f"but observed {n_subjects}."
        )


    print(
        "Complete OOF MRI count audit: PASS"
    )

    print(
        "Complete OOF subject count audit: PASS"
    )


else:

    print(
        "\nPARTIAL DEBUG MODE:"
    )

    print(
        f"Observed MRI:      "
        f"{n_mri}"
    )

    print(
        f"Observed subjects: "
        f"{n_subjects}"
    )

    print(
        "Expected total counts are not enforced "
        "until all folds are available."
    )


# ============================================================
# OBSERVED FOLD COUNT
# ============================================================

if n_folds != len(
    loaded_folds
):

    raise RuntimeError(
        "Observed fold count does not match "
        "loaded-fold registry."
    )


# ============================================================
# SUBJECT-FOLD EXCLUSIVITY
# ============================================================

subject_fold_counts = (
    pred_df
    .groupby(
        "subject_id"
    )[
        "fold"
    ]
    .nunique()
)


bad_subject_fold = (
    subject_fold_counts[
        subject_fold_counts
        !=
        1
    ]
)


if len(
    bad_subject_fold
) > 0:

    raise RuntimeError(
        "Subject-fold exclusivity failed:\n"
        +
        bad_subject_fold.to_string()
    )


print(
    "Subject-fold exclusivity: PASS"
)


# ============================================================
# SUBJECT-AGE UNIQUENESS
# ============================================================

duplicate_subject_age = (
    pred_df
    .duplicated(
        subset=[
            "subject_id",
            "age_month",
        ],
        keep=False,
    )
)


if duplicate_subject_age.any():

    duplicate_table = (
        pred_df[
            duplicate_subject_age
        ][
            [
                "subject_id",
                "age_month",
                "fold",
            ]
        ]
        .sort_values(
            [
                "subject_id",
                "age_month",
            ]
        )
    )


    raise RuntimeError(
        "Duplicate subject-age observations found:\n"
        +
        duplicate_table.to_string(
            index=False
        )
    )


print(
    "Subject-age uniqueness: PASS"
)


# ============================================================
# PREDICTION FINITE AUDIT
# ============================================================

for column in [
    "age_month",
    "developmental_score",
    "age_pred",
]:

    values = pd.to_numeric(
        pred_df[
            column
        ],
        errors="coerce",
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Non-finite values found "
            f"in prediction column: "
            f"{column}"
        )


print(
    "Prediction finite-value audit: PASS"
)


# ============================================================
# PRIMARY DEVELOPMENTAL METRICS
# ============================================================

print("\n" + "=" * 80)
print("OOF DEVELOPMENTAL REPRESENTATION RESULTS")
print("=" * 80)


# ------------------------------------------------------------
# DEVELOPMENTAL AXIS VS CHRONOLOGICAL AGE
# ------------------------------------------------------------

axis_age_pearson = safe_pearson(
    pred_df[
        "age_month"
    ],
    pred_df[
        "developmental_score"
    ],
)


axis_age_spearman = safe_spearman(
    pred_df[
        "age_month"
    ],
    pred_df[
        "developmental_score"
    ],
)


# ------------------------------------------------------------
# WITHIN-SUBJECT TEMPORAL ORDER
# ------------------------------------------------------------

within_order_accuracy = float(
    within_df[
        "axis_order_correct"
    ].mean()
)


# ------------------------------------------------------------
# DELTA AXIS VS DELTA TIME
# ------------------------------------------------------------

delta_axis_time_pearson = safe_pearson(
    within_df[
        "delta_age_true"
    ],
    within_df[
        "delta_score"
    ],
)


delta_axis_time_spearman = safe_spearman(
    within_df[
        "delta_age_true"
    ],
    within_df[
        "delta_score"
    ],
)


# ------------------------------------------------------------
# SPACING DISTANCE VS DELTA TIME
# ------------------------------------------------------------

spacing_delta_time_pearson = safe_pearson(
    within_df[
        "delta_age_true"
    ],
    within_df[
        "spacing_distance"
    ],
)


spacing_delta_time_spearman = safe_spearman(
    within_df[
        "delta_age_true"
    ],
    within_df[
        "spacing_distance"
    ],
)


# ============================================================
# GLOBAL TEMPORAL SPACING
# ============================================================

triplet_df[
    "global_spacing_correct"
] = (

    (
        triplet_df[
            "d13"
        ]
        >
        triplet_df[
            "d12"
        ]
    )

    &

    (
        triplet_df[
            "d13"
        ]
        >
        triplet_df[
            "d23"
        ]
    )

).astype(
    int
)


global_spacing_accuracy = float(
    triplet_df[
        "global_spacing_correct"
    ].mean()
)


# ============================================================
# LOCAL RELATIVE TEMPORAL SPACING
# ============================================================

local_valid = (
    triplet_df[
        "delta12"
    ]
    !=
    triplet_df[
        "delta23"
    ]
)


local_triplets = (
    triplet_df[
        local_valid
    ]
    .copy()
)


if len(
    local_triplets
) == 0:

    local_spacing_accuracy = np.nan


else:

    local_triplets[
        "local_spacing_correct"
    ] = np.where(

        local_triplets[
            "delta12"
        ]
        >
        local_triplets[
            "delta23"
        ],

        (
            local_triplets[
                "d12"
            ]
            >
            local_triplets[
                "d23"
            ]
        ),

        (
            local_triplets[
                "d23"
            ]
            >
            local_triplets[
                "d12"
            ]
        ),

    ).astype(
        int
    )


    local_spacing_accuracy = float(
        local_triplets[
            "local_spacing_correct"
        ].mean()
    )


# ============================================================
# CROSS-SUBJECT DEVELOPMENTAL ORDER
# ============================================================

cross_order_accuracy = float(
    cross_df[
        "correct"
    ].mean()
)


def cross_gap_accuracy(
    gap_bin,
):

    subset = (
        cross_df[
            cross_df[
                "gap_bin"
            ]
            ==
            gap_bin
        ]
    )

    if len(
        subset
    ) == 0:

        return np.nan

    return float(
        subset[
            "correct"
        ].mean()
    )


cross_small_accuracy = (
    cross_gap_accuracy(
        0
    )
)

cross_medium_accuracy = (
    cross_gap_accuracy(
        1
    )
)

cross_large_accuracy = (
    cross_gap_accuracy(
        2
    )
)


# ============================================================
# SUBJECT-LEVEL DEVELOPMENTAL GEOMETRY
# ============================================================

subject_monotonicity = mean_or_nan(
    subject_df[
        "developmental_monotonicity"
    ]
)


subject_monotonicity_sd = std_or_nan(
    subject_df[
        "developmental_monotonicity"
    ]
)


subject_consecutive_accuracy = mean_or_nan(
    subject_df[
        "developmental_consecutive_accuracy"
    ]
)


subject_consecutive_sd = std_or_nan(
    subject_df[
        "developmental_consecutive_accuracy"
    ]
)


# ============================================================
# PRIMARY METRIC DICTIONARY
# ============================================================

primary_metrics = {

    "n_mri":
        int(
            n_mri
        ),

    "n_subjects":
        int(
            n_subjects
        ),

    "n_within_pairs":
        int(
            len(
                within_df
            )
        ),

    "n_triplets":
        int(
            len(
                triplet_df
            )
        ),

    "n_local_triplets":
        int(
            len(
                local_triplets
            )
        ),

    "n_cross_pairs":
        int(
            len(
                cross_df
            )
        ),

    "developmental_score_age_pearson":
        axis_age_pearson,

    "developmental_score_age_spearman":
        axis_age_spearman,

    "within_subject_order_accuracy":
        within_order_accuracy,

    "delta_axis_delta_time_pearson":
        delta_axis_time_pearson,

    "delta_axis_delta_time_spearman":
        delta_axis_time_spearman,

    "subject_monotonicity":
        subject_monotonicity,

    "subject_monotonicity_sd":
        subject_monotonicity_sd,

    "subject_consecutive_order_accuracy":
        subject_consecutive_accuracy,

    "subject_consecutive_order_sd":
        subject_consecutive_sd,

    "global_spacing_accuracy":
        global_spacing_accuracy,

    "local_spacing_accuracy":
        local_spacing_accuracy,

    "spacing_distance_delta_time_pearson":
        spacing_delta_time_pearson,

    "spacing_distance_delta_time_spearman":
        spacing_delta_time_spearman,

    "cross_subject_order_accuracy":
        cross_order_accuracy,

    "cross_small_accuracy":
        cross_small_accuracy,

    "cross_medium_accuracy":
        cross_medium_accuracy,

    "cross_large_accuracy":
        cross_large_accuracy,
}


for key, value in primary_metrics.items():

    if isinstance(
        value,
        float,
    ):

        print(
            f"{key:44s}: "
            f"{value:.6f}"
        )

    else:

        print(
            f"{key:44s}: "
            f"{value}"
        )


# ============================================================
# SECONDARY AGE PROBE
# ============================================================

print("\n" + "=" * 80)
print("SECONDARY AGE-PROBE OOF RESULTS")
print("=" * 80)


age_true = (
    pred_df[
        "age_month"
    ]
    .values
    .astype(float)
)


age_pred = (
    pred_df[
        "age_pred"
    ]
    .values
    .astype(float)
)


age_mae = float(
    np.mean(
        np.abs(
            age_pred
            -
            age_true
        )
    )
)


age_rmse = float(
    np.sqrt(
        np.mean(
            (
                age_pred
                -
                age_true
            )
            **
            2
        )
    )
)


ss_res = float(
    np.sum(
        (
            age_true
            -
            age_pred
        )
        **
        2
    )
)


age_mean = float(
    np.mean(
        age_true
    )
)


ss_tot = float(
    np.sum(
        (
            age_true
            -
            age_mean
        )
        **
        2
    )
)


if ss_tot > 0:

    age_r2 = float(
        1.0
        -
        (
            ss_res
            /
            ss_tot
        )
    )

else:

    age_r2 = np.nan


age_pearson = safe_pearson(
    age_true,
    age_pred,
)


age_spearman = safe_spearman(
    age_true,
    age_pred,
)


age_probe_order_accuracy = float(
    within_df[
        "age_probe_order_correct"
    ].mean()
)


delta_age_true = (
    within_df[
        "delta_age_true"
    ]
    .values
    .astype(float)
)


delta_age_pred = (
    within_df[
        "delta_age_pred"
    ]
    .values
    .astype(float)
)


delta_age_mae = float(
    np.mean(
        np.abs(
            delta_age_pred
            -
            delta_age_true
        )
    )
)


delta_age_correlation = safe_pearson(
    delta_age_true,
    delta_age_pred,
)


delta_age_spearman = safe_spearman(
    delta_age_true,
    delta_age_pred,
)


# ============================================================
# EFFECTIVE RANK
# ============================================================

if (
    "test_effective_rank"
    in fold_metrics_df.columns
):

    test_rank_mean = mean_or_nan(
        fold_metrics_df[
            "test_effective_rank"
        ]
    )

    test_rank_sd = std_or_nan(
        fold_metrics_df[
            "test_effective_rank"
        ]
    )

else:

    test_rank_mean = np.nan

    test_rank_sd = np.nan


secondary_metrics = {

    "age_mae":
        age_mae,

    "age_rmse":
        age_rmse,

    "age_r2":
        age_r2,

    "age_correlation":
        age_pearson,

    "age_spearman":
        age_spearman,

    "age_probe_order_accuracy":
        age_probe_order_accuracy,

    "delta_age_mae":
        delta_age_mae,

    "delta_age_correlation":
        delta_age_correlation,

    "delta_age_spearman":
        delta_age_spearman,

    "test_effective_rank_mean":
        test_rank_mean,

    "test_effective_rank_sd":
        test_rank_sd,
}


for key, value in secondary_metrics.items():

    print(
        f"{key:44s}: "
        f"{value:.6f}"
    )


# ============================================================
# FOLD-LEVEL TABLE
# ============================================================

print("\n" + "=" * 80)
print("FOLD-LEVEL DEVELOPMENTAL METRICS")
print("=" * 80)


fold_display_columns = [

    "fold",

    "unified_within_order_accuracy",

    "cross_subject_order_accuracy",

    "cross_small_accuracy",

    "cross_medium_accuracy",

    "cross_large_accuracy",

    "subject_developmental_monotonicity",

    "subject_consecutive_order_accuracy",

    "global_spacing_accuracy",

    "local_spacing_accuracy",

    "developmental_score_age_pearson",

    "developmental_score_age_spearman",

    "developmental_delta_time_pearson",

    "spacing_distance_delta_time_pearson",

    "age_mae",

    "age_r2",

    "test_effective_rank",
]


available_fold_columns = [

    column

    for column
    in fold_display_columns

    if column
    in fold_metrics_df.columns
]


if available_fold_columns:

    print(
        fold_metrics_df[
            available_fold_columns
        ]
        .sort_values(
            "fold"
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# SUBJECT-LEVEL TABLE
# ============================================================

print("\n" + "=" * 80)
print("SUBJECT-LEVEL DEVELOPMENTAL GEOMETRY")
print("=" * 80)


subject_table_columns = [

    "subject_id",
    "fold",
    "n_scans",
    "developmental_monotonicity",
    "developmental_consecutive_accuracy",
    "age_probe_monotonicity",
    "age_probe_consecutive_accuracy",
]


subject_table = (

    subject_df[
        subject_table_columns
    ]
    .sort_values(
        [
            "fold",
            "subject_id",
        ]
    )
)


print(
    subject_table.to_string(
        index=False
    )
)


# ============================================================
# SUBJECT-LEVEL CONSECUTIVE ORDER VS CHANCE
# ============================================================

subject_consecutive_values = (

    subject_df[
        "developmental_consecutive_accuracy"
    ]
    .dropna()
    .values
    .astype(float)
)


wilcoxon_statistic = np.nan
wilcoxon_p_value = np.nan


if len(
    subject_consecutive_values
) >= 5:

    differences = (
        subject_consecutive_values
        -
        0.5
    )


    if np.allclose(
        differences,
        0,
    ):

        wilcoxon_statistic = np.nan
        wilcoxon_p_value = 1.0


    else:

        try:

            (
                wilcoxon_statistic,
                wilcoxon_p_value,
            ) = wilcoxon(

                differences,

                alternative="greater",
            )


        except ValueError:

            wilcoxon_statistic = np.nan
            wilcoxon_p_value = np.nan


print("\n" + "=" * 80)
print("SUBJECT-LEVEL CONSECUTIVE ORDER VS CHANCE")
print("=" * 80)


print(
    f"N subjects:             "
    f"{len(subject_consecutive_values)}"
)

print(
    f"Mean consecutive acc:   "
    f"{mean_or_nan(subject_consecutive_values):.6f}"
)

print(
    f"SD consecutive acc:     "
    f"{std_or_nan(subject_consecutive_values):.6f}"
)

print(
    f"Wilcoxon statistic:     "
    f"{wilcoxon_statistic}"
)

print(
    f"One-sided p-value:      "
    f"{wilcoxon_p_value}"
)


# ============================================================
# DESCRIPTIVE DEVELOPMENTAL STRUCTURE SCORE
# ============================================================

"""
This composite is descriptive only.

It is NOT:
- a training objective
- a statistical test
- a replacement for individual metrics

Age prediction is intentionally excluded.
"""

developmental_structure_score = (

    0.25
    *
    within_order_accuracy

    +

    0.25
    *
    cross_order_accuracy

    +

    0.20
    *
    subject_monotonicity

    +

    0.15
    *
    global_spacing_accuracy

    +

    0.15
    *
    local_spacing_accuracy
)


print("\n" + "=" * 80)
print("DEVELOPMENTAL STRUCTURE SCORE")
print("=" * 80)


print(
    f"Developmental structure score: "
    f"{developmental_structure_score:.6f}"
)

print(
    "Age MAE/R2 are excluded from "
    "this descriptive score."
)


# ============================================================
# TABLE 1 — PRIMARY REPRESENTATION
# ============================================================

table1 = pd.DataFrame(
    [
        {

            "Model":
                "TempoDINO-v7",

            "Checkpoint":
                CHECKPOINT_NAME,

            "Locked Complete":
                locked_complete,

            "MRI":
                n_mri,

            "Subjects":
                n_subjects,

            "Within Order Acc":
                within_order_accuracy,

            "Cross-subject Order Acc":
                cross_order_accuracy,

            "Small-gap Cross Acc":
                cross_small_accuracy,

            "Medium-gap Cross Acc":
                cross_medium_accuracy,

            "Large-gap Cross Acc":
                cross_large_accuracy,

            "Subject Monotonicity":
                subject_monotonicity,

            "Consecutive Order Acc":
                subject_consecutive_accuracy,

            "Global Spacing Acc":
                global_spacing_accuracy,

            "Local Spacing Acc":
                local_spacing_accuracy,

            "Axis-Age Pearson":
                axis_age_pearson,

            "Axis-Age Spearman":
                axis_age_spearman,

            "DeltaAxis-DeltaTime Pearson":
                delta_axis_time_pearson,

            "Spacing-DeltaTime Pearson":
                spacing_delta_time_pearson,

            "Developmental Structure Score":
                developmental_structure_score,
        }
    ]
)


# ============================================================
# TABLE 2 — SECONDARY AGE PROBE
# ============================================================

table2 = pd.DataFrame(
    [
        {

            "Model":
                "TempoDINO-v7",

            "Checkpoint":
                CHECKPOINT_NAME,

            "Locked Complete":
                locked_complete,

            "Age MAE":
                age_mae,

            "Age RMSE":
                age_rmse,

            "Age R2":
                age_r2,

            "Age Pearson":
                age_pearson,

            "Age Spearman":
                age_spearman,

            "Age-probe Order Acc":
                age_probe_order_accuracy,

            "Delta-Age MAE":
                delta_age_mae,

            "Delta-Age Pearson":
                delta_age_correlation,

            "Delta-Age Spearman":
                delta_age_spearman,

            "Effective Rank Mean":
                test_rank_mean,

            "Effective Rank SD":
                test_rank_sd,
        }
    ]
)


# ============================================================
# SAVE RAW OOF TABLES
# ============================================================

pred_df.to_csv(
    OOF_PREDICTIONS_PATH,
    index=False,
)

within_df.to_csv(
    OOF_WITHIN_PATH,
    index=False,
)

triplet_df.to_csv(
    OOF_TRIPLETS_PATH,
    index=False,
)

cross_df.to_csv(
    OOF_CROSS_PATH,
    index=False,
)

subject_df.to_csv(
    OOF_SUBJECT_GEOMETRY_PATH,
    index=False,
)

fold_metrics_df.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)

table1.to_csv(
    TABLE1_PATH,
    index=False,
)

table2.to_csv(
    TABLE2_PATH,
    index=False,
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_flat = {

    "model":
        "TempoDINO-v7",

    "checkpoint":
        CHECKPOINT_NAME,

    "locked_complete":
        locked_complete,

    "partial_debug_mode":
        (
            args.allow_partial
            and
            not locked_complete
        ),

    "loaded_folds":
        ",".join(
            str(
                x
            )
            for x in loaded_folds
        ),

    "missing_folds":
        ",".join(
            str(
                x
            )
            for x in missing_folds
        ),

    "developmental_structure_score":
        developmental_structure_score,

    **primary_metrics,

    **secondary_metrics,
}


pd.DataFrame(
    [
        summary_flat
    ]
).to_csv(
    OOF_SUMMARY_PATH,
    index=False,
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "model":
        "TempoDINO-v7",

    "checkpoint":
        CHECKPOINT_NAME,

    "checkpoint_policy":
        (
            "same v7_final checkpoint "
            "used for every fold"
        ),

    "primary_task":
        (
            "longitudinal developmental "
            "representation learning"
        ),

    "age_prediction_role":
        (
            "secondary downstream probe"
        ),

    "requested_folds":
        [
            int(
                x
            )
            for x in args.folds
        ],

    "loaded_folds":
        [
            int(
                x
            )
            for x in loaded_folds
        ],

    "missing_folds":
        [
            int(
                x
            )
            for x in missing_folds
        ],

    "locked_complete":
        bool(
            locked_complete
        ),

    "partial_debug_mode":
        bool(
            args.allow_partial
            and
            not locked_complete
        ),

    "n_mri":
        int(
            n_mri
        ),

    "n_subjects":
        int(
            n_subjects
        ),

    "n_folds_loaded":
        int(
            n_folds
        ),

    "n_within_pairs":
        int(
            len(
                within_df
            )
        ),

    "n_triplets":
        int(
            len(
                triplet_df
            )
        ),

    "n_cross_pairs":
        int(
            len(
                cross_df
            )
        ),

    "primary_metrics":
        {

            key:
                (
                    int(
                        value
                    )

                    if isinstance(
                        value,
                        (
                            int,
                            np.integer,
                        ),
                    )

                    else
                    json_safe_float(
                        value
                    )
                )

            for key, value
            in primary_metrics.items()
        },

    "secondary_age_probe":
        {

            key:
                json_safe_float(
                    value
                )

            for key, value
            in secondary_metrics.items()
        },

    "developmental_structure_score":
        json_safe_float(
            developmental_structure_score
        ),

    "subject_consecutive_order_vs_chance":
        {

            "n_subjects":
                int(
                    len(
                        subject_consecutive_values
                    )
                ),

            "mean":
                json_safe_float(
                    mean_or_nan(
                        subject_consecutive_values
                    )
                ),

            "sd":
                json_safe_float(
                    std_or_nan(
                        subject_consecutive_values
                    )
                ),

            "wilcoxon_statistic":
                json_safe_float(
                    wilcoxon_statistic
                ),

            "one_sided_p_value":
                json_safe_float(
                    wilcoxon_p_value
                ),
        },
}


with open(
    OOF_SUMMARY_JSON_PATH,
    "w",
) as handle:

    json.dump(
        summary,
        handle,
        indent=2,
    )


# ============================================================
# AUDIT JSON
# ============================================================

audit = {

    "checkpoint":
        CHECKPOINT_NAME,

    "requested_folds":
        [
            int(
                x
            )
            for x in args.folds
        ],

    "loaded_folds":
        [
            int(
                x
            )
            for x in loaded_folds
        ],

    "missing_folds":
        [
            int(
                x
            )
            for x in missing_folds
        ],

    "missing_files_by_fold":
        missing_files_by_fold,

    "locked_complete":
        bool(
            locked_complete
        ),

    "allow_partial":
        bool(
            args.allow_partial
        ),

    "expected_mri":
        int(
            args.expected_mri
        ),

    "observed_mri":
        int(
            n_mri
        ),

    "expected_subjects":
        int(
            args.expected_subjects
        ),

    "observed_subjects":
        int(
            n_subjects
        ),

    "observed_folds":
        sorted(
            [
                int(
                    x
                )
                for x in pred_df[
                    "fold"
                ].unique()
            ]
        ),

    "subject_fold_exclusivity":
        True,

    "subject_age_uniqueness":
        True,

    "prediction_finite_values":
        True,

    "checkpoint_locked_across_folds":
        CHECKPOINT_NAME,
}


with open(
    AUDIT_PATH,
    "w",
) as handle:

    json.dump(
        audit,
        handle,
        indent=2,
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("FINAL AGGREGATION STATUS")
print("=" * 80)


if locked_complete:

    print(
        "LOCKED_COMPLETE = TRUE"
    )

    print(
        "This aggregation contains all requested folds."
    )

else:

    print(
        "LOCKED_COMPLETE = FALSE"
    )

    print(
        "PARTIAL DEBUG RESULT ONLY."
    )

    print(
        "Do not use this output as the final OOF result."
    )


# ============================================================
# FILES WRITTEN
# ============================================================

print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)


for path in [

    OOF_PREDICTIONS_PATH,
    OOF_WITHIN_PATH,
    OOF_TRIPLETS_PATH,
    OOF_CROSS_PATH,
    OOF_SUBJECT_GEOMETRY_PATH,
    FOLD_METRICS_PATH,
    OOF_SUMMARY_PATH,
    OOF_SUMMARY_JSON_PATH,
    TABLE1_PATH,
    TABLE2_PATH,
    AUDIT_PATH,

]:

    print(
        path
    )


print("\nDONE.")
