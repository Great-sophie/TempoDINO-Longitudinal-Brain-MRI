#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19b_probe_tempodino_v7_unified_axis.py

TempoDINO-v7 unified developmental-axis evaluation.

Primary representation evaluation
---------------------------------
1. Unified developmental-axis within-subject order
2. Unified developmental-axis cross-subject order
3. Subject-level developmental-axis monotonicity
4. Consecutive-order accuracy
5. Global relative-spacing accuracy
6. Local relative-spacing accuracy
7. Spacing distance / delta-time correlation
8. Unified developmental score / chronological-age correlation

Secondary probes
----------------
1. Frozen-encoder age linear probe
2. Age MAE / RMSE / R2 / Pearson / Spearman
3. Age-probe longitudinal ordering
4. Effective rank / feature std / cosine similarity

Comparison
----------
- TempoDINO-v6 epoch2
- TempoDINO-v7 epoch1
- TempoDINO-v7 epoch2
- TempoDINO-v7 epoch3/final

No model is trained in this script.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from scipy.stats import (
    pearsonr,
    spearmanr,
)


# ============================================================
# PATHS
# ============================================================

import os

RELEASE_DIR = Path(__file__).resolve().parents[1]

PROJECT_DIR = Path(
    os.environ.get("TEMPODINO_PROJECT_DIR", RELEASE_DIR)
).expanduser()

RUN_ROOT = Path(
    os.environ.get(
        "TEMPODINO_RUN_ROOT",
        str(RELEASE_DIR / "local_runs"),
    )
).expanduser()

PROCESSED_ROOT = Path(
    os.environ.get(
        "TEMPODINO_PROCESSED_ROOT",
        str(PROJECT_DIR.parent / "monkey_development_preprocessed_96"),
    )
).expanduser()

META_PATH = PROJECT_DIR / "metadata" / "folds_balanced_preprocessed96.csv"

V6_CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "tempodino_v6_crossordinal"
V7_CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "tempodino_v7_unified_axis"

RESULT_ROOT = RUN_ROOT / "results" / "tempodino_v7_unified_axis_probe"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_preprocessed_path(old_path):
    old_path = Path(str(old_path))
    marker = "monkey_development_preprocessed_96"

    if marker not in old_path.parts:
        raise ValueError(f"Unexpected MRI path: {old_path}")

    index = old_path.parts.index(marker)
    relative = Path(*old_path.parts[index + 1:])
    resolved = PROCESSED_ROOT / relative

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    return str(resolved)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fold",
    type=int,
    default=0,
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="all",
    choices=[
        "v6_epoch2",
        "v7_epoch1",
        "v7_epoch2",
        "v7_epoch3",
        "v7_final",
        "all",
    ],
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
)

parser.add_argument(
    "--ridge_alpha",
    type=float,
    default=10.0,
)

args = parser.parse_args()


if args.fold not in [0, 1, 2, 3, 4]:
    raise ValueError("--fold must be 0-4")


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("TEMPODINO V7 UNIFIED DEVELOPMENTAL AXIS PROBE")
print("=" * 80)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# HELPERS
# ============================================================

def normalize_subject_id(x):
    return str(x).zfill(3)


def safe_pearson(x, y):

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

    x = x[valid]
    y = y[valid]

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


def safe_spearman(x, y):

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

    x = x[valid]
    y = y[valid]

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


# ============================================================
# NETWORK
# ============================================================

def make_group_norm(channels):

    if channels >= 32:
        groups = 8
    elif channels >= 16:
        groups = 4
    else:
        groups = 1

    return nn.GroupNorm(
        groups,
        channels,
    )


class BasicBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):

        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.norm1 = make_group_norm(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.norm2 = make_group_norm(
            out_channels
        )

        self.downsample = None

        if (
            stride != 1
            or
            in_channels != out_channels
        ):

            self.downsample = nn.Sequential(

                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),

                make_group_norm(
                    out_channels
                ),
            )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet18Encoder3D(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
    ):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv3d(
                1,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),

            make_group_norm(32),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.layer1 = self._make_layer(
            32, 32, 2, 1
        )

        self.layer2 = self._make_layer(
            32, 64, 2, 2
        )

        self.layer3 = self._make_layer(
            64, 128, 2, 2
        )

        self.layer4 = self._make_layer(
            128, 256, 2, 2
        )

        self.pool = nn.AdaptiveAvgPool3d(
            1
        )

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                embedding_dim,
            ),

            nn.LayerNorm(
                embedding_dim
            ),
        )

    def _make_layer(
        self,
        in_channels,
        out_channels,
        blocks,
        stride,
    ):

        layers = [

            BasicBlock3D(
                in_channels,
                out_channels,
                stride,
            )
        ]

        for _ in range(
            1,
            blocks,
        ):

            layers.append(

                BasicBlock3D(
                    out_channels,
                    out_channels,
                    1,
                )
            )

        return nn.Sequential(
            *layers
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.embedding(x)

        return x


class DevelopmentalHead(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
    ):

        super().__init__()

        self.head = nn.Sequential(

            nn.Linear(
                embedding_dim,
                64,
            ),

            nn.GELU(),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, z):

        return (
            self.head(z)
            .squeeze(-1)
        )


class RelativeSpacingProjector(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        spacing_dim=64,
    ):

        super().__init__()

        self.projector = nn.Sequential(

            nn.Linear(
                embedding_dim,
                128,
            ),

            nn.GELU(),

            nn.Linear(
                128,
                spacing_dim,
            ),
        )

    def forward(self, z):

        q = self.projector(z)

        return F.normalize(
            q,
            p=2,
            dim=-1,
        )


# ============================================================
# DATASET
# ============================================================

class MRIDataset(Dataset):

    def __init__(
        self,
        dataframe,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
            .copy()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[index]

        volume = np.load(
            row[
                "preprocessed_path"
            ]
        ).astype(
            np.float32
        )

        image = (
            torch.from_numpy(
                volume
            )
            .unsqueeze(0)
        )

        return {
            "image":
                image,

            "index":
                int(index),
        }


# ============================================================
# REPRESENTATION METRICS
# ============================================================

def effective_rank(
    features,
    eps=1e-12,
):

    x = np.asarray(
        features,
        dtype=np.float64,
    )

    x = (
        x
        -
        x.mean(
            axis=0,
            keepdims=True,
        )
    )

    s = np.linalg.svd(
        x,
        compute_uv=False,
    )

    eigenvalues = s ** 2

    total = eigenvalues.sum()

    if total <= eps:
        return 1.0

    p = (
        eigenvalues
        /
        total
    )

    entropy = -np.sum(
        p
        *
        np.log(
            p + eps
        )
    )

    return float(
        np.exp(
            entropy
        )
    )


def mean_dimension_std(features):

    return float(
        np.asarray(
            features,
            dtype=float,
        )
        .std(
            axis=0
        )
        .mean()
    )


def mean_offdiag_cosine(features):

    x = np.asarray(
        features,
        dtype=float,
    )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    x = (
        x
        /
        np.clip(
            norms,
            1e-12,
            None,
        )
    )

    sim = x @ x.T

    n = len(x)

    if n <= 1:
        return np.nan

    mask = ~np.eye(
        n,
        dtype=bool,
    )

    return float(
        sim[
            mask
        ].mean()
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(
    encoder,
    developmental_head,
    spacing_projector,
    dataframe,
):

    loader = DataLoader(
        MRIDataset(
            dataframe
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    features = []
    developmental_scores = []
    spacing_embeddings = []
    indices = []

    encoder.eval()
    developmental_head.eval()
    spacing_projector.eval()

    with torch.no_grad():

        for batch in loader:

            x = (
                batch[
                    "image"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):

                z = encoder(x)

                score = (
                    developmental_head(z)
                )

                q = (
                    spacing_projector(z)
                )

            features.append(
                z.float()
                .cpu()
                .numpy()
            )

            developmental_scores.append(
                score.float()
                .cpu()
                .numpy()
            )

            spacing_embeddings.append(
                q.float()
                .cpu()
                .numpy()
            )

            indices.extend(
                batch[
                    "index"
                ]
                .cpu()
                .numpy()
                .tolist()
            )

    X = np.concatenate(
        features,
        axis=0,
    )

    scores = np.concatenate(
        developmental_scores,
        axis=0,
    )

    spacing = np.concatenate(
        spacing_embeddings,
        axis=0,
    )

    if not np.array_equal(
        np.asarray(indices),
        np.arange(
            len(dataframe)
        ),
    ):

        raise RuntimeError(
            "Feature extraction order mismatch."
        )

    return (
        X,
        scores,
        spacing,
    )


# ============================================================
# PAIRS
# ============================================================

def build_within_pairs(
    dataframe,
):

    rows = []

    for (
        subject_id,
        group
    ) in dataframe.groupby(
        "subject_id"
    ):

        group = (
            group
            .sort_values(
                "age_month"
            )
            .reset_index(
                drop=True
            )
        )

        for i in range(
            len(group)
        ):

            for j in range(
                i + 1,
                len(group),
            ):

                age1 = float(
                    group.loc[
                        i,
                        "age_month"
                    ]
                )

                age2 = float(
                    group.loc[
                        j,
                        "age_month"
                    ]
                )

                if age2 <= age1:
                    continue

                rows.append(
                    {
                        "subject_id":
                            subject_id,

                        "idx1":
                            int(
                                group.loc[
                                    i,
                                    "_row_index"
                                ]
                            ),

                        "idx2":
                            int(
                                group.loc[
                                    j,
                                    "_row_index"
                                ]
                            ),

                        "age1":
                            age1,

                        "age2":
                            age2,

                        "delta_age_true":
                            age2 - age1,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TRIPLETS
# ============================================================

def build_triplets(
    dataframe,
):

    rows = []

    for (
        subject_id,
        group
    ) in dataframe.groupby(
        "subject_id"
    ):

        group = (
            group
            .sort_values(
                "age_month"
            )
            .reset_index(
                drop=True
            )
        )

        n = len(group)

        for i in range(n):

            for j in range(
                i + 1,
                n,
            ):

                for k in range(
                    j + 1,
                    n,
                ):

                    age1 = float(
                        group.loc[
                            i,
                            "age_month"
                        ]
                    )

                    age2 = float(
                        group.loc[
                            j,
                            "age_month"
                        ]
                    )

                    age3 = float(
                        group.loc[
                            k,
                            "age_month"
                        ]
                    )

                    if not (
                        age1
                        <
                        age2
                        <
                        age3
                    ):
                        continue

                    rows.append(
                        {
                            "subject_id":
                                subject_id,

                            "idx1":
                                int(
                                    group.loc[
                                        i,
                                        "_row_index"
                                    ]
                                ),

                            "idx2":
                                int(
                                    group.loc[
                                        j,
                                        "_row_index"
                                    ]
                                ),

                            "idx3":
                                int(
                                    group.loc[
                                        k,
                                        "_row_index"
                                    ]
                                ),

                            "age1":
                                age1,

                            "age2":
                                age2,

                            "age3":
                                age3,

                            "delta12":
                                age2 - age1,

                            "delta23":
                                age3 - age2,

                            "delta13":
                                age3 - age1,
                        }
                    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CROSS-SUBJECT PAIRS
# ============================================================

def build_cross_pairs(
    dataframe,
):

    rows = []

    n = len(
        dataframe
    )

    for i in range(n):

        row_i = dataframe.iloc[i]

        subject_i = str(
            row_i[
                "subject_id"
            ]
        )

        age_i = float(
            row_i[
                "age_month"
            ]
        )

        for j in range(
            i + 1,
            n,
        ):

            row_j = dataframe.iloc[j]

            subject_j = str(
                row_j[
                    "subject_id"
                ]
            )

            if subject_i == subject_j:
                continue

            age_j = float(
                row_j[
                    "age_month"
                ]
            )

            gap = abs(
                age_j - age_i
            )

            if gap < 2.0:
                continue

            if gap < 4.0:
                gap_bin = 0

            elif gap < 8.0:
                gap_bin = 1

            else:
                gap_bin = 2

            if age_i < age_j:

                younger_idx = i
                older_idx = j

                younger_age = age_i
                older_age = age_j

            else:

                younger_idx = j
                older_idx = i

                younger_age = age_j
                older_age = age_i

            rows.append(
                {
                    "younger_idx":
                        int(
                            younger_idx
                        ),

                    "older_idx":
                        int(
                            older_idx
                        ),

                    "younger_age":
                        younger_age,

                    "older_age":
                        older_age,

                    "age_gap":
                        gap,

                    "gap_bin":
                        gap_bin,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUBJECT-LEVEL GEOMETRY
# ============================================================

def compute_subject_geometry(
    dataframe,
):

    rows = []

    for (
        subject_id,
        group
    ) in dataframe.groupby(
        "subject_id"
    ):

        group = (
            group
            .sort_values(
                "age_month"
            )
            .reset_index(
                drop=True
            )
        )

        ages = (
            group[
                "age_month"
            ]
            .values
            .astype(float)
        )

        developmental = (
            group[
                "developmental_score"
            ]
            .values
            .astype(float)
        )

        age_pred = (
            group[
                "age_pred"
            ]
            .values
            .astype(float)
        )

        developmental_monotonicity = (
            safe_spearman(
                ages,
                developmental,
            )
        )

        age_probe_monotonicity = (
            safe_spearman(
                ages,
                age_pred,
            )
        )

        developmental_consecutive = []

        age_probe_consecutive = []

        for i in range(
            len(group) - 1
        ):

            developmental_consecutive.append(
                int(
                    developmental[
                        i + 1
                    ]
                    >
                    developmental[
                        i
                    ]
                )
            )

            age_probe_consecutive.append(
                int(
                    age_pred[
                        i + 1
                    ]
                    >
                    age_pred[
                        i
                    ]
                )
            )

        rows.append(
            {
                "subject_id":
                    subject_id,

                "n_scans":
                    len(group),

                "developmental_monotonicity":
                    developmental_monotonicity,

                "developmental_consecutive_accuracy":
                    (
                        float(
                            np.mean(
                                developmental_consecutive
                            )
                        )
                        if developmental_consecutive
                        else np.nan
                    ),

                "age_probe_monotonicity":
                    age_probe_monotonicity,

                "age_probe_consecutive_accuracy":
                    (
                        float(
                            np.mean(
                                age_probe_consecutive
                            )
                        )
                        if age_probe_consecutive
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# METADATA
# ============================================================

if not META_PATH.exists():
    raise FileNotFoundError(
        META_PATH
    )


df = pd.read_csv(
    META_PATH,
    dtype={
        "subject_id": str
    },
)


df["preprocessed_path"] = df["preprocessed_path"].map(
    resolve_preprocessed_path
)


df[
    "subject_id"
] = (
    df[
        "subject_id"
    ]
    .map(
        normalize_subject_id
    )
)


train_df = (
    df[
        df[
            "fold"
        ]
        !=
        args.fold
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


test_df = (
    df[
        df[
            "fold"
        ]
        ==
        args.fold
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


train_subjects = set(
    train_df[
        "subject_id"
    ]
)


test_subjects = set(
    test_df[
        "subject_id"
    ]
)


if (
    train_subjects
    &
    test_subjects
):

    raise RuntimeError(
        "Subject leakage detected."
    )


print("\n" + "=" * 80)
print(
    f"V7 UNIFIED AXIS PROBE — FOLD {args.fold}"
)
print("=" * 80)

print(
    f"Train MRI:          {len(train_df)}"
)

print(
    f"Train subjects:     {len(train_subjects)}"
)

print(
    f"Held-out MRI:       {len(test_df)}"
)

print(
    f"Held-out subjects:  {len(test_subjects)}"
)

print(
    "Subject leakage audit: PASS"
)


# ============================================================
# CHECKPOINT MAP
# ============================================================

v6_dir = (
    V6_CHECKPOINT_ROOT
    /
    f"fold_{args.fold}"
)


v7_dir = (
    V7_CHECKPOINT_ROOT
    /
    f"fold_{args.fold}"
)


checkpoint_map = {

    "v6_epoch2":
        v6_dir
        /
        "epoch_2.pt",

    "v7_epoch1":
        v7_dir
        /
        "epoch_1.pt",

    "v7_epoch2":
        v7_dir
        /
        "epoch_2.pt",

    "v7_epoch3":
        v7_dir
        /
        "epoch_3.pt",

    "v7_final":
        v7_dir
        /
        "final.pt",
}


if args.checkpoint == "all":

    checkpoint_names = [
        "v6_epoch2",
        "v7_epoch1",
        "v7_epoch2",
        "v7_epoch3",
        "v7_final",
    ]

else:

    checkpoint_names = [
        args.checkpoint
    ]


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_models(
    checkpoint_name,
    checkpoint_path,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    embedding_dim = int(
        checkpoint.get(
            "embedding_dim",
            checkpoint.get(
                "config",
                {}
            ).get(
                "embedding_dim",
                256,
            )
        )
    )

    spacing_dim = int(
        checkpoint.get(
            "spacing_dim",
            checkpoint.get(
                "config",
                {}
            ).get(
                "spacing_dim",
                64,
            )
        )
    )

    encoder = ResNet18Encoder3D(
        embedding_dim
    )

    encoder.load_state_dict(
        checkpoint[
            "student_encoder_state_dict"
        ],
        strict=True,
    )


    developmental_head = (
        DevelopmentalHead(
            embedding_dim
        )
    )

    # ========================================================
    # V7 unified head
    # ========================================================

    if (
        "unified_developmental_head_state_dict"
        in checkpoint
    ):

        developmental_head.load_state_dict(
            checkpoint[
                "unified_developmental_head_state_dict"
            ],
            strict=True,
        )

        head_source = (
            "v7_unified"
        )

    elif (
        "developmental_head_state_dict"
        in checkpoint
    ):

        developmental_head.load_state_dict(
            checkpoint[
                "developmental_head_state_dict"
            ],
            strict=True,
        )

        head_source = (
            "v7_developmental_alias"
        )

    # ========================================================
    # V6 baseline:
    # use cross-subject head as the global developmental score.
    #
    # It is more appropriate for cross-subject calibration than
    # the within-only temporal head.
    # ========================================================

    elif (
        "cross_subject_head_state_dict"
        in checkpoint
    ):

        developmental_head.load_state_dict(
            checkpoint[
                "cross_subject_head_state_dict"
            ],
            strict=True,
        )

        head_source = (
            "v6_cross_subject_head"
        )

    elif (
        "temporal_head_state_dict"
        in checkpoint
    ):

        developmental_head.load_state_dict(
            checkpoint[
                "temporal_head_state_dict"
            ],
            strict=True,
        )

        head_source = (
            "v6_temporal_head"
        )

    else:

        raise KeyError(
            "No developmental scalar head found."
        )


    spacing_projector = (
        RelativeSpacingProjector(
            embedding_dim,
            spacing_dim,
        )
    )

    spacing_projector.load_state_dict(
        checkpoint[
            "spacing_projector_state_dict"
        ],
        strict=True,
    )

    return (
        checkpoint,
        encoder,
        developmental_head,
        spacing_projector,
        embedding_dim,
        spacing_dim,
        head_source,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_checkpoint(
    checkpoint_name,
    checkpoint_path,
):

    print("\n" + "=" * 80)
    print(
        f"CHECKPOINT: {checkpoint_name.upper()}"
    )
    print("=" * 80)

    (
        checkpoint,
        encoder,
        developmental_head,
        spacing_projector,
        embedding_dim,
        spacing_dim,
        head_source,
    ) = load_models(
        checkpoint_name,
        checkpoint_path,
    )

    print(
        checkpoint_path
    )

    print(
        f"Epoch:      "
        f"{checkpoint.get('epoch', 'NA')}"
    )

    print(
        f"Mode:       "
        f"{checkpoint.get('mode', 'NA')}"
    )

    print(
        f"Axis source:"
        f" {head_source}"
    )

    print(
        f"Embedding:  "
        f"{embedding_dim}"
    )

    print(
        f"Spacing:    "
        f"{spacing_dim}"
    )

    encoder = encoder.to(
        device
    )

    developmental_head = (
        developmental_head
        .to(device)
    )

    spacing_projector = (
        spacing_projector
        .to(device)
    )


    # ========================================================
    # EXTRACT
    # ========================================================

    print(
        "\nExtracting training representations..."
    )

    (
        X_train,
        train_score,
        train_spacing,
    ) = extract_features(
        encoder,
        developmental_head,
        spacing_projector,
        train_df,
    )

    print(
        "Extracting held-out representations..."
    )

    (
        X_test,
        test_score,
        test_spacing,
    ) = extract_features(
        encoder,
        developmental_head,
        spacing_projector,
        test_df,
    )

    print(
        f"Train features: {X_train.shape}"
    )

    print(
        f"Test features:  {X_test.shape}"
    )


    # ========================================================
    # REPRESENTATION DIAGNOSTICS
    # ========================================================

    train_rank = effective_rank(
        X_train
    )

    test_rank = effective_rank(
        X_test
    )

    train_std = mean_dimension_std(
        X_train
    )

    test_std = mean_dimension_std(
        X_test
    )

    train_cos = mean_offdiag_cosine(
        X_train
    )

    test_cos = mean_offdiag_cosine(
        X_test
    )

    print("\n" + "-" * 80)
    print("REPRESENTATION DIAGNOSTICS")
    print("-" * 80)

    print(
        f"Train effective rank: {train_rank:.4f}"
    )

    print(
        f"Test effective rank:  {test_rank:.4f}"
    )

    print(
        f"Train mean dim std:   {train_std:.4f}"
    )

    print(
        f"Test mean dim std:    {test_std:.4f}"
    )

    print(
        f"Train mean cosine:    {train_cos:.4f}"
    )

    print(
        f"Test mean cosine:     {test_cos:.4f}"
    )


    # ========================================================
    # SECONDARY AGE LINEAR PROBE
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    y_train = (
        train_df[
            "age_month"
        ]
        .values
        .astype(float)
    )

    y_test = (
        test_df[
            "age_month"
        ]
        .values
        .astype(float)
    )

    ridge = Ridge(
        alpha=args.ridge_alpha
    )

    ridge.fit(
        X_train_scaled,
        y_train,
    )

    age_pred = ridge.predict(
        X_test_scaled
    )

    age_mae = float(
        mean_absolute_error(
            y_test,
            age_pred,
        )
    )

    age_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                age_pred,
            )
        )
    )

    age_r2 = float(
        r2_score(
            y_test,
            age_pred,
        )
    )

    age_r = safe_pearson(
        y_test,
        age_pred,
    )

    age_rho = safe_spearman(
        y_test,
        age_pred,
    )

    print("\n" + "-" * 80)
    print("SECONDARY AGE LINEAR PROBE")
    print("-" * 80)

    print(
        f"MAE:          {age_mae:.4f}"
    )

    print(
        f"RMSE:         {age_rmse:.4f}"
    )

    print(
        f"R2:           {age_r2:.4f}"
    )

    print(
        f"Pearson r:    {age_r:.4f}"
    )

    print(
        f"Spearman rho: {age_rho:.4f}"
    )


    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    pred_df = (
        test_df[
            [
                "subject_id",
                "age_month",
                "fold",
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    pred_df[
        "_row_index"
    ] = np.arange(
        len(pred_df)
    )

    pred_df[
        "developmental_score"
    ] = test_score

    pred_df[
        "age_pred"
    ] = age_pred


    # ========================================================
    # DEVELOPMENTAL SCORE / AGE CORRELATION
    # ========================================================

    score_age_r = safe_pearson(
        pred_df[
            "age_month"
        ],
        pred_df[
            "developmental_score"
        ],
    )

    score_age_rho = safe_spearman(
        pred_df[
            "age_month"
        ],
        pred_df[
            "developmental_score"
        ],
    )

    print("\n" + "-" * 80)
    print("UNIFIED DEVELOPMENTAL AXIS")
    print("-" * 80)

    print(
        f"Score / age Pearson:  "
        f"{score_age_r:.4f}"
    )

    print(
        f"Score / age Spearman: "
        f"{score_age_rho:.4f}"
    )


    # ========================================================
    # WITHIN-SUBJECT
    # ========================================================

    within_df = build_within_pairs(
        pred_df
    )

    score1 = []
    score2 = []

    age_pred1 = []
    age_pred2 = []

    spacing_distance = []

    for _, row in within_df.iterrows():

        i = int(
            row[
                "idx1"
            ]
        )

        j = int(
            row[
                "idx2"
            ]
        )

        score1.append(
            float(
                test_score[i]
            )
        )

        score2.append(
            float(
                test_score[j]
            )
        )

        age_pred1.append(
            float(
                age_pred[i]
            )
        )

        age_pred2.append(
            float(
                age_pred[j]
            )
        )

        spacing_distance.append(
            float(
                np.linalg.norm(
                    test_spacing[j]
                    -
                    test_spacing[i]
                )
            )
        )


    within_df[
        "score1"
    ] = score1

    within_df[
        "score2"
    ] = score2

    within_df[
        "delta_score"
    ] = (
        within_df[
            "score2"
        ]
        -
        within_df[
            "score1"
        ]
    )

    within_df[
        "axis_order_correct"
    ] = (
        within_df[
            "delta_score"
        ]
        >
        0
    ).astype(int)


    within_df[
        "age_pred1"
    ] = age_pred1

    within_df[
        "age_pred2"
    ] = age_pred2

    within_df[
        "delta_age_pred"
    ] = (
        within_df[
            "age_pred2"
        ]
        -
        within_df[
            "age_pred1"
        ]
    )

    within_df[
        "age_probe_order_correct"
    ] = (
        within_df[
            "delta_age_pred"
        ]
        >
        0
    ).astype(int)


    within_df[
        "spacing_distance"
    ] = spacing_distance


    unified_within_accuracy = float(
        within_df[
            "axis_order_correct"
        ]
        .mean()
    )


    developmental_delta_r = safe_pearson(
        within_df[
            "delta_age_true"
        ],
        within_df[
            "delta_score"
        ],
    )


    developmental_delta_rho = safe_spearman(
        within_df[
            "delta_age_true"
        ],
        within_df[
            "delta_score"
        ],
    )


    age_probe_order_accuracy = float(
        within_df[
            "age_probe_order_correct"
        ]
        .mean()
    )


    delta_age_mae = float(
        mean_absolute_error(
            within_df[
                "delta_age_true"
            ],
            within_df[
                "delta_age_pred"
            ],
        )
    )


    delta_age_r = safe_pearson(
        within_df[
            "delta_age_true"
        ],
        within_df[
            "delta_age_pred"
        ],
    )


    spacing_delta_r = safe_pearson(
        within_df[
            "delta_age_true"
        ],
        within_df[
            "spacing_distance"
        ],
    )


    spacing_delta_rho = safe_spearman(
        within_df[
            "delta_age_true"
        ],
        within_df[
            "spacing_distance"
        ],
    )


    print("\n" + "-" * 80)
    print("HELD-OUT WITHIN-SUBJECT DEVELOPMENT")
    print("-" * 80)

    print(
        f"Pairs:                    "
        f"{len(within_df)}"
    )

    print(
        f"Unified-axis Order Acc:   "
        f"{unified_within_accuracy:.4f}"
    )

    print(
        f"Delta-score / Delta-t r:  "
        f"{developmental_delta_r:.4f}"
    )

    print(
        f"Delta-score / Delta-t rho:"
        f" {developmental_delta_rho:.4f}"
    )

    print(
        f"Age-probe Order Acc:      "
        f"{age_probe_order_accuracy:.4f}"
    )

    print(
        f"Age-probe Delta MAE:      "
        f"{delta_age_mae:.4f}"
    )

    print(
        f"Age-probe Delta r:        "
        f"{delta_age_r:.4f}"
    )


    # ========================================================
    # RELATIVE SPACING
    # ========================================================

    triplet_df = build_triplets(
        pred_df
    )

    global_correct = []
    local_correct = []

    d12_values = []
    d23_values = []
    d13_values = []

    for _, row in triplet_df.iterrows():

        i = int(
            row[
                "idx1"
            ]
        )

        j = int(
            row[
                "idx2"
            ]
        )

        k = int(
            row[
                "idx3"
            ]
        )

        q1 = test_spacing[i]
        q2 = test_spacing[j]
        q3 = test_spacing[k]

        d12 = float(
            np.linalg.norm(
                q2 - q1
            )
        )

        d23 = float(
            np.linalg.norm(
                q3 - q2
            )
        )

        d13 = float(
            np.linalg.norm(
                q3 - q1
            )
        )

        d12_values.append(d12)
        d23_values.append(d23)
        d13_values.append(d13)

        global_correct.append(
            int(
                (
                    d13 > d12
                )
                and
                (
                    d13 > d23
                )
            )
        )

        delta12 = float(
            row[
                "delta12"
            ]
        )

        delta23 = float(
            row[
                "delta23"
            ]
        )

        if delta12 > delta23:

            local_correct.append(
                int(
                    d12 > d23
                )
            )

        elif delta23 > delta12:

            local_correct.append(
                int(
                    d23 > d12
                )
            )


    triplet_df[
        "d12"
    ] = d12_values

    triplet_df[
        "d23"
    ] = d23_values

    triplet_df[
        "d13"
    ] = d13_values


    global_spacing_accuracy = float(
        np.mean(
            global_correct
        )
    )


    local_spacing_accuracy = (
        float(
            np.mean(
                local_correct
            )
        )
        if local_correct
        else np.nan
    )


    print("\n" + "-" * 80)
    print("HELD-OUT RELATIVE TEMPORAL SPACING")
    print("-" * 80)

    print(
        f"Triplets:              "
        f"{len(triplet_df)}"
    )

    print(
        f"Global Spacing Acc:    "
        f"{global_spacing_accuracy:.4f}"
    )

    print(
        f"Local Spacing Acc:     "
        f"{local_spacing_accuracy:.4f}"
    )

    print(
        f"Spacing distance / Δt r:   "
        f"{spacing_delta_r:.4f}"
    )

    print(
        f"Spacing distance / Δt rho: "
        f"{spacing_delta_rho:.4f}"
    )


    # ========================================================
    # CROSS-SUBJECT ORDER
    # ========================================================

    cross_df = build_cross_pairs(
        pred_df
    )

    younger_score = []
    older_score = []

    for _, row in cross_df.iterrows():

        younger_idx = int(
            row[
                "younger_idx"
            ]
        )

        older_idx = int(
            row[
                "older_idx"
            ]
        )

        younger_score.append(
            float(
                test_score[
                    younger_idx
                ]
            )
        )

        older_score.append(
            float(
                test_score[
                    older_idx
                ]
            )
        )


    cross_df[
        "younger_score"
    ] = younger_score

    cross_df[
        "older_score"
    ] = older_score

    cross_df[
        "score_difference"
    ] = (
        cross_df[
            "older_score"
        ]
        -
        cross_df[
            "younger_score"
        ]
    )

    cross_df[
        "correct"
    ] = (
        cross_df[
            "score_difference"
        ]
        >
        0
    ).astype(int)


    cross_accuracy = float(
        cross_df[
            "correct"
        ]
        .mean()
    )


    def cross_bin_accuracy(
        bin_id
    ):

        subset = cross_df[
            cross_df[
                "gap_bin"
            ]
            ==
            bin_id
        ]

        if len(subset) == 0:
            return np.nan

        return float(
            subset[
                "correct"
            ]
            .mean()
        )


    cross_small = (
        cross_bin_accuracy(0)
    )

    cross_medium = (
        cross_bin_accuracy(1)
    )

    cross_large = (
        cross_bin_accuracy(2)
    )


    print("\n" + "-" * 80)
    print("HELD-OUT CROSS-SUBJECT DEVELOPMENTAL ORDER")
    print("-" * 80)

    print(
        f"Cross pairs:         "
        f"{len(cross_df)}"
    )

    print(
        f"Overall Accuracy:    "
        f"{cross_accuracy:.4f}"
    )

    print(
        f"Small-gap Accuracy:  "
        f"{cross_small:.4f}"
    )

    print(
        f"Medium Accuracy:     "
        f"{cross_medium:.4f}"
    )

    print(
        f"Large-gap Accuracy:  "
        f"{cross_large:.4f}"
    )


    # ========================================================
    # SUBJECT MONOTONICITY
    # ========================================================

    geometry_df = (
        compute_subject_geometry(
            pred_df
        )
    )


    subject_monotonicity = float(
        geometry_df[
            "developmental_monotonicity"
        ]
        .mean()
    )


    subject_consecutive = float(
        geometry_df[
            "developmental_consecutive_accuracy"
        ]
        .mean()
    )


    age_probe_monotonicity = float(
        geometry_df[
            "age_probe_monotonicity"
        ]
        .mean()
    )


    print("\n" + "-" * 80)
    print("SUBJECT-LEVEL DEVELOPMENTAL GEOMETRY")
    print("-" * 80)

    print(
        f"Developmental monotonicity: "
        f"{subject_monotonicity:.4f}"
    )

    print(
        f"Consecutive Order Accuracy: "
        f"{subject_consecutive:.4f}"
    )

    print(
        f"Age-probe monotonicity:      "
        f"{age_probe_monotonicity:.4f}"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "model":
            (
                "TempoDINO-v6"
                if checkpoint_name
                ==
                "v6_epoch2"
                else
                "TempoDINO-v7"
            ),

        "checkpoint":
            checkpoint_name,

        "checkpoint_epoch":
            int(
                checkpoint.get(
                    "epoch",
                    -1
                )
            ),

        "fold":
            int(
                args.fold
            ),

        "axis_source":
            head_source,

        "n_test_mri":
            int(
                len(test_df)
            ),

        "n_test_subjects":
            int(
                len(
                    test_subjects
                )
            ),

        # ====================================================
        # PRIMARY REPRESENTATION METRICS
        # ====================================================

        "developmental_score_age_pearson":
            score_age_r,

        "developmental_score_age_spearman":
            score_age_rho,

        "unified_within_order_accuracy":
            unified_within_accuracy,

        "developmental_delta_time_pearson":
            developmental_delta_r,

        "developmental_delta_time_spearman":
            developmental_delta_rho,

        "cross_subject_order_accuracy":
            cross_accuracy,

        "cross_small_accuracy":
            cross_small,

        "cross_medium_accuracy":
            cross_medium,

        "cross_large_accuracy":
            cross_large,

        "subject_developmental_monotonicity":
            subject_monotonicity,

        "subject_consecutive_order_accuracy":
            subject_consecutive,

        "global_spacing_accuracy":
            global_spacing_accuracy,

        "local_spacing_accuracy":
            local_spacing_accuracy,

        "spacing_distance_delta_time_pearson":
            spacing_delta_r,

        "spacing_distance_delta_time_spearman":
            spacing_delta_rho,

        # ====================================================
        # SECONDARY AGE PROBE
        # ====================================================

        "age_mae":
            age_mae,

        "age_rmse":
            age_rmse,

        "age_r2":
            age_r2,

        "age_correlation":
            age_r,

        "age_spearman":
            age_rho,

        "age_probe_order_accuracy":
            age_probe_order_accuracy,

        "delta_age_mae":
            delta_age_mae,

        "delta_age_correlation":
            delta_age_r,

        "age_probe_subject_monotonicity":
            age_probe_monotonicity,

        # ====================================================
        # REPRESENTATION DIAGNOSTICS
        # ====================================================

        "train_effective_rank":
            train_rank,

        "test_effective_rank":
            test_rank,

        "train_mean_dimension_std":
            train_std,

        "test_mean_dimension_std":
            test_std,

        "train_mean_cosine":
            train_cos,

        "test_mean_cosine":
            test_cos,
    }


    # ========================================================
    # SAVE
    # ========================================================

    out_dir = (
        RESULT_ROOT
        /
        f"fold_{args.fold}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        out_dir
        /
        checkpoint_name
    )


    pred_df.to_csv(
        Path(
            str(prefix)
            +
            "_predictions.csv"
        ),
        index=False,
    )


    within_df.to_csv(
        Path(
            str(prefix)
            +
            "_within_pairs.csv"
        ),
        index=False,
    )


    triplet_df.to_csv(
        Path(
            str(prefix)
            +
            "_triplets.csv"
        ),
        index=False,
    )


    cross_df.to_csv(
        Path(
            str(prefix)
            +
            "_cross_pairs.csv"
        ),
        index=False,
    )


    geometry_df.to_csv(
        Path(
            str(prefix)
            +
            "_subject_geometry.csv"
        ),
        index=False,
    )


    pd.DataFrame(
        [summary]
    ).to_csv(
        Path(
            str(prefix)
            +
            "_summary.csv"
        ),
        index=False,
    )


    with open(
        Path(
            str(prefix)
            +
            "_summary.json"
        ),
        "w",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )


    del encoder
    del developmental_head
    del spacing_projector

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ============================================================
# RUN ALL
# ============================================================

summaries = []


for checkpoint_name in checkpoint_names:

    summaries.append(

        evaluate_checkpoint(
            checkpoint_name,
            checkpoint_map[
                checkpoint_name
            ],
        )
    )


# ============================================================
# COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    summaries
)


print("\n" + "=" * 80)
print("V6 / V7 DEVELOPMENTAL REPRESENTATION COMPARISON")
print("=" * 80)


primary_columns = [

    "checkpoint",
    "checkpoint_epoch",

    "unified_within_order_accuracy",

    "cross_subject_order_accuracy",
    "cross_small_accuracy",

    "subject_developmental_monotonicity",
    "subject_consecutive_order_accuracy",

    "global_spacing_accuracy",
    "local_spacing_accuracy",

    "developmental_score_age_spearman",

    "developmental_delta_time_pearson",

    "spacing_distance_delta_time_pearson",
]


print(
    comparison_df[
        primary_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("SECONDARY AGE PROBE COMPARISON")
print("=" * 80)


secondary_columns = [

    "checkpoint",

    "age_mae",
    "age_r2",
    "age_correlation",

    "age_probe_order_accuracy",

    "delta_age_mae",
    "delta_age_correlation",

    "test_effective_rank",
]


print(
    comparison_df[
        secondary_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# COMPOSITE DEVELOPMENTAL SCORE
# ============================================================

"""
This composite is NOT a training objective.

It is used only as a descriptive checkpoint-selection diagnostic.

Metrics are chosen to represent:
- within-subject direction
- cross-subject order
- subject monotonicity
- global spacing
- local spacing

Age MAE is deliberately excluded.
"""

comparison_df[
    "developmental_structure_score"
] = (

    0.25
    *
    comparison_df[
        "unified_within_order_accuracy"
    ]

    +

    0.25
    *
    comparison_df[
        "cross_subject_order_accuracy"
    ]

    +

    0.20
    *
    comparison_df[
        "subject_developmental_monotonicity"
    ]

    +

    0.15
    *
    comparison_df[
        "global_spacing_accuracy"
    ]

    +

    0.15
    *
    comparison_df[
        "local_spacing_accuracy"
    ]
)


best_structure_row = (
    comparison_df
    .sort_values(
        "developmental_structure_score",
        ascending=False,
    )
    .iloc[0]
)


print("\n" + "=" * 80)
print("DEVELOPMENTAL REPRESENTATION DECISION")
print("=" * 80)


print(
    f"Best structural checkpoint: "
    f"{best_structure_row['checkpoint']}"
)

print(
    f"Structure score: "
    f"{best_structure_row['developmental_structure_score']:.4f}"
)

print(
    f"Within order: "
    f"{best_structure_row['unified_within_order_accuracy']:.4f}"
)

print(
    f"Cross order: "
    f"{best_structure_row['cross_subject_order_accuracy']:.4f}"
)

print(
    f"Cross small: "
    f"{best_structure_row['cross_small_accuracy']:.4f}"
)

print(
    f"Subject monotonicity: "
    f"{best_structure_row['subject_developmental_monotonicity']:.4f}"
)

print(
    f"Consecutive order: "
    f"{best_structure_row['subject_consecutive_order_accuracy']:.4f}"
)

print(
    f"Global spacing: "
    f"{best_structure_row['global_spacing_accuracy']:.4f}"
)

print(
    f"Local spacing: "
    f"{best_structure_row['local_spacing_accuracy']:.4f}"
)

print(
    f"Axis-age Spearman: "
    f"{best_structure_row['developmental_score_age_spearman']:.4f}"
)

print(
    f"Delta-axis / Delta-t r: "
    f"{best_structure_row['developmental_delta_time_pearson']:.4f}"
)


print("\nSecondary age probe:")

print(
    f"MAE: "
    f"{best_structure_row['age_mae']:.4f}"
)

print(
    f"R2: "
    f"{best_structure_row['age_r2']:.4f}"
)


# ============================================================
# SAVE COMPARISON
# ============================================================

out_dir = (
    RESULT_ROOT
    /
    f"fold_{args.fold}"
)

comparison_path = (
    out_dir
    /
    "checkpoint_comparison.csv"
)

comparison_json_path = (
    out_dir
    /
    "checkpoint_comparison.json"
)


comparison_df.to_csv(
    comparison_path,
    index=False,
)


with open(
    comparison_json_path,
    "w",
) as handle:

    json.dump(
        comparison_df.to_dict(
            orient="records"
        ),
        handle,
        indent=2,
    )


print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)

print(
    comparison_path
)

print(
    comparison_json_path
)

print("\nDONE.")
