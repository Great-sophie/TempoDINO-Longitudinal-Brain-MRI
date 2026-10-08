#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TempoDINO-v7: Unified Developmental Axis

Core objective
--------------
Learn a longitudinal developmental representation rather than
directly predict chronological age.

Initialization:
    TempoDINO-v6 cross-ordinal epoch_2.pt

Training objectives:
    1. DINO self-distillation
    2. Within-subject temporal order
    3. Cross-subject ordinal developmental alignment
    4. Relative temporal spacing
    5. Variance / covariance regularization

IMPORTANT
---------
The SAME scalar developmental head s(z) is used for:
    - within-subject order
    - cross-subject order

No:
    - absolute age regression
    - exact delta-month regression
    - age-bin classification
    - sex classification
"""

import argparse
import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


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

META_PATH = (
    PROJECT_DIR
    / "metadata"
    / "folds_balanced_preprocessed96.csv"
)

V6_CHECKPOINT_ROOT = (
    PROJECT_DIR
    / "checkpoints"
    / "tempodino_v6_crossordinal"
)

CHECKPOINT_ROOT = (
    RUN_ROOT
    / "checkpoints"
    / "tempodino_v7_unified_axis"
)

RESULT_ROOT = (
    RUN_ROOT
    / "results"
    / "tempodino_v7_unified_axis"
)

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fold",
    type=int,
    default=0,
)

parser.add_argument(
    "--init_checkpoint",
    type=str,
    default="epoch_2.pt",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=3,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
)

parser.add_argument(
    "--temporal_batch_size",
    type=int,
    default=4,
)

parser.add_argument(
    "--triplet_batch_size",
    type=int,
    default=4,
)

parser.add_argument(
    "--cross_batch_size",
    type=int,
    default=8,
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
)

parser.add_argument(
    "--lr",
    type=float,
    default=5e-5,
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
)

parser.add_argument(
    "--lambda_within",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--lambda_cross",
    type=float,
    default=0.20,
)

parser.add_argument(
    "--lambda_spacing",
    type=float,
    default=0.10,
)

parser.add_argument(
    "--lambda_var",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--lambda_cov",
    type=float,
    default=0.05,
)

parser.add_argument(
    "--within_margin",
    type=float,
    default=0.20,
)

parser.add_argument(
    "--spacing_margin",
    type=float,
    default=0.10,
)

parser.add_argument(
    "--var_target",
    type=float,
    default=0.05,
)

parser.add_argument(
    "--student_temp",
    type=float,
    default=0.10,
)

parser.add_argument(
    "--teacher_temp",
    type=float,
    default=0.07,
)

parser.add_argument(
    "--teacher_momentum",
    type=float,
    default=0.998,
)

parser.add_argument(
    "--center_momentum",
    type=float,
    default=0.99,
)

parser.add_argument(
    "--seed",
    type=int,
    default=20260820,
)

parser.add_argument(
    "--save_every",
    type=int,
    default=1,
)

args = parser.parse_args()


if args.fold not in [0, 1, 2, 3, 4]:
    raise ValueError("--fold must be 0-4")


# ============================================================
# OUTPUT PATHS
# ============================================================

CHECKPOINT_DIR = (
    CHECKPOINT_ROOT
    / f"fold_{args.fold}"
)

RESULT_DIR = (
    RESULT_ROOT
    / f"fold_{args.fold}"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_PATH = CHECKPOINT_DIR / "final.pt"
BEST_PATH = CHECKPOINT_DIR / "best.pt"

HISTORY_PATH = RESULT_DIR / "history.csv"
SUMMARY_PATH = RESULT_DIR / "summary.json"

WITHIN_PAIR_PATH = (
    RESULT_DIR
    / "within_subject_pairs.csv"
)

TRIPLET_PATH = (
    RESULT_DIR
    / "within_subject_triplets.csv"
)

CROSS_PAIR_PATH = (
    RESULT_DIR
    / "cross_subject_pairs.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(
    args.seed
    +
    args.fold
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)
print("TEMPODINO V7")
print("UNIFIED DEVELOPMENTAL AXIS")
print("=" * 80)

print(f"\nFold:   {args.fold}")
print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
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


class DINOHead(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
    ):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(
                in_dim,
                512,
            ),

            nn.GELU(),

            nn.Linear(
                512,
                256,
            ),

            nn.GELU(),
        )

        self.last_layer = nn.Linear(
            256,
            out_dim,
            bias=False,
        )

    def forward(self, z):

        x = self.mlp(z)

        x = F.normalize(
            x,
            dim=-1,
        )

        return self.last_layer(x)


class TempoDINO(nn.Module):

    def __init__(
        self,
        embedding_dim,
        out_dim,
    ):

        super().__init__()

        self.encoder = ResNet18Encoder3D(
            embedding_dim
        )

        self.dino_head = DINOHead(
            embedding_dim,
            out_dim,
        )

    def forward(self, x):

        z = self.encoder(x)
        logits = self.dino_head(z)

        return logits, z


# ============================================================
# UNIFIED DEVELOPMENTAL HEAD
# ============================================================

class UnifiedDevelopmentalHead(nn.Module):

    """
    One scalar developmental coordinate s(z).

    The SAME score is used for:
        within-subject temporal order
        cross-subject developmental order
    """

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


# ============================================================
# SPACING PROJECTOR
# ============================================================

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
# DINO LOSS
# ============================================================

class DINOLoss(nn.Module):

    def __init__(
        self,
        out_dim,
        student_temp,
        center_momentum,
    ):

        super().__init__()

        self.student_temp = (
            student_temp
        )

        self.center_momentum = (
            center_momentum
        )

        self.register_buffer(
            "center",
            torch.zeros(
                1,
                out_dim,
            ),
        )

    def forward(
        self,
        student_output,
        teacher_output,
        teacher_temp,
    ):

        student_log_prob = F.log_softmax(
            student_output
            /
            self.student_temp,
            dim=-1,
        )

        teacher_prob = F.softmax(
            (
                teacher_output
                -
                self.center
            )
            /
            teacher_temp,
            dim=-1,
        ).detach()

        return -(
            teacher_prob
            *
            student_log_prob
        ).sum(
            dim=-1
        ).mean()

    @torch.no_grad()
    def update_center(
        self,
        teacher_output,
    ):

        batch_center = (
            teacher_output
            .mean(
                dim=0,
                keepdim=True,
            )
        )

        self.center.mul_(
            self.center_momentum
        ).add_(
            batch_center
            *
            (
                1
                -
                self.center_momentum
            )
        )


# ============================================================
# REGULARIZATION
# ============================================================

def variance_loss(
    z,
    target_std,
    eps=1e-4,
):

    std = torch.sqrt(
        z.var(
            dim=0,
            unbiased=False,
        )
        +
        eps
    )

    return F.relu(
        target_std
        -
        std
    ).mean()


def covariance_loss(z):

    z = z.float()

    n, d = z.shape

    if n <= 1:

        return torch.zeros(
            (),
            device=z.device,
        )

    z = (
        z
        -
        z.mean(
            dim=0,
            keepdim=True,
        )
    )

    cov = (
        z.T @ z
    ) / (
        n - 1
    )

    diag = torch.diagonal(
        cov
    )

    offdiag = (
        cov.pow(2).sum()
        -
        diag.pow(2).sum()
    )

    return offdiag / d


# ============================================================
# UNIFIED WITHIN LOSS
# ============================================================

def within_subject_order_loss(
    early_score,
    late_score,
    margin,
):

    delta = (
        late_score
        -
        early_score
    )

    loss = F.relu(
        margin
        -
        delta
    ).mean()

    accuracy = (
        delta > 0
    ).float().mean()

    return (
        loss,
        accuracy,
        delta.detach(),
    )


# ============================================================
# UNIFIED CROSS-SUBJECT LOSS
# ============================================================

def cross_subject_order_loss(
    younger_score,
    older_score,
):

    delta = (
        older_score
        -
        younger_score
    )

    loss = F.softplus(
        -delta
    ).mean()

    accuracy = (
        delta > 0
    ).float().mean()

    return (
        loss,
        accuracy,
        delta.detach(),
    )


# ============================================================
# RELATIVE SPACING
# ============================================================

def relative_temporal_spacing_loss(
    z1,
    z2,
    z3,
    delta12,
    delta23,
    spacing_projector,
    margin,
    eps=1e-6,
):

    q1 = spacing_projector(z1)
    q2 = spacing_projector(z2)
    q3 = spacing_projector(z3)

    d12 = torch.norm(
        q2 - q1,
        p=2,
        dim=1,
    )

    d23 = torch.norm(
        q3 - q2,
        p=2,
        dim=1,
    )

    d13 = torch.norm(
        q3 - q1,
        p=2,
        dim=1,
    )

    # --------------------------------------------------------
    # Global:
    # d13 > d12 and d13 > d23
    # --------------------------------------------------------

    global_loss = (
        F.relu(
            margin
            +
            d12
            -
            d13
        ).mean()

        +

        F.relu(
            margin
            +
            d23
            -
            d13
        ).mean()
    )

    global_accuracy = (
        (
            d13 > d12
        )
        &
        (
            d13 > d23
        )
    ).float().mean()

    # --------------------------------------------------------
    # Local relative spacing
    # --------------------------------------------------------

    interval_difference = (
        delta12.float()
        -
        delta23.float()
    )

    valid = (
        torch.abs(
            interval_difference
        )
        >
        eps
    )

    if valid.any():

        target = torch.sign(
            interval_difference[
                valid
            ]
        )

        pred_difference = (
            d12[
                valid
            ]
            -
            d23[
                valid
            ]
        )

        local_loss = F.relu(
            margin
            -
            target
            *
            pred_difference
        ).mean()

        local_accuracy = (
            target
            *
            pred_difference
            >
            0
        ).float().mean()

    else:

        local_loss = (
            d12.sum()
            *
            0.0
        )

        local_accuracy = torch.tensor(
            float("nan"),
            device=d12.device,
        )

    spacing_loss = (
        global_loss
        +
        local_loss
    )

    return (
        spacing_loss,
        global_accuracy,
        local_accuracy,
        d12.detach(),
        d23.detach(),
        d13.detach(),
    )


# ============================================================
# EMA
# ============================================================

@torch.no_grad()
def update_teacher(
    student,
    teacher,
    momentum,
):

    for student_param, teacher_param in zip(
        student.parameters(),
        teacher.parameters(),
    ):

        teacher_param.data.mul_(
            momentum
        ).add_(
            student_param.data
            *
            (
                1
                -
                momentum
            )
        )


# ============================================================
# DATASETS
# ============================================================

class DINODataset(Dataset):

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

        return len(
            self.df
        )

    def augment(self, image):

        x = image.clone()

        if random.random() < 0.5:

            x = (
                x
                *
                random.uniform(
                    0.95,
                    1.05,
                )
            )

        if random.random() < 0.5:

            noise = (
                torch.randn_like(
                    x
                )
                *
                random.uniform(
                    0.005,
                    0.02,
                )
            )

            mask = (
                x != 0
            )

            x = torch.where(
                mask,
                x + noise,
                x,
            )

        return x

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

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

            "view1":
                self.augment(
                    image
                ),

            "view2":
                self.augment(
                    image
                ),
        }


class WithinPairDataset(Dataset):

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

        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        early = np.load(
            row[
                "path_early"
            ]
        ).astype(
            np.float32
        )

        late = np.load(
            row[
                "path_late"
            ]
        ).astype(
            np.float32
        )

        return {

            "early":
                torch.from_numpy(
                    early
                ).unsqueeze(0),

            "late":
                torch.from_numpy(
                    late
                ).unsqueeze(0),
        }


class TripletDataset(Dataset):

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

        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        x1 = np.load(
            row["path1"]
        ).astype(np.float32)

        x2 = np.load(
            row["path2"]
        ).astype(np.float32)

        x3 = np.load(
            row["path3"]
        ).astype(np.float32)

        return {

            "image1":
                torch.from_numpy(
                    x1
                ).unsqueeze(0),

            "image2":
                torch.from_numpy(
                    x2
                ).unsqueeze(0),

            "image3":
                torch.from_numpy(
                    x3
                ).unsqueeze(0),

            "delta12":
                torch.tensor(
                    float(
                        row[
                            "delta12"
                        ]
                    ),
                    dtype=torch.float32,
                ),

            "delta23":
                torch.tensor(
                    float(
                        row[
                            "delta23"
                        ]
                    ),
                    dtype=torch.float32,
                ),
        }


class CrossSubjectPairDataset(Dataset):

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

        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        younger = np.load(
            row[
                "path_younger"
            ]
        ).astype(
            np.float32
        )

        older = np.load(
            row[
                "path_older"
            ]
        ).astype(
            np.float32
        )

        return {

            "younger":
                torch.from_numpy(
                    younger
                ).unsqueeze(0),

            "older":
                torch.from_numpy(
                    older
                ).unsqueeze(0),

            "gap_bin":
                torch.tensor(
                    int(
                        row[
                            "gap_bin"
                        ]
                    ),
                    dtype=torch.long,
                ),
        }


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

# Rebase only the processed-array paths in memory; do not edit the original CSV.
# Expected layout: <processed_root>/sub-001/sub-001_age-012p0m_T1_96.npy
PROCESSED_ROOT = Path(os.environ.get(
    "TEMPODINO_PROCESSED_ROOT", PROJECT_DIR.parent / "monkey_development_preprocessed_96"
)).expanduser().resolve()

def resolve_preprocessed_path(stored_path):
    p = Path(str(stored_path))
    if p.is_file():
        return str(p)
    candidate = PROCESSED_ROOT / p.parent.name / p.name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"Missing preprocessed MRI: {p.name}. Expected under {PROCESSED_ROOT}. "
        "Check TEMPODINO_PROCESSED_ROOT; original metadata is unchanged."
    )

df["preprocessed_path"] = df["preprocessed_path"].map(resolve_preprocessed_path)



df[
    "subject_id"
] = (
    df[
        "subject_id"
    ]
    .astype(str)
    .str.zfill(3)
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


heldout_df = (
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


heldout_subjects = set(
    heldout_df[
        "subject_id"
    ]
)


overlap = (
    train_subjects
    &
    heldout_subjects
)


if overlap:

    raise RuntimeError(
        f"Subject leakage: {overlap}"
    )


print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)

print(
    f"Train MRI:          {len(train_df)}"
)

print(
    f"Train subjects:     {len(train_subjects)}"
)

print(
    f"Held-out MRI:       {len(heldout_df)}"
)

print(
    f"Held-out subjects:  {len(heldout_subjects)}"
)

print(
    "Subject leakage audit: PASS"
)


# ============================================================
# BUILD WITHIN-SUBJECT REGISTRY
# ============================================================

within_rows = []
triplet_rows = []


for (
    subject_id,
    group
) in train_df.groupby(
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

    # --------------------------------------------------------
    # Pairs
    # --------------------------------------------------------

    for i in range(n):

        for j in range(
            i + 1,
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

            if age2 <= age1:
                continue

            within_rows.append(
                {

                    "subject_id":
                        subject_id,

                    "path_early":
                        group.loc[
                            i,
                            "preprocessed_path"
                        ],

                    "path_late":
                        group.loc[
                            j,
                            "preprocessed_path"
                        ],

                    "age_early":
                        age1,

                    "age_late":
                        age2,

                    "delta_months":
                        age2 - age1,
                }
            )

    # --------------------------------------------------------
    # Triplets
    # --------------------------------------------------------

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

                triplet_rows.append(
                    {

                        "subject_id":
                            subject_id,

                        "path1":
                            group.loc[
                                i,
                                "preprocessed_path"
                            ],

                        "path2":
                            group.loc[
                                j,
                                "preprocessed_path"
                            ],

                        "path3":
                            group.loc[
                                k,
                                "preprocessed_path"
                            ],

                        "delta12":
                            age2 - age1,

                        "delta23":
                            age3 - age2,
                    }
                )


within_df = pd.DataFrame(
    within_rows
)

triplet_df = pd.DataFrame(
    triplet_rows
)


within_df.to_csv(
    WITHIN_PAIR_PATH,
    index=False,
)

triplet_df.to_csv(
    TRIPLET_PATH,
    index=False,
)


# ============================================================
# BUILD CROSS-SUBJECT ORDINAL REGISTRY
# ============================================================

cross_rows = []


for i in range(
    len(
        train_df
    )
):

    row_i = train_df.iloc[
        i
    ]

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
        len(
            train_df
        ),
    ):

        row_j = train_df.iloc[
            j
        ]

        subject_j = str(
            row_j[
                "subject_id"
            ]
        )

        if (
            subject_i
            ==
            subject_j
        ):

            continue

        age_j = float(
            row_j[
                "age_month"
            ]
        )

        gap = abs(
            age_j
            -
            age_i
        )

        # Avoid effectively same-age comparisons.
        if gap < 2.0:
            continue

        if gap < 4.0:
            gap_bin = 0

        elif gap < 8.0:
            gap_bin = 1

        else:
            gap_bin = 2


        if age_i < age_j:

            younger = row_i
            older = row_j

        else:

            younger = row_j
            older = row_i


        cross_rows.append(
            {

                "subject_younger":
                    str(
                        younger[
                            "subject_id"
                        ]
                    ),

                "subject_older":
                    str(
                        older[
                            "subject_id"
                        ]
                    ),

                "path_younger":
                    younger[
                        "preprocessed_path"
                    ],

                "path_older":
                    older[
                        "preprocessed_path"
                    ],

                "age_younger":
                    float(
                        younger[
                            "age_month"
                        ]
                    ),

                "age_older":
                    float(
                        older[
                            "age_month"
                        ]
                    ),

                "age_gap":
                    gap,

                "gap_bin":
                    gap_bin,
            }
        )


cross_df = pd.DataFrame(
    cross_rows
)


# ============================================================
# BALANCE CROSS GAP BINS
# ============================================================

counts = (
    cross_df[
        "gap_bin"
    ]
    .value_counts()
)


if len(counts) < 3:

    raise RuntimeError(
        "Missing cross-subject gap bin."
    )


target_per_bin = int(
    counts.min()
)


balanced_parts = []


for gap_bin in [0, 1, 2]:

    part = (
        cross_df[
            cross_df[
                "gap_bin"
            ]
            ==
            gap_bin
        ]
        .sample(
            n=target_per_bin,
            random_state=(
                args.seed
                +
                gap_bin
            ),
        )
    )

    balanced_parts.append(
        part
    )


cross_df = (
    pd.concat(
        balanced_parts,
        ignore_index=True,
    )
    .sample(
        frac=1.0,
        random_state=args.seed,
    )
    .reset_index(
        drop=True
    )
)


cross_df.to_csv(
    CROSS_PAIR_PATH,
    index=False,
)


print("\n" + "=" * 80)
print("TRAINING REGISTRY")
print("=" * 80)

print(
    f"Within-subject pairs:    {len(within_df)}"
)

print(
    f"Within-subject triplets: {len(triplet_df)}"
)

print(
    f"Cross-subject pairs:     {len(cross_df)}"
)

print(
    "\nCross gap bins:"
)

print(
    cross_df[
        "gap_bin"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# LOADERS
# ============================================================

dino_loader = DataLoader(
    DINODataset(
        train_df
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(
        args.num_workers > 0
    ),
    drop_last=True,
)


within_loader = DataLoader(
    WithinPairDataset(
        within_df
    ),
    batch_size=args.temporal_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(
        args.num_workers > 0
    ),
    drop_last=True,
)


triplet_loader = DataLoader(
    TripletDataset(
        triplet_df
    ),
    batch_size=args.triplet_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(
        args.num_workers > 0
    ),
    drop_last=True,
)


cross_loader = DataLoader(
    CrossSubjectPairDataset(
        cross_df
    ),
    batch_size=args.cross_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(
        args.num_workers > 0
    ),
    drop_last=True,
)


# ============================================================
# LOAD V6 INITIALIZATION
# ============================================================

init_path = (
    V6_CHECKPOINT_ROOT
    /
    f"fold_{args.fold}"
    /
    args.init_checkpoint
)


if not init_path.exists():

    raise FileNotFoundError(
        init_path
    )


print("\n" + "=" * 80)
print("INITIALIZATION")
print("=" * 80)

print(
    init_path
)


checkpoint = torch.load(
    init_path,
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


out_dim = int(
    checkpoint.get(
        "config",
        {}
    ).get(
        "out_dim",
        2048,
    )
)


# ============================================================
# STUDENT / TEACHER
# ============================================================

student = TempoDINO(
    embedding_dim,
    out_dim,
).to(
    device
)


teacher = TempoDINO(
    embedding_dim,
    out_dim,
).to(
    device
)


student.load_state_dict(
    checkpoint[
        "student_state_dict"
    ],
    strict=True,
)


teacher.load_state_dict(
    checkpoint[
        "teacher_state_dict"
    ],
    strict=True,
)


for parameter in teacher.parameters():

    parameter.requires_grad = False


teacher.eval()


# ============================================================
# SPACING PROJECTOR
# ============================================================

spacing_projector = (
    RelativeSpacingProjector(
        embedding_dim,
        spacing_dim,
    )
    .to(
        device
    )
)


spacing_projector.load_state_dict(
    checkpoint[
        "spacing_projector_state_dict"
    ],
    strict=True,
)


# ============================================================
# NEW UNIFIED DEVELOPMENTAL HEAD
# ============================================================

developmental_head = (
    UnifiedDevelopmentalHead(
        embedding_dim
    )
    .to(
        device
    )
)


# Optional intelligent initialization:
# initialize from the old V6 temporal head, because that head
# already encodes within-subject developmental direction.

if (
    "temporal_head_state_dict"
    in checkpoint
):

    old_state = checkpoint[
        "temporal_head_state_dict"
    ]

    try:

        developmental_head.load_state_dict(
            old_state,
            strict=True,
        )

        print(
            "Unified head initialized from "
            "V6 temporal head."
        )

    except RuntimeError:

        print(
            "V6 temporal head shape mismatch; "
            "using random unified-head initialization."
        )


# ============================================================
# DINO LOSS
# ============================================================

dino_criterion = DINOLoss(
    out_dim,
    args.student_temp,
    args.center_momentum,
).to(
    device
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    list(
        student.parameters()
    )

    +

    list(
        developmental_head.parameters()
    )

    +

    list(
        spacing_projector.parameters()
    ),

    lr=args.lr,
    weight_decay=args.weight_decay,
)


scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),
)


# ============================================================
# CONFIG
# ============================================================

print("\n" + "=" * 80)
print("TEMPODINO V7 CONFIGURATION")
print("=" * 80)

print(
    f"Initialization:       {args.init_checkpoint}"
)

print(
    f"Embedding dim:       {embedding_dim}"
)

print(
    f"DINO output:         {out_dim}"
)

print(
    f"Spacing dim:         {spacing_dim}"
)

print(
    f"Epochs:              {args.epochs}"
)

print(
    f"Learning rate:       {args.lr}"
)

print(
    f"Lambda within:       {args.lambda_within}"
)

print(
    f"Lambda cross:        {args.lambda_cross}"
)

print(
    f"Lambda spacing:      {args.lambda_spacing}"
)

print(
    f"Lambda var:          {args.lambda_var}"
)

print(
    f"Lambda cov:          {args.lambda_cov}"
)

print(
    "\nUnified developmental head:"
)

print(
    "Embedding 256 -> 64 -> 1"
)

print(
    "Same scalar used for within + cross ordering."
)

print(
    "\nNo age regression."
)

print(
    "No sex classification."
)


# ============================================================
# TRAINING
# ============================================================

history = []

best_total_loss = float(
    "inf"
)

best_epoch = None

global_step = 0

total_steps = (
    args.epochs
    *
    len(
        dino_loader
    )
)


for epoch in range(
    1,
    args.epochs + 1,
):

    student.train()

    developmental_head.train()

    spacing_projector.train()

    teacher.eval()


    within_iter = iter(
        within_loader
    )

    triplet_iter = iter(
        triplet_loader
    )

    cross_iter = iter(
        cross_loader
    )


    running = {

        "total":
            0.0,

        "dino":
            0.0,

        "within":
            0.0,

        "cross":
            0.0,

        "spacing":
            0.0,

        "var":
            0.0,

        "cov":
            0.0,

        "within_acc":
            0.0,

        "cross_acc":
            0.0,

        "gspace":
            0.0,

        "lspace":
            0.0,

        "d12":
            0.0,

        "d23":
            0.0,

        "d13":
            0.0,
    }


    cross_bin_correct = {
        0: [],
        1: [],
        2: [],
    }


    n_batches = 0
    n_local_batches = 0


    for dino_batch in dino_loader:

        try:

            within_batch = next(
                within_iter
            )

        except StopIteration:

            within_iter = iter(
                within_loader
            )

            within_batch = next(
                within_iter
            )


        try:

            triplet_batch = next(
                triplet_iter
            )

        except StopIteration:

            triplet_iter = iter(
                triplet_loader
            )

            triplet_batch = next(
                triplet_iter
            )


        try:

            cross_batch = next(
                cross_iter
            )

        except StopIteration:

            cross_iter = iter(
                cross_loader
            )

            cross_batch = next(
                cross_iter
            )


        view1 = (
            dino_batch[
                "view1"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        view2 = (
            dino_batch[
                "view2"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        early = (
            within_batch[
                "early"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        late = (
            within_batch[
                "late"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        x1 = (
            triplet_batch[
                "image1"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        x2 = (
            triplet_batch[
                "image2"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        x3 = (
            triplet_batch[
                "image3"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        delta12 = (
            triplet_batch[
                "delta12"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        delta23 = (
            triplet_batch[
                "delta23"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        younger = (
            cross_batch[
                "younger"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        older = (
            cross_batch[
                "older"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        gap_bin = (
            cross_batch[
                "gap_bin"
            ]
            .cpu()
            .numpy()
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available(),
        ):

            # =================================================
            # DINO
            # =================================================

            (
                student_out1,
                z_view1,
            ) = student(
                view1
            )

            (
                student_out2,
                z_view2,
            ) = student(
                view2
            )


            with torch.no_grad():

                (
                    teacher_out1,
                    _,
                ) = teacher(
                    view1
                )

                (
                    teacher_out2,
                    _,
                ) = teacher(
                    view2
                )


            dino_loss = (

                dino_criterion(
                    student_out1,
                    teacher_out2,
                    args.teacher_temp,
                )

                +

                dino_criterion(
                    student_out2,
                    teacher_out1,
                    args.teacher_temp,
                )

            ) / 2.0


            z_dino = torch.cat(
                [
                    z_view1,
                    z_view2,
                ],
                dim=0,
            )


            var_loss_value = (
                variance_loss(
                    z_dino,
                    args.var_target,
                )
            )


            cov_loss_value = (
                covariance_loss(
                    z_dino
                )
            )


            # =================================================
            # UNIFIED WITHIN-SUBJECT ORDER
            # =================================================

            z_early = student.encoder(
                early
            )

            z_late = student.encoder(
                late
            )


            score_early = (
                developmental_head(
                    z_early
                )
            )

            score_late = (
                developmental_head(
                    z_late
                )
            )


            (
                within_loss,
                within_accuracy,
                within_delta,
            ) = within_subject_order_loss(
                score_early,
                score_late,
                args.within_margin,
            )


            # =================================================
            # UNIFIED CROSS-SUBJECT ORDER
            # =================================================

            z_younger = student.encoder(
                younger
            )

            z_older = student.encoder(
                older
            )


            score_younger = (
                developmental_head(
                    z_younger
                )
            )

            score_older = (
                developmental_head(
                    z_older
                )
            )


            (
                cross_loss,
                cross_accuracy,
                cross_delta,
            ) = cross_subject_order_loss(
                score_younger,
                score_older,
            )


            # =================================================
            # RELATIVE SPACING
            # =================================================

            z1 = student.encoder(
                x1
            )

            z2 = student.encoder(
                x2
            )

            z3 = student.encoder(
                x3
            )


            (
                spacing_loss,
                global_spacing_accuracy,
                local_spacing_accuracy,
                d12,
                d23,
                d13,
            ) = relative_temporal_spacing_loss(

                z1,
                z2,
                z3,

                delta12,
                delta23,

                spacing_projector,

                args.spacing_margin,
            )


            # =================================================
            # TOTAL LOSS
            # =================================================

            total_loss = (

                dino_loss

                +

                args.lambda_within
                *
                within_loss

                +

                args.lambda_cross
                *
                cross_loss

                +

                args.lambda_spacing
                *
                spacing_loss

                +

                args.lambda_var
                *
                var_loss_value

                +

                args.lambda_cov
                *
                cov_loss_value
            )


        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )


        # =====================================================
        # BACKPROP
        # =====================================================

        scaler.scale(
            total_loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(

            list(
                student.parameters()
            )

            +

            list(
                developmental_head.parameters()
            )

            +

            list(
                spacing_projector.parameters()
            ),

            max_norm=5.0,
        )


        scaler.step(
            optimizer
        )

        scaler.update()


        # =====================================================
        # EMA TEACHER
        # =====================================================

        progress = (
            global_step
            /
            max(
                total_steps - 1,
                1,
            )
        )


        momentum = (

            1.0

            -

            (
                1.0
                -
                args.teacher_momentum
            )

            *

            (
                1.0
                +
                math.cos(
                    math.pi
                    *
                    progress
                )
            )

            /

            2.0
        )


        update_teacher(
            student,
            teacher,
            momentum,
        )


        with torch.no_grad():

            dino_criterion.update_center(

                torch.cat(
                    [
                        teacher_out1,
                        teacher_out2,
                    ],
                    dim=0,
                ).float()
            )


        # =====================================================
        # CROSS GAP BIN DIAGNOSTICS
        # =====================================================

        cross_correct = (
            cross_delta > 0
        ).float().cpu().numpy()


        for item_idx, bin_id in enumerate(
            gap_bin
        ):

            cross_bin_correct[
                int(
                    bin_id
                )
            ].append(
                float(
                    cross_correct[
                        item_idx
                    ]
                )
            )


        # =====================================================
        # ACCUMULATE
        # =====================================================

        running[
            "total"
        ] += float(
            total_loss.item()
        )

        running[
            "dino"
        ] += float(
            dino_loss.item()
        )

        running[
            "within"
        ] += float(
            within_loss.item()
        )

        running[
            "cross"
        ] += float(
            cross_loss.item()
        )

        running[
            "spacing"
        ] += float(
            spacing_loss.item()
        )

        running[
            "var"
        ] += float(
            var_loss_value.item()
        )

        running[
            "cov"
        ] += float(
            cov_loss_value.item()
        )

        running[
            "within_acc"
        ] += float(
            within_accuracy.item()
        )

        running[
            "cross_acc"
        ] += float(
            cross_accuracy.item()
        )

        running[
            "gspace"
        ] += float(
            global_spacing_accuracy.item()
        )


        if torch.isfinite(
            local_spacing_accuracy
        ):

            running[
                "lspace"
            ] += float(
                local_spacing_accuracy.item()
            )

            n_local_batches += 1


        running[
            "d12"
        ] += float(
            d12.mean().item()
        )

        running[
            "d23"
        ] += float(
            d23.mean().item()
        )

        running[
            "d13"
        ] += float(
            d13.mean().item()
        )


        n_batches += 1

        global_step += 1


    # ========================================================
    # EPOCH END
    # ========================================================

    scheduler.step()


    n = max(
        n_batches,
        1,
    )


    def avg(key):

        return (
            running[key]
            /
            n
        )


    def safe_mean(values):

        if len(values) == 0:
            return np.nan

        return float(
            np.mean(
                values
            )
        )


    mean_total = avg(
        "total"
    )

    mean_dino = avg(
        "dino"
    )

    mean_within = avg(
        "within"
    )

    mean_cross = avg(
        "cross"
    )

    mean_spacing = avg(
        "spacing"
    )

    mean_var = avg(
        "var"
    )

    mean_cov = avg(
        "cov"
    )

    mean_within_acc = avg(
        "within_acc"
    )

    mean_cross_acc = avg(
        "cross_acc"
    )

    mean_gspace = avg(
        "gspace"
    )

    mean_lspace = (
        running[
            "lspace"
        ]
        /
        max(
            n_local_batches,
            1,
        )
    )

    mean_d12 = avg(
        "d12"
    )

    mean_d23 = avg(
        "d23"
    )

    mean_d13 = avg(
        "d13"
    )


    cross_small = safe_mean(
        cross_bin_correct[0]
    )

    cross_medium = safe_mean(
        cross_bin_correct[1]
    )

    cross_large = safe_mean(
        cross_bin_correct[2]
    )


    row = {

        "epoch":
            epoch,

        "total_loss":
            mean_total,

        "dino_loss":
            mean_dino,

        "unified_within_loss":
            mean_within,

        "unified_cross_loss":
            mean_cross,

        "spacing_loss":
            mean_spacing,

        "variance_loss":
            mean_var,

        "covariance_loss":
            mean_cov,

        "unified_within_accuracy":
            mean_within_acc,

        "unified_cross_accuracy":
            mean_cross_acc,

        "cross_small_accuracy":
            cross_small,

        "cross_medium_accuracy":
            cross_medium,

        "cross_large_accuracy":
            cross_large,

        "global_spacing_accuracy":
            mean_gspace,

        "local_spacing_accuracy":
            mean_lspace,

        "mean_d12":
            mean_d12,

        "mean_d23":
            mean_d23,

        "mean_d13":
            mean_d13,

        "lr":
            optimizer.param_groups[
                0
            ][
                "lr"
            ],

        "teacher_momentum":
            momentum,
    }


    history.append(
        row
    )


    print(
        f"Fold {args.fold} | "
        f"Epoch {epoch:03d} | "
        f"tot {mean_total:.4f} | "
        f"dino {mean_dino:.4f} | "
        f"within {mean_within:.4f} | "
        f"cross {mean_cross:.4f} | "
        f"space {mean_spacing:.4f} | "
        f"WithinAcc {mean_within_acc:.3f} | "
        f"CrossAcc {mean_cross_acc:.3f} | "
        f"small {cross_small:.3f} | "
        f"medium {cross_medium:.3f} | "
        f"large {cross_large:.3f} | "
        f"GSpace {mean_gspace:.3f} | "
        f"LSpace {mean_lspace:.3f} | "
        f"d13 {mean_d13:.3f}"
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_data = {

        "student_state_dict":
            student.state_dict(),

        "teacher_state_dict":
            teacher.state_dict(),

        "student_encoder_state_dict":
            student.encoder.state_dict(),

        "unified_developmental_head_state_dict":
            developmental_head.state_dict(),

        # compatibility alias for later probe
        "developmental_head_state_dict":
            developmental_head.state_dict(),

        "spacing_projector_state_dict":
            spacing_projector.state_dict(),

        "epoch":
            epoch,

        "fold":
            args.fold,

        "mode":
            "unified_developmental_axis",

        "architecture":
            "3d_resnet18",

        "embedding_dim":
            embedding_dim,

        "spacing_dim":
            spacing_dim,

        "config":
            vars(
                args
            ),

        "uses_absolute_age_regression":
            False,

        "uses_exact_delta_regression":
            False,

        "uses_age_bin_classification":
            False,

        "uses_sex_auxiliary":
            False,

        "uses_unified_developmental_axis":
            True,
    }


    if (
        epoch
        %
        args.save_every
        ==
        0
    ):

        torch.save(
            checkpoint_data,
            CHECKPOINT_DIR
            /
            f"epoch_{epoch}.pt",
        )


    # Train-loss best is saved for completeness,
    # but downstream checkpoint selection should not rely on this alone.
    if mean_total < best_total_loss:

        best_total_loss = (
            mean_total
        )

        best_epoch = (
            epoch
        )

        torch.save(
            checkpoint_data,
            BEST_PATH,
        )


# ============================================================
# FINAL
# ============================================================

final_checkpoint = {

    "student_state_dict":
        student.state_dict(),

    "teacher_state_dict":
        teacher.state_dict(),

    "student_encoder_state_dict":
        student.encoder.state_dict(),

    "unified_developmental_head_state_dict":
        developmental_head.state_dict(),

    "developmental_head_state_dict":
        developmental_head.state_dict(),

    "spacing_projector_state_dict":
        spacing_projector.state_dict(),

    "epoch":
        args.epochs,

    "fold":
        args.fold,

    "mode":
        "unified_developmental_axis",

    "architecture":
        "3d_resnet18",

    "embedding_dim":
        embedding_dim,

    "spacing_dim":
        spacing_dim,

    "n_train_mri":
        len(
            train_df
        ),

    "n_train_subjects":
        len(
            train_subjects
        ),

    "n_within_pairs":
        len(
            within_df
        ),

    "n_triplets":
        len(
            triplet_df
        ),

    "n_cross_pairs":
        len(
            cross_df
        ),

    "uses_absolute_age_regression":
        False,

    "uses_exact_delta_regression":
        False,

    "uses_age_bin_classification":
        False,

    "uses_sex_auxiliary":
        False,

    "uses_unified_developmental_axis":
        True,

    "config":
        vars(
            args
        ),
}


torch.save(
    final_checkpoint,
    FINAL_PATH,
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "model":
        "TempoDINO-v7-unified-axis",

    "fold":
        args.fold,

    "initialization":
        str(
            init_path
        ),

    "best_train_loss_epoch":
        best_epoch,

    "best_train_loss":
        best_total_loss,

    "n_train_mri":
        len(
            train_df
        ),

    "n_train_subjects":
        len(
            train_subjects
        ),

    "n_heldout_mri":
        len(
            heldout_df
        ),

    "n_heldout_subjects":
        len(
            heldout_subjects
        ),

    "n_within_pairs":
        len(
            within_df
        ),

    "n_triplets":
        len(
            triplet_df
        ),

    "n_cross_pairs":
        len(
            cross_df
        ),

    "lambda_within":
        args.lambda_within,

    "lambda_cross":
        args.lambda_cross,

    "lambda_spacing":
        args.lambda_spacing,

    "final_unified_within_accuracy":
        history[-1][
            "unified_within_accuracy"
        ],

    "final_unified_cross_accuracy":
        history[-1][
            "unified_cross_accuracy"
        ],

    "final_global_spacing_accuracy":
        history[-1][
            "global_spacing_accuracy"
        ],

    "final_local_spacing_accuracy":
        history[-1][
            "local_spacing_accuracy"
        ],

    "uses_absolute_age_regression":
        False,

    "uses_sex_auxiliary":
        False,

    "primary_training_goal":
        (
            "longitudinal developmental representation learning"
        ),
}


with open(
    SUMMARY_PATH,
    "w",
) as handle:

    json.dump(
        summary,
        handle,
        indent=2,
    )


# ============================================================
# GPU
# ============================================================

if torch.cuda.is_available():

    print("\n" + "=" * 80)
    print("GPU MEMORY")
    print("=" * 80)

    print(
        "Peak allocated:",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )

    print(
        "Peak reserved: ",
        f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GB"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("TEMPODINO V7 TRAINING COMPLETE")
print("=" * 80)

print(
    f"Best train-loss epoch: {best_epoch}"
)

print(
    f"Best train loss:       {best_total_loss:.6f}"
)

print(
    "\nFinal checkpoint:"
)

print(
    FINAL_PATH
)

print(
    "\nBest checkpoint:"
)

print(
    BEST_PATH
)

print(
    "\nHistory:"
)

print(
    HISTORY_PATH
)

print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)

print(
    "\nDONE."
)
