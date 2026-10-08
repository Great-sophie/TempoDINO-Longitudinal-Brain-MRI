# Methods (TempoDINO-v7)

## Dataset and folds
The research dataset includes 114 longitudinal 3D T1-weighted non-human-primate brain MRI scans from 23 subjects. Age is encoded in months. Five subject-disjoint folds were defined in the historical metadata. For each held-out fold, all scans from its subjects were excluded from model training. Per-fold MRI counts were 25, 25, 25, 20, and 19. MRI volume arrays are stored in the historical environment as preprocessed `.npy` files; no imaging data are redistributed here.

## Architecture
The uploaded v7 training source implements a GroupNorm-based 3D ResNet-18 encoder, a DINO student/EMA teacher, and a DINO projection head. It also implements a single scalar unified developmental head `s(z)` for both within- and cross-subject ordering and a normalized projector for temporal spacing. V7 initializes from fold-specific V6 cross-ordinal epoch-2 checkpoints.

## Learning objective
The optimization combines DINO self-distillation, positive within-subject order-margin loss, cross-subject softplus ordinal loss, relative temporal-spacing triplet loss, and variance/covariance regularization. It does not directly regress chronological age, exact interscan month differences, sex, or age bins.

## Pair/triplet construction
Pairs and triplets are created from training subjects only. Within-subject pairs have strictly increasing age, triplets satisfy `age1 < age2 < age3`. Cross-subject pairs have at least a 2-month age gap and are balanced across [2,4), [4,8), and [8,+inf) month intervals via seeded sampling.

## Evaluation
The locked OOF aggregator reads per-fold `v7_final` probe outputs and reports ordering, developmental-score/age correlations, spacing geometry, and downstream Ridge age-probe metrics. Age-probe fitting and MRI feature extraction are performed by the historical `19b_probe_tempodino_v7_unified_axis.py`, which should be copied and path-migrated locally before independent inference. Aggregate counts and metrics in `results/` are historical observations; they are not newly generated synthetic estimates.

## Scope and limitations
The tested cohort is small; fold-level performance varies. The average test effective rank is low (1.444), so representation dimensionality/collapse should be investigated. No external test cohort or claim of superiority over DINO/V6 is asserted here. The historical data license, species details, acquisition parameters, and preprocessing provenance need confirmation from the study records before publication.
