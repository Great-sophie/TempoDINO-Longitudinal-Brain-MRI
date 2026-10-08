# Reproducibility record

## Verified on original workstation (user-reported execution)
- 114/114 metadata MRI paths resolve following relocation.
- V6 `epoch_2.pt` and V7 `final.pt` exist for all five folds.
- Five-fold locked OOF aggregation from archived probe CSV was regenerated in an independent `local_runs` directory.
- Historical vs regenerated *primary* and *secondary* summary metric values: 0 differences; `locked_complete=True`.
- Checkpoint-based GPU inference plus Ridge age probe was rerun across five folds. Of 170 compared fold-level numeric fields, 166 were within the chosen numerical tolerance (`rtol=1e-3`, `atol=2e-3`); four differences occurred in fold 2.
- A near-tie consecutive pair in fold 2 (12→16 months) changed score difference from zero to ~0.000122, flipping one strict ordering decision. Maximum per-scan score difference for fold 2 was ~0.000977, mean ~0.000142. This accounts for the 0.02 difference in pair accuracy and 0.05 difference in mean subject consecutive-order accuracy; remaining correlation differences are not ascribed solely to this pair.

## Reproduce aggregate metrics without GPU
Set `TEMPODINO_PROJECT_DIR` to the historical `monkey_development_scripit` containing `results/tempodino_v7_unified_axis_probe/fold_*/v7_final_*.csv`; set `TEMPODINO_RUN_ROOT` to the release's local output directory. Run `python scripts/20_evaluate_tempodino_v7_oof.py`. Generated files stay in `local_runs/results/tempodino_v7_oof/`. This reproduces *aggregation*, not MRI inference or model training.

## Reproduce checkpoint inference
The probe source was only available on the user's Ubuntu workstation during preparation, so it is **not included in this downloadable archive**. On that workstation, transfer the already path-migrated `scripts/19b_probe_tempodino_v7_unified_axis.py` from the previous release directory, review it, and run `python -m py_compile` first. The probe requires local preprocessed data and trusted checkpoint files.

For each fold, use `--fold N --checkpoint v7_final --batch_size 4 --num_workers 2 --ridge_alpha 10`. Archive historical outputs separately; do not overwrite them. Never change checkpoint policy to select best-performing held-out results.

## Security and leakage boundaries
The historical V7 training script splits subjects before constructing within- and cross-subject pairs; a subject overlap guard raises on leakage. The OOF aggregator checks subject exclusivity, duplicate subject/age values, finite predictions, and total counts. Check upstream preprocessing, V6 initialization, and age-probe fitting independently before claiming end-to-end leak-free methodology. `torch.load(weights_only=False)` must only be used with trusted locally generated checkpoints.
