# Analysis

This folder contains the feature-representation comparison pipeline for hippocampal sclerosis classification.

## Input

- Default dataset: `../data/full_dataset_111patients.csv`
- Expected columns:
  - `Subject`
  - bilateral features as `*_Left` and `*_Right`
  - `Label` in `{-1, 1}` or `{0, 1}`

## Run

```bash
cd analysis
python3 run_feature_engineering_comparison.py \
  --input ../data/full_dataset_111patients.csv \
  --outdir results_repeat10
```

To regenerate manuscript summary files from an existing `fold_metrics.csv` without refitting models:

```bash
cd analysis
python3 run_feature_engineering_comparison.py \
  --existing-fold-metrics results_repeat10/fold_metrics.csv \
  --outdir results_repeat10
```

## Outputs

- `results_repeat10/fold_metrics.csv`
- `results_repeat10/summary_metrics.csv`
- `results_repeat10/effect_sizes.csv`
- `results_repeat10/feature_set_diagnostics.csv`
- `results_repeat10/roc_auc_heatmap.png`
- `results_repeat10/feature_set_rank.csv`
- `results_repeat10/model_rank.csv`
- `results_repeat10/run_config.json`
