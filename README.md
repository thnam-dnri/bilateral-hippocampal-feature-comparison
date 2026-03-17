# Bilateral Hippocampal Feature Comparison

This repository contains the data, analysis code, and generated results for the study:

`How to Represent Bilateral Hippocampal Morphometry for Machine Learning: A Systematic Feature Engineering Comparison`

## Repository Layout

- `data/`
  - input dataset used for the manuscript analysis
- `analysis/`
  - feature-engineering comparison script
  - Python requirements
  - reproducible result files used in the manuscript

## Reproduce the Analysis

Install dependencies:

```bash
pip install -r analysis/requirements.txt
```

Run the full repeated nested cross-validation analysis:

```bash
cd analysis
python3 run_feature_engineering_comparison.py \
  --input ../data/full_dataset_111patients.csv \
  --outdir results_repeat10
```

Regenerate summary files from the stored `fold_metrics.csv` without refitting models:

```bash
cd analysis
python3 run_feature_engineering_comparison.py \
  --existing-fold-metrics results_repeat10/fold_metrics.csv \
  --outdir results_repeat10
```

## Included Outputs

The committed `analysis/results_repeat10/` directory contains the outputs used in the manuscript, including:

- `fold_metrics.csv`
- `summary_metrics.csv`
- `effect_sizes.csv`
- `feature_set_diagnostics.csv`
- `feature_set_rank.csv`
- `model_rank.csv`
- `roc_auc_heatmap.png`
- `run_config.json`
