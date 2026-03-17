#!/usr/bin/env python3
"""
Feature representation comparison for HS classification.

Compares 7 bilateral representation strategies across LR-EN, RF, XGBoost
using repeated nested cross-validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering comparison for hippocampal sclerosis classification")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "full_dataset_111patients.csv"),
        help="Path to full 111-patient dataset",
    )
    parser.add_argument("--id-col", type=str, default="Subject", help="Subject ID column")
    parser.add_argument("--label-col", type=str, default="Label", help="Label column")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--outer-folds", type=int, default=5, help="Outer CV folds")
    parser.add_argument("--inner-folds", type=int, default=3, help="Inner CV folds")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated outer CV runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV")
    parser.add_argument(
        "--existing-fold-metrics",
        type=str,
        default="",
        help="Optional path to an existing fold_metrics.csv; if provided, model fitting is skipped and summary outputs are regenerated from that file.",
    )
    return parser.parse_args()


def map_labels(y_raw: pd.Series) -> np.ndarray:
    unique_vals = set(pd.unique(y_raw))
    if unique_vals == {-1, 1}:
        return y_raw.map({-1: 0, 1: 1}).astype(int).to_numpy()
    if unique_vals == {0, 1}:
        return y_raw.astype(int).to_numpy()
    text_map = {
        "confirmed": 1,
        "no illness": 0,
        "illness": 1,
        "healthy": 0,
        "hs+": 1,
        "hs-": 0,
    }
    mapped = y_raw.astype(str).str.strip().str.lower().map(text_map)
    if mapped.isna().any():
        raise ValueError(f"Unsupported labels: {sorted(unique_vals)}")
    return mapped.astype(int).to_numpy()


def infer_pairs(df: pd.DataFrame, id_col: str, label_col: str) -> List[Tuple[str, str, str]]:
    left_cols = [col for col in df.columns if col.endswith("_Left")]
    pairs: List[Tuple[str, str, str]] = []
    for left_col in sorted(left_cols):
        base = left_col[: -len("_Left")]
        right_col = f"{base}_Right"
        if right_col in df.columns and left_col not in {id_col, label_col} and right_col not in {id_col, label_col}:
            pairs.append((base, left_col, right_col))
    if not pairs:
        raise ValueError("No *_Left/*_Right feature pairs found.")
    return pairs


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den_arr = den.to_numpy(float)
    out = np.zeros(len(den_arr), dtype=float)
    mask = np.abs(den_arr) > 1e-12
    out[mask] = (num.to_numpy(float)[mask]) / den_arr[mask]
    return pd.Series(out, index=num.index)


def build_feature_sets(df: pd.DataFrame, pairs: List[Tuple[str, str, str]]) -> Dict[str, pd.DataFrame]:
    raw_cols: List[str] = []
    ai_df = pd.DataFrame(index=df.index)
    diff_df = pd.DataFrame(index=df.index)
    ratio_df = pd.DataFrame(index=df.index)
    sum_df = pd.DataFrame(index=df.index)
    mean_df = pd.DataFrame(index=df.index)

    for base, left_col, right_col in pairs:
        left = pd.to_numeric(df[left_col], errors="coerce")
        right = pd.to_numeric(df[right_col], errors="coerce")
        raw_cols.extend([left_col, right_col])

        den = left + right
        ai_df[f"{base}_AI"] = _safe_divide(right - left, den)
        diff_df[f"{base}_Diff"] = right - left
        ratio_df[f"{base}_Ratio_LR"] = _safe_divide(left, right.replace(0.0, np.nan).fillna(0.0))
        sum_df[f"{base}_Sum"] = den
        mean_df[f"{base}_Mean"] = den / 2.0

    raw_df = df[raw_cols].apply(pd.to_numeric, errors="coerce")

    feature_sets = {
        "A_raw_lr": raw_df,
        "B_sum_mean_ai": pd.concat([sum_df, mean_df, ai_df], axis=1),
        "C_ai_only": ai_df,
        "D_diff_only": diff_df,
        "E_ratio_only": ratio_df,
        "F_combined_acde": pd.concat([raw_df, ai_df, diff_df, ratio_df], axis=1),
        "G_mean_ai": pd.concat([mean_df, ai_df], axis=1),
    }

    for set_name, features in feature_sets.items():
        feature_sets[set_name] = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if feature_sets[set_name].shape[1] == 0:
            raise ValueError(f"Feature set {set_name} has no columns.")
    return feature_sets


def compute_scale_pos_weight(y: np.ndarray) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return float(negatives / positives)


def build_models(seed: int, scale_pos_weight: float) -> List[Tuple[str, Pipeline, Dict[str, List[float]]]]:
    models: List[Tuple[str, Pipeline, Dict[str, List[float]]]] = []

    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    lr_grid = {"model__C": [0.01, 0.1, 1.0, 10.0], "model__l1_ratio": [0.1, 0.5, 0.9]}
    models.append(("lr_en", lr, lr_grid))

    rf = Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            )
        ]
    )
    rf_grid = {"model__n_estimators": [200, 500], "model__max_depth": [None, 3, 5], "model__min_samples_leaf": [1, 2]}
    models.append(("random_forest", rf, rf_grid))

    try:
        from xgboost import XGBClassifier

        xgb = Pipeline(
            steps=[
                (
                    "model",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=1,
                        tree_method="hist",
                        scale_pos_weight=scale_pos_weight,
                    ),
                )
            ]
        )
        xgb_grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
        }
        models.append(("xgboost", xgb, xgb_grid))
    except Exception as exc:
        print(f"Warning: xgboost is unavailable and will be skipped: {exc}")

    return models


def safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, y_score))
    except Exception:
        return float("nan")


def _standardize_matrix(x_data: np.ndarray) -> np.ndarray:
    mu = x_data.mean(axis=0)
    sigma = x_data.std(axis=0, ddof=0)
    sigma[sigma < 1e-12] = 1.0
    return (x_data - mu) / sigma


def _effective_dimensionality(x_data: np.ndarray, variance_target: float = 0.95) -> int:
    if x_data.shape[1] == 0:
        return 0
    x_std = _standardize_matrix(x_data)
    cov = np.cov(x_std, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.clip(eigvals, 0.0, None))[::-1]
    total = float(eigvals.sum())
    if total <= 0.0:
        return 0
    cumulative = np.cumsum(eigvals) / total
    return int(np.searchsorted(cumulative, variance_target) + 1)


def _correlation_summary(x_data: np.ndarray) -> Dict[str, float]:
    if x_data.shape[1] < 2:
        return {
            "median_abs_correlation": 0.0,
            "p95_abs_correlation": 0.0,
            "max_abs_correlation": 0.0,
        }
    corr = np.corrcoef(x_data, rowvar=False)
    upper = np.triu_indices(corr.shape[0], k=1)
    values = np.abs(corr[upper])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "median_abs_correlation": float("nan"),
            "p95_abs_correlation": float("nan"),
            "max_abs_correlation": float("nan"),
        }
    return {
        "median_abs_correlation": float(np.median(values)),
        "p95_abs_correlation": float(np.percentile(values, 95)),
        "max_abs_correlation": float(np.max(values)),
    }


def _vif_summary(x_data: np.ndarray) -> Dict[str, float]:
    x_std = _standardize_matrix(x_data)
    n_features = x_std.shape[1]
    vifs: List[float] = []
    for idx in range(n_features):
        y_vec = x_std[:, idx]
        x_other = np.delete(x_std, idx, axis=1)
        if x_other.shape[1] == 0:
            vifs.append(1.0)
            continue
        design = np.column_stack([np.ones(len(y_vec)), x_other])
        coef, *_ = np.linalg.lstsq(design, y_vec, rcond=None)
        fitted = design @ coef
        ss_res = float(np.sum((y_vec - fitted) ** 2))
        ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
        if ss_tot <= 1e-12:
            vifs.append(float("inf"))
            continue
        r_squared = 1.0 - (ss_res / ss_tot)
        if (1.0 - r_squared) <= 1e-10:
            vifs.append(float("inf"))
        else:
            vifs.append(float(1.0 / (1.0 - r_squared)))

    finite_vifs = [val for val in vifs if np.isfinite(val)]
    return {
        "median_vif": float(np.median(finite_vifs)) if finite_vifs else float("inf"),
        "max_vif": float(np.max(finite_vifs)) if finite_vifs else float("inf"),
        "n_infinite_vif": int(sum(not np.isfinite(val) for val in vifs)),
    }


def compute_feature_set_diagnostics(feature_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature_set_name, x_df in feature_sets.items():
        x_data = x_df.to_numpy(float)
        corr_summary = _correlation_summary(x_data)
        vif_summary = _vif_summary(x_data)
        rows.append(
            {
                "feature_set": feature_set_name,
                "n_features": int(x_data.shape[1]),
                "effective_dim_95": _effective_dimensionality(x_data, variance_target=0.95),
                "condition_number": float(np.linalg.cond(_standardize_matrix(x_data))),
                **corr_summary,
                **vif_summary,
            }
        )
    return pd.DataFrame(rows).sort_values(["effective_dim_95", "n_features", "feature_set"]).reset_index(drop=True)


def summarize_fold_metrics(fold_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = (
        fold_df.groupby(["feature_set", "model"], dropna=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            n_eval=("roc_auc", "count"),
        )
        .reset_index()
        .sort_values(["roc_auc_mean", "average_precision_mean"], ascending=[False, False])
    )

    feature_rank = (
        summary_df.groupby("feature_set")
        .agg(
            mean_roc_auc=("roc_auc_mean", "mean"),
            mean_pr_auc=("average_precision_mean", "mean"),
            mean_brier=("brier_mean", "mean"),
        )
        .sort_values(["mean_roc_auc", "mean_pr_auc"], ascending=[False, False])
        .reset_index()
    )

    model_rank = (
        summary_df.groupby("model")
        .agg(
            mean_roc_auc=("roc_auc_mean", "mean"),
            mean_pr_auc=("average_precision_mean", "mean"),
            mean_brier=("brier_mean", "mean"),
        )
        .sort_values(["mean_roc_auc", "mean_pr_auc"], ascending=[False, False])
        .reset_index()
    )
    return summary_df, feature_rank, model_rank


def compute_effect_decomposition(fold_df: pd.DataFrame) -> pd.DataFrame:
    overall = float(fold_df["roc_auc"].mean())
    feature_means = fold_df.groupby("feature_set")["roc_auc"].mean()
    model_means = fold_df.groupby("model")["roc_auc"].mean()
    cell_means = fold_df.groupby(["feature_set", "model"])["roc_auc"].mean()

    ss_total = float(((fold_df["roc_auc"] - overall) ** 2).sum())
    ss_feature = float((fold_df.groupby("feature_set").size() * ((feature_means - overall) ** 2)).sum())
    ss_model = float((fold_df.groupby("model").size() * ((model_means - overall) ** 2)).sum())

    ss_interaction = 0.0
    ss_error = 0.0
    for (feature_set_name, model_name), cell_values in fold_df.groupby(["feature_set", "model"])["roc_auc"]:
        cell_mean = float(cell_means.loc[(feature_set_name, model_name)])
        ss_interaction += float(
            len(cell_values)
            * (cell_mean - float(feature_means.loc[feature_set_name]) - float(model_means.loc[model_name]) + overall) ** 2
        )
        ss_error += float(((cell_values - cell_mean) ** 2).sum())

    def _partial_eta_squared(sum_squares: float) -> float:
        denom = sum_squares + ss_error
        return float(sum_squares / denom) if denom > 0 else float("nan")

    return pd.DataFrame(
        [
            {
                "component": "feature_set",
                "sum_squares": ss_feature,
                "proportion_of_total": ss_feature / ss_total if ss_total > 0 else np.nan,
                "partial_eta_squared": _partial_eta_squared(ss_feature),
            },
            {
                "component": "model",
                "sum_squares": ss_model,
                "proportion_of_total": ss_model / ss_total if ss_total > 0 else np.nan,
                "partial_eta_squared": _partial_eta_squared(ss_model),
            },
            {
                "component": "interaction",
                "sum_squares": ss_interaction,
                "proportion_of_total": ss_interaction / ss_total if ss_total > 0 else np.nan,
                "partial_eta_squared": _partial_eta_squared(ss_interaction),
            },
            {
                "component": "residual_error",
                "sum_squares": ss_error,
                "proportion_of_total": ss_error / ss_total if ss_total > 0 else np.nan,
                "partial_eta_squared": np.nan,
            },
        ]
    )


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    sid = df[args.id_col].astype(str).to_numpy()
    y = map_labels(df[args.label_col])
    pairs = infer_pairs(df, args.id_col, args.label_col)
    feature_sets = build_feature_sets(df, pairs)
    scale_pos_weight = compute_scale_pos_weight(y)

    if args.existing_fold_metrics:
        fold_df = pd.read_csv(args.existing_fold_metrics)
        required_cols = {
            "feature_set",
            "model",
            "roc_auc",
            "average_precision",
            "brier",
            "balanced_accuracy",
        }
        missing_cols = required_cols.difference(fold_df.columns)
        if missing_cols:
            raise ValueError(f"Existing fold metrics file is missing columns: {sorted(missing_cols)}")
        models_run = sorted(fold_df["model"].dropna().astype(str).unique().tolist())
    else:
        models = build_models(args.seed, scale_pos_weight)
        if not models:
            raise RuntimeError("No models available. Install scikit-learn and xgboost.")

        fold_rows: List[Dict[str, object]] = []
        for repeat in range(args.repeats):
            outer_cv = StratifiedKFold(
                n_splits=args.outer_folds,
                shuffle=True,
                random_state=args.seed + repeat,
            )
            for feature_set_name, x_df in feature_sets.items():
                x_all = x_df.to_numpy(float)
                for model_name, estimator, param_grid in models:
                    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x_all, y), start=1):
                        x_train, x_test = x_all[train_idx], x_all[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        inner_cv = StratifiedKFold(
                            n_splits=args.inner_folds,
                            shuffle=True,
                            random_state=args.seed + repeat + 100 * outer_fold,
                        )
                        estimator_fold = clone(estimator)
                        if model_name == "xgboost":
                            estimator_fold.set_params(model__scale_pos_weight=compute_scale_pos_weight(y_train))
                        search = GridSearchCV(
                            estimator=estimator_fold,
                            param_grid=param_grid,
                            scoring="roc_auc",
                            cv=inner_cv,
                            n_jobs=args.n_jobs,
                            refit=True,
                        )
                        search.fit(x_train, y_train)
                        prob = search.predict_proba(x_test)[:, 1]
                        pred = (prob >= 0.5).astype(int)

                        fold_rows.append(
                            {
                                "repeat": repeat + 1,
                                "outer_fold": outer_fold,
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "n_train": int(len(train_idx)),
                                "n_test": int(len(test_idx)),
                                "roc_auc": safe_metric(roc_auc_score, y_test, prob),
                                "average_precision": safe_metric(average_precision_score, y_test, prob),
                                "brier": safe_metric(brier_score_loss, y_test, prob),
                                "balanced_accuracy": safe_metric(balanced_accuracy_score, y_test, pred),
                                "best_params": json.dumps(search.best_params_),
                                "subject_ids_test": ";".join(sid[test_idx].tolist()),
                            }
                        )

        fold_df = pd.DataFrame(fold_rows)
        models_run = [name for name, _, _ in models]

    fold_df.to_csv(outdir / "fold_metrics.csv", index=False)

    summary_df, feature_rank, model_rank = summarize_fold_metrics(fold_df)
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    feature_rank.to_csv(outdir / "feature_set_rank.csv", index=False)
    model_rank.to_csv(outdir / "model_rank.csv", index=False)

    diagnostics_df = compute_feature_set_diagnostics(feature_sets)
    diagnostics_df.to_csv(outdir / "feature_set_diagnostics.csv", index=False)

    effect_sizes = compute_effect_decomposition(fold_df)
    effect_sizes.to_csv(outdir / "effect_sizes.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        heat = summary_df.pivot(index="feature_set", columns="model", values="roc_auc_mean")
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("ROC-AUC by Feature Set and Model")
        fig.savefig(outdir / "roc_auc_heatmap.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to create ROC heatmap: {exc}")

    run_cfg = {
        "input": str(args.input),
        "id_col": args.id_col,
        "label_col": args.label_col,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "existing_fold_metrics": args.existing_fold_metrics,
        "xgboost_scale_pos_weight": scale_pos_weight,
        "pairs_detected": [base for base, _, _ in pairs],
        "models_run": models_run,
    }
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"Completed. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    run(parse_args())
