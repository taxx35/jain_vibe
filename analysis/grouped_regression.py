import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PCA_GRID = (2, 4, 8, 16, 32)
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 300.0)


def assert_disjoint_groups(train_idx, test_idx, groups):
    groups = np.asarray(groups)
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise ValueError(f"group leakage detected: {sorted(overlap)[:5]}")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def nested_grouped_regression(
    X,
    y,
    groups,
    outer_splits=5,
    inner_splits=4,
    pca_grid=PCA_GRID,
    alpha_grid=ALPHA_GRID,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    if len(np.unique(groups)) < outer_splits:
        raise ValueError("not enough independent groups for requested outer folds")

    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    fold_rows = []

    outer = GroupKFold(n_splits=outer_splits)

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(X, y, groups), start=1
    ):
        assert_disjoint_groups(train_idx, test_idx, groups)

        train_groups = groups[train_idx]
        n_train_groups = len(np.unique(train_groups))
        inner_k = min(inner_splits, n_train_groups)
        if inner_k < 2:
            raise ValueError("not enough training groups for inner cross-validation")

        max_components = min(X.shape[1], len(train_idx) - 1)
        valid_pca = [n for n in pca_grid if n <= max_components]
        if not valid_pca:
            valid_pca = [max_components]

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA()),
            ("ridge", Ridge()),
        ])

        search = GridSearchCV(
            estimator=pipe,
            param_grid={
                "pca__n_components": valid_pca,
                "ridge__alpha": list(alpha_grid),
            },
            cv=GroupKFold(n_splits=inner_k),
            scoring="neg_mean_absolute_error",
            n_jobs=None,
            refit=True,
        )
        search.fit(X[train_idx], y[train_idx], groups=train_groups)

        pred = search.predict(X[test_idx])
        oof[test_idx] = pred
        fold_id[test_idx] = fold

        train_counts = pd.Series(groups[train_idx]).value_counts()
        test_counts = pd.Series(groups[test_idx]).value_counts()

        row = {
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_groups": int(len(train_counts)),
            "n_test_groups": int(len(test_counts)),
            "largest_train_group_fraction": float(train_counts.max() / len(train_idx)),
            "largest_test_group_fraction": float(test_counts.max() / len(test_idx)),
            "pca_components": int(search.best_params_["pca__n_components"]),
            "ridge_alpha": float(search.best_params_["ridge__alpha"]),
        }
        row.update({f"test_{k}": v for k, v in regression_metrics(y[test_idx], pred).items()})
        fold_rows.append(row)

    if np.isnan(oof).any() or np.any(fold_id < 0):
        raise RuntimeError("some samples did not receive exactly one outer-fold prediction")

    metrics = regression_metrics(y, oof)
    return oof, fold_id, pd.DataFrame(fold_rows), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--target", default="VIBE1")
    parser.add_argument("--group-col", default="sequence_group")
    parser.add_argument("--feature-prefix", default="feat_")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--outdir", default="results/grouped_regression")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    if not feature_cols:
        raise ValueError("no feature columns matched the requested prefix")

    needed = [args.target, args.group_col] + feature_cols
    work = df.dropna(subset=needed).reset_index(drop=True)

    y = pd.to_numeric(work[args.target], errors="raise").to_numpy(dtype=float)
    if args.log_target:
        if np.any(y <= 0):
            raise ValueError("log target requested but target contains non-positive values")
        y = np.log(y)

    X = work[feature_cols].to_numpy(dtype=float)
    groups = work[args.group_col].astype(str).to_numpy()

    oof, fold_id, folds, metrics = nested_grouped_regression(
        X,
        y,
        groups,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred = work[[args.group_col]].copy()
    if "antibody_id" in work.columns:
        pred.insert(0, "antibody_id", work["antibody_id"])
    pred["outer_fold"] = fold_id
    pred["target_used"] = y
    pred["oof_prediction"] = oof

    pred.to_csv(outdir / "oof_predictions.csv", index=False)
    folds.to_csv(outdir / "fold_diagnostics.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
