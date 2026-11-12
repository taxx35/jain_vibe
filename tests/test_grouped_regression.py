import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "analysis" / "grouped_regression.py"
spec = importlib.util.spec_from_file_location("grouped_regression", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_small_problem(seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(10), 4)
    X = rng.normal(size=(40, 6))
    y = X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.2, size=40)
    return X, y, groups


def test_perfect_monotonic_prediction_has_spearman_one():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    pred = np.array([10, 20, 30, 40, 50], dtype=float)
    assert np.isclose(mod.regression_metrics(y, pred)["spearman"], 1.0, atol=1e-12)


def test_group_overlap_is_rejected():
    groups = np.array(["A", "A", "B", "B"])
    train_idx = np.array([0, 2])
    test_idx = np.array([1, 3])
    try:
        mod.assert_disjoint_groups(train_idx, test_idx, groups)
    except ValueError:
        return
    raise AssertionError("expected group-leakage check to fail")


def test_oof_prediction_covers_every_sample():
    X, y, groups = make_small_problem()
    oof, fold_id, folds, metrics = mod.nested_grouped_regression(
        X,
        y,
        groups,
        outer_splits=5,
        inner_splits=3,
        pca_grid=(2, 4),
        alpha_grid=(0.1, 1.0),
    )
    assert len(oof) == len(y)
    assert not np.isnan(oof).any()
    assert np.all(fold_id > 0)
    assert len(folds) == 5
    assert set(metrics) == {"spearman", "mae", "rmse", "r2"}


def test_other_heldout_samples_do_not_change_one_prediction():
    X, y, groups = make_small_problem(seed=3)

    oof1, fold_id1, _, _ = mod.nested_grouped_regression(
        X,
        y,
        groups,
        outer_splits=5,
        inner_splits=3,
        pca_grid=(2, 4),
        alpha_grid=(0.1, 1.0),
    )

    target_index = 0
    target_fold = fold_id1[target_index]
    same_test_fold = np.where(fold_id1 == target_fold)[0]
    other_test_rows = same_test_fold[same_test_fold != target_index]

    X_perturbed = X.copy()
    X_perturbed[other_test_rows] += 1_000_000.0

    oof2, fold_id2, _, _ = mod.nested_grouped_regression(
        X_perturbed,
        y,
        groups,
        outer_splits=5,
        inner_splits=3,
        pca_grid=(2, 4),
        alpha_grid=(0.1, 1.0),
    )

    assert np.array_equal(fold_id1, fold_id2)
    assert np.isclose(oof1[target_index], oof2[target_index], atol=1e-10, rtol=0.0)
