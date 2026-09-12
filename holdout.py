import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_cv import FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--cohort-col", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    train = df[df[args.cohort_col].astype(str) == args.train].copy()
    test = df[df[args.cohort_col].astype(str) == args.test].copy()

    train = train.dropna(subset=["Name", "VIBE1"] + FEATURES)
    test = test.dropna(subset=["Name", "VIBE1"] + FEATURES)

    overlap = set(train["Name"].astype(str)) & set(test["Name"].astype(str))
    if overlap:
        test = test[~test["Name"].astype(str).isin(overlap)].copy()

    gate = train["VIBE1"].std(ddof=1)

    y_train = (train["VIBE1"] >= gate).astype(int)
    y_test = (test["VIBE1"] >= gate).astype(int)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=args.c,
            l1_ratio=args.l1_ratio,
            max_iter=5000,
            random_state=42
        ))
    ])

    model.fit(train[FEATURES], y_train)
    pred = model.predict_proba(test[FEATURES])[:, 1]

    result = pd.DataFrame({
        "Name": test["Name"].values,
        "VIBE1": test["VIBE1"].values,
        "high_vibe": y_test.values,
        "model_score": pred,
        "GRAVY_VH": test["GRAVY_VH"].values
    })

    Path("results").mkdir(exist_ok=True)
    result.to_csv("results/holdout_predictions.csv", index=False)

    print("train n:", len(train))
    print("test n:", len(test))
    print("removed overlap:", len(overlap))
    print("gate:", gate)
    print("model ROC-AUC:", roc_auc_score(y_test, pred))
    print("model PR-AUC:", average_precision_score(y_test, pred))
    print("GRAVY ROC-AUC:", roc_auc_score(y_test, test["GRAVY_VH"]))
    print("GRAVY PR-AUC:", average_precision_score(y_test, test["GRAVY_VH"]))


if __name__ == "__main__":
    main()
