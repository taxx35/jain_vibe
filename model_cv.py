import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "GRAVY_VH",
    "SAP_pos_Fv",
    "GRAVY_HCDR3",
    "SAP_pos_Hv",
    "SAP_pos_Lv",
]


def fit_model(c, l1_ratio, class_weight=None):
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=c,
            l1_ratio=l1_ratio,
            class_weight=class_weight,
            max_iter=5000,
            random_state=42
        ))
    ])


def cv_scores(x, y, label, c, l1_ratio, class_weight=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for fold, (tr, te) in enumerate(cv.split(x, y), 1):
        model = fit_model(c, l1_ratio, class_weight)
        model.fit(x.iloc[tr], y.iloc[tr])
        pred = model.predict_proba(x.iloc[te])[:, 1]

        rows.append({
            "model": label,
            "fold": fold,
            "roc_auc": roc_auc_score(y.iloc[te], pred),
            "pr_auc": average_precision_score(y.iloc[te], pred)
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--gate", type=float)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.2)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["Status"].isin(["Approved", "Terminated"])].copy()
    df = df.dropna(subset=["VIBE1"] + FEATURES).reset_index(drop=True)

    gate = args.gate
    if gate is None:
        gate = df["VIBE1"].std(ddof=1)

    y = (df["VIBE1"] >= gate).astype(int)
    cw = "balanced" if args.balanced else None

    rows = []
    rows += cv_scores(df[FEATURES], y, "five_feature", args.c, args.l1_ratio, cw)
    rows += cv_scores(df[["GRAVY_VH"]], y, "GRAVY_VH", args.c, args.l1_ratio, cw)

    out = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/model_cv_folds.csv", index=False)

    summary = out.groupby("model")[["roc_auc", "pr_auc"]].agg(["mean", "std"])
    summary.to_csv("results/model_cv_summary.csv")

    print("n:", len(df))
    print("gate:", gate)
    print(summary.to_string())


if __name__ == "__main__":
    main()
