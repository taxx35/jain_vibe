import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


DESCRIPTORS = [
    "SAP_pos_CDRH1", "SAP_pos_CDRH2", "SAP_pos_CDRH3",
    "SAP_pos_CDRL1", "SAP_pos_CDRL2", "SAP_pos_CDRL3",
    "SAP_pos_CDR", "SAP_pos_Hv", "SAP_pos_Lv", "SAP_pos_Fv",
    "SCM_neg_CDRH1", "SCM_neg_CDRH2", "SCM_neg_CDRH3",
    "SCM_neg_CDRL1", "SCM_neg_CDRL2", "SCM_neg_CDRL3",
    "SCM_neg_CDR", "SCM_neg_Hv", "SCM_neg_Lv", "SCM_neg_Fv",
    "SCM_pos_CDRH1", "SCM_pos_CDRH2", "SCM_pos_CDRH3",
    "SCM_pos_CDRL1", "SCM_pos_CDRL2", "SCM_pos_CDRL3",
    "SCM_pos_CDR", "SCM_pos_Hv", "SCM_pos_Lv", "SCM_pos_Fv",
    "GRAVY_VH", "GRAVY_VL",
    "GRAVY_HCDR1", "GRAVY_HCDR2", "GRAVY_HCDR3",
    "GRAVY_LCDR1", "GRAVY_LCDR2", "GRAVY_LCDR3",
]


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty(n)
    out[order] = q
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--gate", type=float)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    available = [c for c in DESCRIPTORS if c in df.columns]
    if len(available) != len(DESCRIPTORS):
        print(f"using {len(available)} of {len(DESCRIPTORS)} descriptor columns")

    corr_rows = []
    for col in available:
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df["VIBE1"], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
        rho, p = spearmanr(pair["x"], pair["y"])
        corr_rows.append([col, len(pair), rho, p])

    corr = pd.DataFrame(corr_rows, columns=["feature", "n", "rho", "p"])
    corr["q"] = bh(corr["p"])
    corr = corr.sort_values(["q", "p"])

    outcome = df[df["Status"].isin(["Approved", "Terminated"])].copy()
    outcome["VIBE1"] = pd.to_numeric(outcome["VIBE1"], errors="coerce")
    outcome = outcome.dropna(subset=["VIBE1"])

    gate = args.gate
    if gate is None:
        gate = outcome["VIBE1"].std(ddof=1)

    outcome["high_vibe"] = outcome["VIBE1"] >= gate

    rows = []
    for col in available:
        hi = pd.to_numeric(outcome.loc[outcome["high_vibe"], col], errors="coerce").dropna()
        lo = pd.to_numeric(outcome.loc[~outcome["high_vibe"], col], errors="coerce").dropna()

        stat = mannwhitneyu(hi, lo, alternative="two-sided")
        rb = 2 * stat.statistic / (len(hi) * len(lo)) - 1

        rows.append([
            col,
            len(hi),
            len(lo),
            hi.median(),
            lo.median(),
            stat.statistic,
            stat.pvalue,
            rb
        ])

    groups = pd.DataFrame(
        rows,
        columns=["feature", "n_high", "n_low", "median_high", "median_low", "u", "p", "rank_biserial"]
    )
    groups["q"] = bh(groups["p"])
    groups = groups.sort_values(["q", "p"])

    Path("results").mkdir(exist_ok=True)
    corr.to_csv("results/descriptor_correlations.csv", index=False)
    groups.to_csv("results/descriptor_group_tests.csv", index=False)

    print("gate:", gate)
    print(corr.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
