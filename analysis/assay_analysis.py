import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--gate", type=float, required=True)
    parser.add_argument("--tail", type=float, default=0.10)
    parser.add_argument("--config", default="data/assays.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cfg = pd.read_csv(args.config)

    if "VIBE1" not in df.columns:
        raise ValueError("VIBE1 column not found")

    rows = []
    assay_flag = pd.Series(False, index=df.index)

    for _, r in cfg.iterrows():
        assay_name = r["assay"]
        if assay_name not in df.columns:
            continue

        x = pd.to_numeric(df[assay_name], errors="coerce")
        pair = pd.DataFrame({"VIBE1": pd.to_numeric(df["VIBE1"], errors="coerce"), "assay": x}).dropna()
        rho, p = spearmanr(pair["VIBE1"], pair["assay"])

        rows.append({
            "assay": assay_name,
            "n": len(pair),
            "rho": rho,
            "p": p
        })

        if str(r["direction"]).lower() == "high":
            cutoff = x.quantile(1 - args.tail)
            assay_flag |= (x >= cutoff).fillna(False)
        else:
            cutoff = x.quantile(args.tail)
            assay_flag |= (x <= cutoff).fillna(False)

    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    out.to_csv("results/assay_correlations.csv", index=False)

    vibe_flag = pd.to_numeric(df["VIBE1"], errors="coerce") >= args.gate
    group = np.select(
        [vibe_flag & assay_flag, vibe_flag & ~assay_flag, ~vibe_flag & assay_flag],
        ["both", "vibe_only", "assay_only"],
        default="neither"
    )
    overlap = pd.Series(group).value_counts().rename_axis("group").reset_index(name="n")
    overlap.to_csv("results/assay_overlap.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=out, x="assay", y="rho", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig("figures/assay_correlations.png", dpi=300)
    plt.close(fig)

    print(out.to_string(index=False))
    print()
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
