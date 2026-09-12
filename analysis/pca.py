import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from descriptor_stats import DESCRIPTORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cols = [c for c in DESCRIPTORS if c in df.columns]

    x = df[cols].apply(pd.to_numeric, errors="coerce")
    keep = x.notna().all(axis=1)

    x = x.loc[keep]
    meta = df.loc[keep, ["Name", "Status"]].copy()

    z = StandardScaler().fit_transform(x)
    fit = PCA(n_components=2)
    pcs = fit.fit_transform(z)

    meta["PC1"] = pcs[:, 0]
    meta["PC2"] = pcs[:, 1]

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    meta.to_csv("results/pca_scores.csv", index=False)

    loadings = pd.DataFrame(
        fit.components_.T,
        index=cols,
        columns=["PC1", "PC2"]
    )
    loadings.to_csv("results/pca_loadings.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=meta, x="PC1", y="PC2", hue="Status", ax=ax, s=35)
    ax.set_xlabel(f"PC1 ({fit.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({fit.explained_variance_ratio_[1]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig("figures/pca.png", dpi=300)
    plt.close(fig)

    print("n used:", len(meta))


if __name__ == "__main__":
    main()
