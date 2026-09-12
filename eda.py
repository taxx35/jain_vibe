import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    counts = df["Status"].value_counts().rename_axis("Status").reset_index(name="n")
    counts.to_csv("results/status_counts.csv", index=False)

    missing = df.isna().mean().sort_values(ascending=False)
    missing.rename("missing_fraction").to_csv("results/missingness.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=counts, x="Status", y="n", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Antibodies")
    fig.tight_layout()
    fig.savefig("figures/status_counts.png", dpi=300)
    plt.close(fig)

    plot_df = df.dropna(subset=["VIBE1"]).copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.stripplot(data=plot_df, x="Status", y="VIBE1", jitter=0.25, alpha=0.7, ax=ax)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig("figures/vibe_by_status.png", dpi=300)
    plt.close(fig)

    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
