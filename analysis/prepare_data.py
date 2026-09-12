import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="results/analysis_table.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    needed = ["Name", "Status", "VIBE1"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Name"] = df["Name"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["VIBE1"] = pd.to_numeric(df["VIBE1"], errors="coerce")

    if df["Name"].duplicated().any():
        dupes = df.loc[df["Name"].duplicated(keep=False), "Name"].unique()
        raise ValueError(f"Duplicate antibody names found: {dupes[:10]}")

    df["outcome_set"] = df["Status"].isin(["Approved", "Terminated"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("rows:", len(df))
    print("unique antibodies:", df["Name"].nunique())
    print(df["Status"].value_counts(dropna=False))
    print("missing VIBE:", df["VIBE1"].isna().sum())
    print("saved:", out)


if __name__ == "__main__":
    main()
