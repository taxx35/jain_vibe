import argparse
from pathlib import Path

import pandas as pd


KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}

SUBS = {
    "A": ["S", "T"],
    "V": ["S", "T"],
    "I": ["S", "T"],
    "L": ["S", "T"],
    "M": ["S", "T"],
    "F": ["S", "N"],
    "Y": ["S", "T"],
    "W": ["S", "N"],
    "K": ["Q", "E"],
    "R": ["Q", "E"],
    "H": ["Q", "N"]
}


def gravy(seq):
    vals = [KD[a] for a in seq if a in KD]
    return sum(vals) / len(vals)


def nglyc(seq):
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in {"S", "T"}:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="results/hcdr3_candidates.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "Name" not in df.columns or "HCDR3" not in df.columns:
        raise ValueError("Need Name and HCDR3 columns")

    rows = []

    for _, r in df.iterrows():
        name = str(r["Name"])
        seq = str(r["HCDR3"]).strip().upper()
        wt_gravy = gravy(seq)

        for i, aa in enumerate(seq):
            if aa not in SUBS:
                continue

            for new_aa in SUBS[aa]:
                mut = seq[:i] + new_aa + seq[i + 1:]

                if nglyc(mut) and not nglyc(seq):
                    continue

                rows.append({
                    "Name": name,
                    "position": i + 1,
                    "wt": aa,
                    "mut": new_aa,
                    "HCDR3_parent": seq,
                    "HCDR3_mutant": mut,
                    "parent_gravy": wt_gravy,
                    "mutant_gravy": gravy(mut),
                    "delta_gravy": gravy(mut) - wt_gravy
                })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("parents:", df["Name"].nunique())
    print("variants:", len(out))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
