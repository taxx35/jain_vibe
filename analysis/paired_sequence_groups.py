import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aligned_identity(seq_a: str, seq_b: str) -> float:
    """Identity for two pre-aligned equal-length amino-acid strings.

    Positions where both sequences contain a gap are excluded from the
    denominator. A residue-gap position counts as a mismatch.
    """
    a = str(seq_a).strip().upper()
    b = str(seq_b).strip().upper()
    if len(a) != len(b):
        raise ValueError("aligned sequences must have equal length")

    compared = 0
    matches = 0
    for aa, bb in zip(a, b):
        if aa == "-" and bb == "-":
            continue
        compared += 1
        if aa == bb:
            matches += 1

    if compared == 0:
        raise ValueError("alignment contains no comparable positions")
    return matches / compared


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def paired_chain_groups(
    df: pd.DataFrame,
    threshold: float,
    vh_col: str = "VH_aligned",
    vl_col: str = "VL_aligned",
) -> pd.Series:
    """Build deterministic connected-component groups from paired VH/VL identity.

    Two antibodies are linked when BOTH aligned-chain identities are greater
    than or equal to threshold. Transitive closure is then applied through
    connected components.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    for col in (vh_col, vl_col):
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")

    n = len(df)
    uf = UnionFind(n)
    vh = df[vh_col].astype(str).tolist()
    vl = df[vl_col].astype(str).tolist()

    for i in range(n):
        for j in range(i + 1, n):
            vh_id = aligned_identity(vh[i], vh[j])
            if vh_id < threshold:
                continue
            vl_id = aligned_identity(vl[i], vl[j])
            if vl_id >= threshold:
                uf.union(i, j)

    components = {}
    for i in range(n):
        root = uf.find(i)
        components.setdefault(root, []).append(i)

    ordered = sorted(components.values(), key=lambda members: min(members))
    group_for_index = {}
    for group_number, members in enumerate(ordered, start=1):
        gid = f"G{group_number:03d}"
        for idx in members:
            group_for_index[idx] = gid

    return pd.Series(
        [group_for_index[i] for i in range(n)],
        index=df.index,
        name="sequence_group",
    )


def group_diagnostics(groups: pd.Series) -> dict:
    counts = groups.value_counts()
    return {
        "n_groups": int(len(counts)),
        "largest_group_n": int(counts.max()),
        "largest_group_fraction": float(counts.max() / len(groups)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--vh-col", default="VH_aligned")
    parser.add_argument("--vl-col", default="VL_aligned")
    parser.add_argument("--out", default="results/synthetic_grouped.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["sequence_group"] = paired_chain_groups(
        df,
        threshold=args.threshold,
        vh_col=args.vh_col,
        vl_col=args.vl_col,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    diag = group_diagnostics(df["sequence_group"])
    print(f"threshold: {args.threshold:.2f}")
    for key, value in diag.items():
        print(f"{key}: {value}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
