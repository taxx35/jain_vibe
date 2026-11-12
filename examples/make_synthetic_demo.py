from pathlib import Path

import numpy as np
import pandas as pd


AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
SEED = 17


def random_sequence(rng, length):
    return "".join(rng.choice(AMINO_ACIDS, size=length))


def mutate_positions(rng, sequence, positions):
    seq = list(sequence)
    for pos in positions:
        choices = AMINO_ACIDS[AMINO_ACIDS != seq[pos]]
        seq[pos] = rng.choice(choices)
    return "".join(seq)


def sequence_features(vh, vl):
    seq = vh + vl
    n = len(seq)

    residue_sets = [
        set("AVILMFWY"),  # hydrophobic
        set("KRH"),       # positive
        set("DE"),        # negative
        set("FWY"),       # aromatic
        set("GP"),        # gly/pro
        set("STNQ"),      # polar uncharged
    ]

    values = [sum(aa in group for aa in seq) / n for group in residue_sets]
    values.extend(seq.count(aa) / n for aa in ["A", "V", "I", "L", "F", "K"])
    return np.asarray(values, dtype=float)


def make_dataset(seed=SEED, n_families=12, members_per_subfamily=3):
    rng = np.random.default_rng(seed)
    rows = []

    for family_index in range(n_families):
        base_vh = random_sequence(rng, 120)
        base_vl = random_sequence(rng, 110)

        # Subfamily B differs from A at exactly six framework positions per chain.
        # Member-specific changes occur elsewhere. This creates a deliberate
        # margin: cross-subfamily identity is ~91-94%, so 90% generally links
        # the subfamilies while 95% keeps them separate.
        prototypes = {
            "A": (base_vh, base_vl),
            "B": (
                mutate_positions(rng, base_vh, range(6)),
                mutate_positions(rng, base_vl, range(6)),
            ),
        }

        family_effect = rng.normal(0, 0.28)

        for subfamily, (prototype_vh, prototype_vl) in prototypes.items():
            for member in range(members_per_subfamily):
                member_positions = [10 + 4 * member, 11 + 4 * member]
                vh = mutate_positions(rng, prototype_vh, member_positions)
                vl = mutate_positions(rng, prototype_vl, member_positions)

                x = sequence_features(vh, vl)

                # Synthetic positive continuous target with both composition
                # signal and planted family structure. It is not an experimental
                # VIBE measurement.
                log_target = (
                    -2.7
                    + 4.2 * x[0]
                    - 2.0 * (x[1] + x[2])
                    + 1.3 * x[3]
                    + family_effect
                    + rng.normal(0, 0.08)
                )

                row = {
                    "antibody_id": f"F{family_index:02d}_{subfamily}{member + 1}",
                    "planted_family": f"F{family_index:02d}",
                    "planted_subfamily": f"F{family_index:02d}_{subfamily}",
                    "VH_aligned": vh,
                    "VL_aligned": vl,
                    "VIBE1": float(np.exp(log_target)),
                }
                row.update(
                    {f"feat_{j:02d}": float(value) for j, value in enumerate(x, start=1)}
                )
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    out = Path(__file__).with_name("synthetic_antibodies.csv")
    df = make_dataset()
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} synthetic rows to {out}")


if __name__ == "__main__":
    main()
