import importlib.util
from pathlib import Path

import pandas as pd


MODULE = Path(__file__).parents[1] / "analysis" / "paired_sequence_groups.py"
spec = importlib.util.spec_from_file_location("paired_sequence_groups", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_aligned_identity_counts_residue_gap_as_mismatch():
    assert mod.aligned_identity("AAAA", "AA-A") == 0.75


def test_transitive_connected_component():
    # At threshold 0.8: A-B link, B-C link, A-C do not directly link.
    seqs = ["AAAAAAAAAA", "AAAAAAAABB", "AAAAAABBBB"]
    df = pd.DataFrame({"VH_aligned": seqs, "VL_aligned": seqs})
    groups = mod.paired_chain_groups(df, threshold=0.8)
    assert groups.nunique() == 1


def test_group_count_is_monotonic_with_threshold():
    df = pd.DataFrame(
        {
            "VH_aligned": ["AAAAAAAAAA", "AAAAAAAABB", "CCCCCCCCCC"],
            "VL_aligned": ["AAAAAAAAAA", "AAAAAAAABB", "CCCCCCCCCC"],
        }
    )
    counts = [
        mod.paired_chain_groups(df, threshold=t).nunique()
        for t in (0.8, 0.9, 1.0)
    ]
    assert counts[0] <= counts[1] <= counts[2]
