# VIBE antibody modelling

Public, data-free code from my MSc Bioinformatics research and follow-on manuscript analysis of antibody VIBE measurements.

The collaboration dataset is not redistributed here. This repository is intended to show the computational workflow, validation logic and reproducibility practices without exposing restricted assay measurements, antibody tables, private embeddings, unpublished manuscript figures or prospective mutation rankings.

## Research question

The project asks how much information about an antibody interfacial-behaviour readout can be recovered from sequence-derived representations, and how robust that signal remains when closely related antibody sequences are kept out of the same train/test split.

The later analysis moved beyond the original five-feature classifier toward continuous modelling of `log(VIBE1)` with stronger controls for biological similarity and model-selection leakage.

## Current modelling framework

The manuscript-stage workflow uses:

- sequence-derived physicochemical descriptors;
- pretrained antibody/protein representations, including IgBert in the strongest matched comparison;
- `StandardScaler -> PCA -> Ridge`;
- nested, sequence-group-aware cross-validation;
- training-fold-only preprocessing and hyperparameter selection;
- 90% paired VH/VL sequence-similarity grouping as the primary leakage-control setting, with 85% and 95% sensitivity analyses;
- pooled out-of-fold Spearman correlation as the primary ranking metric, with MAE, RMSE and R² as complementary metrics;
- repeated grouped evaluations, permutation controls and cross-batch checks in the private research analysis.

The public code demonstrates these methodological ideas on synthetic data. It does not claim to reproduce restricted collaboration results.

## Public benchmark-style example

The repository now includes a synthetic paired-chain benchmark derived from the leakage-control problem in the research:

- `analysis/paired_sequence_groups.py` — deterministic grouping of aligned VH/VL pairs by a fully specified paired-chain identity rule;
- `analysis/grouped_regression.py` — nested `GroupKFold` regression with fold-contained scaling, PCA and Ridge tuning;
- `examples/make_synthetic_demo.py` — generates synthetic aligned antibody-like VH/VL sequences, sequence-derived features and a continuous target with planted family structure;
- `examples/synthetic_antibodies.csv` — generated synthetic data only;
- `tests/test_paired_sequence_groups.py` — identity, transitivity and threshold-sensitivity tests;
- `tests/test_grouped_regression.py` — leakage, out-of-fold coverage and preprocessing-isolation tests;
- `TASK_EXAMPLE.md` — an example AI-evaluation task derived from the research problem;
- `.github/workflows/tests.yml` — continuous integration.

### Synthetic grouping rule

For the public benchmark only, VH and VL sequences are pre-aligned and equal-length within each chain. Pairwise identity is the fraction of aligned positions with the same residue, excluding positions where both sequences contain a gap. Two antibodies are linked only when **both** VH and VL identities meet the chosen threshold. Connected components define the final group IDs.

This explicit rule makes the public benchmark deterministic. It should not be read as a claim that the private research pipeline used this exact implementation.

## Earlier MSc-stage scripts

The repository also retains earlier exploratory and classification analyses as a transparent record of the project:

- `analysis/prepare_data.py` — cleaning and cohort checks;
- `analysis/eda.py` — cohort summaries and VIBE plots;
- `analysis/assay_analysis.py` — VIBE versus other developability assays;
- `analysis/descriptor_stats.py` — descriptor correlations and group comparisons;
- `analysis/pca.py` — descriptor PCA;
- `analysis/model_cv.py` — earlier five-feature elastic-net classifier and GRAVY-VH benchmark;
- `analysis/holdout.py` — earlier cohort holdout analysis;
- `analysis/hcdr3_screen.py` — simple physicochemical HCDR3 substitution screen.

These earlier scripts are retained for provenance and should not be interpreted as the final manuscript modelling pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the synthetic example:

```bash
python examples/make_synthetic_demo.py
```

Build 90% paired-chain groups:

```bash
python analysis/paired_sequence_groups.py examples/synthetic_antibodies.csv \
  --threshold 0.90 \
  --out results/synthetic_grouped.csv
```

Run nested grouped regression:

```bash
python analysis/grouped_regression.py results/synthetic_grouped.csv \
  --target VIBE1 \
  --group-col sequence_group \
  --feature-prefix feat_ \
  --log-target
```

Run tests:

```bash
pytest -q
```

## Interpretation

The grouped evaluation asks a narrower question than ordinary random cross-validation: how well does the model rank antibodies when near-identical sequence groups are not shared between training and test folds?

It should not be interpreted as independent external validation, assay replacement, causal evidence or experimental validation of proposed mutations.

## Data and confidentiality

See [`DATA.md`](DATA.md). All bundled example data are synthetic.

## Research status

The follow-on manuscript is in preparation. Public code is intentionally limited to methods that can be shared without disclosing collaboration data or unpublished prospective-design material.
