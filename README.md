# VIBE antibody analysis

This repository contains a cleaned version of the computational workflow developed during my MSc Bioinformatics dissertation, **Enhancing Antibody Developability through VIBE-Informed HCDR3 Engineering**.

The original project grew through exploratory notebooks. This version separates the work into small scripts so that data preparation, exploratory analysis, statistics, modelling, validation, figures and HCDR3 candidate generation can be inspected independently.

## Data-sharing boundary

The collaborator-supplied VIBE measurements, antibody sequences, unpublished assay data, unpublished manuscript tables, experimental plans and third-party model files are **not included**. The scripts expect local input files. This makes the workflow inspectable without redistributing material that may be subject to collaboration, publication or licensing restrictions.

## Workflow

1. **Input validation and harmonisation** — identifiers, outcome labels, missingness, duplicates and descriptor coverage.
2. **Cohort EDA** — status counts, VIBE distributions and missingness summaries.
3. **VIBE vs standard assays** — pairwise Spearman correlations, per-assay N, worst-tail outliers and VIBE-only / assay-only / both / neither overlap.
4. **Outcome-oriented threshold analysis** — Approved + Terminated subset, documented SD-based gate, precision/recall, continuous ROC-AUC and average precision, bootstrap intervals and percentile sensitivity.
5. **Sequence-descriptor analysis** — continuous Spearman associations, Benjamini-Hochberg FDR, High/Low VIBE Mann-Whitney tests and rank-biserial effect sizes.
6. **Descriptor-space PCA** — standardised 38-feature PCA for exploratory structure.
7. **Sequence surrogate model** — five-feature Elastic-Net logistic regression with fold-contained scaling, stratified cross-validation and GRAVY-VH benchmark.
8. **Cohort-held-out validation** — fit on one cohort, derive the gate from training only, remove identifier overlap, test unchanged on another cohort and bootstrap uncertainty.
9. **HCDR3 variant generation / prioritisation** — uses an already annotated HCDR3 sequence, creates controlled single substitutions, rejects new cysteines and newly introduced N-X-S/T motifs, and records hydropathy changes. This is computational prioritisation, not experimental optimisation.
10. **Figures** — scripts generate cohort, assay, descriptor, PCA and model figures from local data.

## Structure

```text
analysis/
  00_validate_inputs.py
  01_prepare_analysis_table.py
  02_cohort_eda.py
  03_assay_complementarity.py
  04_outcome_threshold.py
  05_descriptor_analysis.py
  06_descriptor_pca.py
  07_surrogate_model_cv.py
  08_cohort_holdout_validation.py
  09_hcdr3_variant_prioritisation.py
src/vibe_analysis/
  constants.py
  io.py
  stats.py
  plotting.py
  sequence.py
data/
  README.md
  assay_config_template.csv
results/
figures/
tests/
run_workflow.py
```

## Expected local files

`final_all_descriptors.csv`: one row per antibody with `Name`, `Status`, `VIBE1`, optional sequence columns (`VH`, `VL`, `HCDR3`), an optional cohort/source column such as `APH`, and the 38 descriptor columns listed in `src/vibe_analysis/constants.py`.

`antibody_jain_vibe_score.csv`: `Name`, `VIBE1`, and conventional assay columns.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# source .venv/Scripts/activate  # Git Bash on Windows
pip install -r requirements.txt
```

## Run

```bash
python run_workflow.py --descriptor-table data/final_all_descriptors.csv
```

If you also have a local assay table and have deliberately chosen the VIBE gate used for overlap analysis:

```bash
python run_workflow.py \
  --descriptor-table data/final_all_descriptors.csv \
  --assay-table data/antibody_jain_vibe_score.csv \
  --assay-config data/assay_config_template.csv \
  --vibe-gate 0.0374866
```

The example gate above is only an example from the historical analysis and should not be copied into a new analysis without checking the exact intended rule and cohort.

## Scientific scope

The sequence surrogate is not a clinical-outcome model and its probabilities should not be interpreted as calibrated probabilities of therapeutic success or failure. A predicted reduction in a surrogate score is not evidence that a mutation improves measured VIBE, binding, expression, stability or any other antibody property. Experimental validation is required.
