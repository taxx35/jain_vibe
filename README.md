# VIBE analysis

Code from my MSc Bioinformatics project on antibody VIBE measurements and the later manuscript analysis.

I kept the repository fairly simple. The main steps are split into separate scripts so I can rerun parts of the analysis without going through one large notebook.

## Files

- `prepare_data.py` - basic cleaning and cohort checks
- `eda.py` - cohort summaries and VIBE plots
- `assay_analysis.py` - VIBE vs other developability assays
- `descriptor_stats.py` - sequence descriptor correlations and High/Low VIBE comparisons
- `pca.py` - PCA on the descriptor set
- `model_cv.py` - five-feature Elastic-Net model and GRAVY-VH benchmark
- `holdout.py` - train/test cohort analysis when a cohort column is available
- `hcdr3_screen.py` - simple single-substitution HCDR3 screening

The original project used antibody data and assay measurements supplied through the research collaboration, so those files are not in the public repository.

## Input

The main working table is a CSV with one row per antibody. The scripts expect columns such as:

`Name`, `Status`, `VIBE1`

and, for the sequence analysis, the descriptor columns used in the thesis/manuscript.

For the HCDR3 screen I use an already annotated `HCDR3` column rather than trying to infer CDR3 boundaries from the raw VH sequence.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows Git Bash:

```bash
source .venv/Scripts/activate
```

## Example

```bash
python analysis/prepare_data.py data/final_all_descriptors.csv
python analysis/eda.py results/analysis_table.csv
python analysis/descriptor_stats.py results/analysis_table.csv
python analysis/pca.py results/analysis_table.csv
python analysis/model_cv.py results/analysis_table.csv
```

The follow-on manuscript work is still in progress, so this repository is mainly a record of the computational analysis 
