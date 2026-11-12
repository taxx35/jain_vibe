# Example AI-evaluation task derived from the VIBE research

## Main problem

Implement a leakage-aware evaluation of sequence-based regression on a synthetic continuous antibody readout.

The solver receives:

- pre-aligned paired VH/VL amino-acid sequences;
- sequence-derived numeric features;
- a positive continuous synthetic target with planted family structure.

The goal is to construct paired-chain similarity groups and evaluate a continuous model without allowing near-identical sequence groups to cross validation folds.

## Biological grouping rule for this benchmark

For each chain separately, sequence identity is:

`matching aligned residues / aligned positions compared`

Positions containing a gap in both sequences are excluded from the denominator; residue-gap positions count as mismatches.

Two antibodies are directly linked only when **both** VH and VL identities are at least the stated threshold. Final groups are connected components of this graph, so grouping is transitive.

The main threshold is 90%. Repeat grouping at 85% and 95% as a sensitivity subproblem.

This benchmark rule is deliberately explicit and deterministic. It is inspired by the homology-control problem in the research but is not presented as the exact private implementation used in the collaboration analysis.

## Modelling subproblem

Using the 90% groups:

1. log-transform the positive target;
2. use five-fold outer `GroupKFold`;
3. use grouped inner cross-validation for model selection;
4. fit `StandardScaler -> PCA -> Ridge` only within training data;
5. tune PCA dimension over `[2, 4, 8, 16, 32]` when valid for the training fold;
6. tune Ridge alpha over `[0.1, 1, 10, 100, 300]`;
7. produce exactly one outer-fold prediction per sample.

## Required output

Return:

- pooled out-of-fold Spearman correlation;
- pooled MAE, RMSE and R²;
- per-fold sample and group counts;
- largest-group fraction per fold;
- selected PCA dimension and Ridge alpha per fold;
- one out-of-fold prediction for every sample.

## Connected tests

A strong test suite should include:

1. **Transitivity:** a chained A-B-C example where A links B and B links C, while A does not directly link C, must still form one component.
2. **Nested-group trap:** a constructed dataset where ungrouped inner tuning selects a different hyperparameter from correctly grouped tuning.
3. **Preprocessing isolation:** after a fold is fixed, perturb other held-out samples while leaving one held-out sample unchanged; that sample's prediction must not change.
4. **Threshold monotonicity:** the number of groups cannot decrease when the identity threshold is raised from 85% to 90% to 95%.
5. **Metric correctness:** pooled metrics must agree with independently calculated reference values.

## Why this is difficult for an AI model

Likely errors include:

- clustering VH and VL independently instead of applying the paired-chain rule;
- missing transitive closure;
- using grouped outer folds but ordinary inner CV;
- fitting scaling or PCA before the outer split;
- allowing a PCA dimension larger than the outer-training fold supports;
- averaging per-fold Spearman values when pooled out-of-fold Spearman was requested.

The task is based on a validation problem I implemented in antibody developability modelling: preventing close biological relatives from making model performance look more transferable than it is.
