## Analogy pipeline

Tests whether learned embeddings support analogical reasoning (e.g., morphological inflection). Two variants:

1. **analogy_v3** (`run_analogy_experiment`): Standard analogy on full representations
2. **analogy_pseudocausal** (`run_analogy_pseudocausal_experiment`): Uses prediction equivalences — given a cohort prefix and next phoneme, evaluates continuation prediction. Average precision scoring. Core logic in `src/analysis/analogy_pseudocausal.py`.

Inputs prepared by `notebooks/analogy/prepare_inputs.ipynb` → inflection instances, cross-speaker instances, false friends, allomorphs.