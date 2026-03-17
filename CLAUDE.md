# Ideal Word Representations

## Scientific question

How does the human brain represent speech during perception — at the word level, phoneme level, or something in between? This project builds computational models that learn representations at different linguistic granularities (word, phoneme, syllable, etc.) from self-supervised speech features, then uses **brain encoding** to test which model best predicts neural activity in different cortical regions. The intrinsic analyses of model representations serve the encoding goal: they validate that models actually learn the intended structure before being used as brain encoding features.

## Key concepts

We train a single kind of contrastive model under different contrastive objectives, defined by **equivalence classes** of speech frames. Each equivalence classer groups frames into classes based on some linguistic property (e.g., same orthographic word, same phoneme, same syllable). The model learns to produce similar embeddings for frames in the same class and dissimilar embeddings for different classes. By comparing which model's embeddings best predict neural activity, we can infer what kind of linguistic structure the brain is encoding.

### Equivalence classes
Frames of speech are grouped by an equivalence classer (defined in `src/datasets/speech_equivalence.py:equivalence_classers`). A model is trained on frozen wav2vec2 features to produce similar embeddings for frames in the same class and dissimilar embeddings for different classes.

Key classers:
- `word_broad`: same orthographic word form (the main "word" model)
- `phoneme`: same current phoneme
- `random`: untrained control

### Pseudocausal extraction
`w2v2_pc_8` vs `w2v2_8`: wav2vec2 is bidirectional, but **pseudocausal** extraction masks future audio input per-frame so the representation at frame _i_ only reflects audio up through frame _i_. This simulates a causal/online listener. Controlled by `pseudo_causal: true` in base_model config. See `scripts/extract_hidden_states.py:extract_representations_pseudo_causal`.

### Fixed-length context
Equivalence configs with `_fixedlen25` use `start_reference: ["fixed", 25]` — the model sees a fixed 25-frame lookback window rather than a lookback up to the current word's onset. This means that any word-level representation in the model have to be induced from the data alone -- we aren't giving it segmentation for free.

### Model naming conventions
Model configs encode architecture + hyperparameters:
- `discrim-rnn_32-pc-mAP1`: discriminative (classification loss), RNN, 32-dim output, pseudocausal, hparam variant 1
- `ffff_32-pc-mAP1`: feedforward (no RNN), 32-dim output, pseudocausal, contrastive loss, hparam variant 1
- `random*`: untrained weights (control)
- `-phon_`: phoneme-targeted hparam search variant
- Loss forms: `classification` (cross-entropy), `ratio` (contrastive), `hinge`

## Pipeline (Snakemake)

```
preprocess (TIMIT / LibriSpeech)
  → extract_hidden_states (wav2vec2, optionally pseudocausal)
  → prepare_equivalence_dataset (frame→class mappings)
  → run (train probe: train_decoder.py → src/train.py → src/models/integrator.py)
  → extract_embeddings (scripts/extract_model_embeddings.py)
  → parallel:
      ├─ run_notebook (intrinsic analyses via papermill)
      ├─ analogy experiments
      ├─ word_recognition
      ├─ estimate_encoder (brain encoding)
      └─ estimate_synthetic_encoder
```

### Output directory structure
```
outputs/
  preprocessed_data/{dataset}/
  hidden_states/{base_model}/{dataset}.h5
  equivalence_datasets/{dataset}/{base_model}/{equivalence}/equivalence.pkl
  models/{dataset}/{base_model}/{model}/{equivalence}/   # trained probes
  model_embeddings/{dataset}/{base_model}/{model}/{equivalence}/{target_dataset}.npy
  notebooks/{dataset}/{base_model}/{model}/{equivalence}/{notebook}/
  encoders/{dataset}/{feature_sets}/{subject}/           # brain encoding
  analogy/inputs/  and  analogy/runs/
```

## Brain encoding (primary analysis)

**Goal**: Predict ECoG neural activity from model embeddings using TRF (temporal receptive field) models, then compare which linguistic-level model best explains activity in each brain region.

### Data
- ECoG grid recordings from epilepsy patients listening to TIMIT sentences
- Subjects configured in `config.yaml:encoding:data`
- Data loaded via the `fomo` (neural-foundation-models) library at `FOMO_LIBRARY_PATH` (standardized ECoG data format). Local `.mat` loading is a legacy path.

### Feature sets (`conf_encoder/feature_sets/`)
Each encoding model combines:
- **Baseline features**: onset, phnfeatConsOnset, maxDtL (auditory envelope), formantMedOnset, F0
- **Model features** (optional): model embeddings aggregated per-phoneme (`state_space: phoneme`, `featurization: mean`), optionally L2-normalized

Naming convention: `ph-ls-pc-word_broad_fixed-w2v2_pc_8-ffff-l2norm` encodes: `ph`=one embedding per phoneme, `ls`=trained on LibriSpeech, `pc`=pseudocausal extraction, `word_broad_fixed`=equivalence class, `w2v2_pc_8`=base model, `ffff`=feedforward architecture, `l2norm`=L2-normalized embeddings.

### Method
- Ridge regression with nested k-fold CV (`src/encoding/ecog/timit.py:strf_nested_cv`)
- Per-electrode correlation between predicted and actual neural signal
- Model comparison: model2 vs model1 (e.g., word model vs baseline-only, or word vs phoneme)
- Permutation tests (`permute_units`) for significance

### Pipeline stages
1. `estimate_encoder` → per-subject per-feature-set TRF model
2. `estimate_encoder_unit_permutation` → permutation baselines
3. `compare_encoder_within_subject` → pairwise model comparison per subject
4. `compare_all_encoders_across_subject` → aggregate across subjects (t-tests)
5. `electrode_contrast` → which electrodes/ROIs prefer which model
6. `electrode_study_within_subject` → detailed per-subject electrode analysis

## Intrinsic analyses

Run via papermill notebooks in `notebooks/`. Each gets: model_dir, embeddings, hidden_states, equivalence datasets, state_space_specs. Full list in `ALL_MODEL_NOTEBOOKS` at top of Snakefile.

Key analyses:
- **lexical_coherence / phoneme_coherence**: within-class vs between-class embedding distance
- **word_discrimination**: classification accuracy from embeddings
- **rsa_phoneme**: representational similarity vs phoneme features
- **temporal_generalization_{word,phoneme}**: cross-timepoint generalization
- **state_space**: embedding trajectories through a word
- **word_boundary / syllable_boundary**: boundary detection from representation changes
- **trf**: temporal receptive field analysis of model internals

## Configuration

- **Hydra** for model training (`conf/`): base_model, dataset, equivalence, model
- **Hydra** for encoding (`conf_encoder/`): feature_sets, cv, model (TRF params)
- **Snakemake config** (`config.yaml`): dataset paths, model list, encoding subjects, model comparisons

## StateSpaceAnalysisSpec data model

Stored in `outputs/state_space_specs/{dataset}/{base_model}/state_space_specs.h5`. Load with `StateSpaceAnalysisSpec.from_hdf5`. Each top-level HDF5 group (e.g. `word`, `phoneme`, `syllable`, `biphone`) defines a state space:

- **`labels`**: array of class labels (e.g. word types, phoneme symbols)
- **`target_frame_spans`**: Nx2 array of `[onset_frame, offset_frame)` for each instance
- **`target_frame_span_boundaries`**: cumulative boundaries into `target_frame_spans` per label (label _i_'s spans are `target_frame_spans[boundaries[i-1]:boundaries[i]]`)
- **`cuts/`** (optional, for word/syllable/biphone): a pandas DataFrame stored via HDFStore with MultiIndex `(label, instance_idx, level)` and columns `(description, onset_frame_idx, offset_frame_idx, item_idx)`. Describes sub-unit decompositions — e.g., for the word state space, `cuts_df` has rows at `level=phoneme` and `level=syllable` giving the phoneme/syllable segmentation of each word instance.
