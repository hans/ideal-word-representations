# French inflection analogy task — design notes (issue #1)

Working doc for the French analog of `notebooks/analogy/run.ipynb`. Captured before pausing #1 to land #2 (per-token morphological tagging of MLS French). When resuming #1, read this top to bottom, then start the implementation plan.

## Status

- Sketch approved by user via brainstorming Q&A on 2026-05-21.
- ~~Blocked on #2~~ — landed at commit 8825e17.
- **Implemented** on 2026-05-21 in commits 4532942, 7bfeff3, ec22478, 9464b70. End-to-end smoke test passes on `mls_french-dev`. The implementation diverged from the sketch in one place (separate join paths for true_friend vs false_friend rows); section "[all_cross_instances construction](#all_cross_instances-construction)" below reflects the as-shipped design.
- Branch state: `feat/issue-1` rebased onto `origin/feat/issue-2` (will rebase forward to `origin/wav2vec2-phoneme-models` after PR #3 merges); `outputs/` is a symlink to the wav2vec2-phoneme-models worktree's outputs (recreate on resume).

## Inputs and what we have

| Source | Path | Used for |
|---|---|---|
| Preprocessed MLS French dev (8 utts, 238 word types) | `outputs/preprocessed_data/mls_french-dev/` | label set, audio instances |
| wav2vec2-FR-1K-base hidden states (pseudocausal, layer 8) | `outputs/hidden_states/w2v2_pc_fr_8/mls_french-dev.h5` | analogy representations |
| Equivalence datasets (word_broad + phoneme, 10frames_fixedlen25) | `outputs/equivalence_datasets/mls_french-dev/w2v2_pc_fr_8/` | exist; trained-probe model not yet (not needed for w2v2-only analogy) |
| `suffix_pairs_top10.csv` (384k rows) | `data/french_morphology/suffix_pairs_top10.csv` | row-level inflection pairs |
| `suffix_directions_report.csv` (1973 rows) | `data/french_morphology/suffix_directions_report.csv` | filter by paradigm direction frequency |
| `word_morph_detail` (per-token morph) | **#2 output** — does not exist yet | disambiguate base/derived cell per audio instance |

IPA inventory verified compatible between MFA output and the CSV (`ʁ ɛ ɔ̃ ə ʃ` all match; spot-checked on *apporte → apɔʁt*, *fasse → fas*, *dit → di*, *fut → fy*).

## Coverage on this dev set

- 22 CSV rows have both `base_orth` and `derived_orth` attested in the 238-word vocab.
- 10 distinct (base_orth, derived_orth) pairs survive.
- Dominant suffixes: /s/, /ʁ/, /e/. Both true_friend and false_friend types represented.
- **This is sufficient for a smoke test, not for science.** Notebooks must be parameterized so re-running on a larger split is a config change.

## Pipeline shape

```
outputs/preprocessed_data/mls_french-dev/
  ├─ (existing) word_detail, word_phonemic_detail
  └─ (NEW, from #2) word_morph_detail  ← spaCy UD features + csv_cell per token

outputs/hidden_states/w2v2_pc_fr_8/mls_french-dev.h5  (exists)

outputs/state_space_specs/mls_french-dev/w2v2_pc_fr_8/state_space_specs.h5  ← NEW
  └─ "word" spec only (French preprocessing has no syllables)

outputs/analogy/inputs/mls_french-dev/w2v2_pc_fr_8/
  ├─ all_cross_instances.parquet  ← NEW
  └─ state_space_spec.h5          ← copy/subsample

notebooks/analogy/prepare_inputs_french.ipynb  ← NEW (papermill-driven)
notebooks/analogy/run_french.ipynb              ← NEW (papermill-driven)
```

## State-space spec adaptation

`scripts/generate_state_space_specs.py:compute_word_state_space` iterates `item["word_syllable_detail"]` to derive word spans. The MLS French preprocessing notebook deliberately omits syllabification ("no French syllabifier"). So:

- Add `compute_word_state_space_french` (or a config-driven variant) that:
  - Uses `word_detail.start/stop` for word frame spans
  - Uses `word_phonemic_detail` for phoneme cuts (already in the dataset)
  - Skips syllable-level cuts entirely (downstream analogy code only consumes `cuts.xs("phoneme", level="level")`)
- Skip `compute_syllable_state_space` and `compute_biphone_state_space` for French (both depend on syllabification)
- Output: `outputs/state_space_specs/mls_french-dev/w2v2_pc_fr_8/state_space_specs.h5` with key `"word"`

## all_cross_instances construction

Final schema (one row = one (base_audio_instance, derived_audio_instance, morph_direction) triple):

| Column | Source | Notes |
|---|---|---|
| `inflection` | computed | `f"{suffix_ipa}::{base_cell}→{derived_cell}"` |
| `suffix_ipa`, `suffix_rank` | suffix_pairs_top10 | |
| `base_cell`, `derived_cell` | suffix_directions_report ∩ suffix_pairs row | a single chosen cell each |
| `type` | suffix_pairs_top10 | `true_friend` / `false_friend` (`missing_base` dropped — we need both) |
| `base`, `inflected` | `base_orth`, `derived_orth` | match state_space_spec.labels |
| `base_idx`, `inflected_idx` | state_space_spec.labels position | |
| `base_instance_idx`, `inflected_instance_idx` | state_space_spec.target_frame_spans | |
| `base_phones`, `inflected_phones` | state_space_spec cuts | space-joined phone strings |
| `base_ipa_csv`, `derived_ipa_csv` | suffix_pairs_top10 | CSV's expected IPA (for IPA sanity checks) |
| `exclude_main` | False by default | True for false_friends so they don't pollute the main experiments |
| `n_stems`, `total_base_count`, `total_derived_count` | suffix_directions_report | row-level direction support |
| `base_lemma`, `derived_lemma` | suffix_pairs_top10 | useful for grouping |

### Construction algorithm (as shipped)

The implementation has **two distinct join paths** because `false_friend` rows are accidental phonological pairs by definition — they don't appear in `suffix_directions_report.csv`. Forcing them through the directions-report join drops every false_friend row.

**Common prefix (both paths):**
1. Drop CSV rows with `type == "missing_base"` (we need both sides attested) and rows with missing orths.
2. Restrict to corpus-attested orth pairs: drop rows where `base_orth` or `derived_orth` not in `state_space_spec.labels`.
3. Parse `base_tags` / `derived_tags` (` ; `-separated) into per-row `frozenset` tag-sets.

**True-friend path (`type == "true_friend"`):**
4a. Pre-filter directions report: `n_stems >= 50 AND total_base_count >= 100 AND total_derived_count >= 100`. On the actual data this keeps **173 / 1972** (suffix_ipa, base_cell, derived_cell) tuples.
4b. Inner-join `tf_rows ⋈ dirs_keep` on `suffix_ipa`.
4c. Tag-consistency filter: keep rows where `base_cell ∈ row.base_tag_set ∧ derived_cell ∈ row.derived_tag_set`.
4d. For each surviving (orth_pair, direction) tuple, expand to (base_instance × derived_instance) where the per-token spaCy `csv_cell` equals the chosen `base_cell` / `derived_cell`.

**False-friend path (`type == "false_friend"`):**
5a. Skip the directions-report join entirely.
5b. For each false_friend row, enumerate base / derived audio instances whose per-token spaCy `csv_cell` is in the row's `base_tag_set` / `derived_tag_set` respectively. The `base_cell` / `derived_cell` columns are filled from the per-token tag itself (not from a directions-report row).
5c. Cartesian product of surviving instances becomes rows.

**Merge:**
6. Concat the two paths, attach `inflection = f"{suffix_ipa}::{base_cell}→{derived_cell}"`, `base_idx`, `inflected_idx`, `base_phones`, `inflected_phones`, `exclude_main = (type == "false_friend")`. Save to `outputs/analogy/inputs/{dataset}/{base_model}/all_cross_instances.parquet`.

### Homophony handling

The English notebook used hand-curated homophone exclusion via CMUDict phonemic identity. For French, we get this *for free* from `word_morph_detail`:
- Two CSV rows can share `(base_orth, derived_orth)` but disagree on `(base_cell, derived_cell)`. With morph tagging, each audio instance is bound to exactly one cell, so the analogy task gets a single canonical direction per (instance, instance) pair.
- We still construct a `homophone_map` (label_idx → set of label_idx that are phonologically identical) for use by `analogy.run_experiment_equiv_level`'s `include_idxs_in_predictions` arg, so predicting a homophone counts as correct. Build this from `cut_phonemic_forms` joined on identical phone tuples, same logic as English notebook cells 18–19.

## Experiment set

The English notebook's `experiments` dict mixes (a) per-inflection accuracy, (b) regular-vs-irregular contrasts, (c) inflection-to-inflection transfer (NNS↔VBZ), (d) false friends, (e) forced-choice allomorphy. For French, the natural analogs:

```python
experiments = {
    # Basic: per-suffix accuracy, all directions pooled
    "basic_by_suffix": {
        "group_by": ["suffix_ipa"],
        "all_query": "type == 'true_friend' and not exclude_main",
    },

    # Per (suffix, direction) — this is the homophony test
    "by_direction": {
        "group_by": ["suffix_ipa", "base_cell", "derived_cell"],
        "all_query": "type == 'true_friend' and not exclude_main",
    },

    # Cross-direction transfer within same suffix: does encoding generalize across
    # morphological directions that share a phonological suffix?
    # (Filled programmatically: for each suffix with >= 2 well-attested directions,
    # add cross experiments)
    # f"cross_dir-{suffix}-{dir_a}-to-{dir_b}": {
    #     "base_query": f"suffix_ipa == '{suffix}' and base_cell == '{dir_a_base}' and derived_cell == '{dir_a_derived}'",
    #     "inflected_query": f"suffix_ipa == '{suffix}' and base_cell == '{dir_b_base}' and derived_cell == '{dir_b_derived}'",
    # }

    # False-friend control: does the analogy vector for true_friend rows transfer
    # to false_friend rows with matching surface suffix? Expect: it shouldn't.
    "false_friend_by_suffix": {
        "group_by": ["suffix_ipa"],
        "base_query": "type == 'true_friend'",
        "inflected_query": "type == 'false_friend'",
    },

    # Optional: forced-choice — for a given base form, does the model prefer the
    # phonologically-licensed allomorph? Skipped for v1; revisit once basic experiments
    # produce signal. French doesn't have phonologically-conditioned allomorphy as
    # clean as English /s,z,ɪz/, but liaison-driven pairs (e.g., final /ʁ/ vs /ʁə/)
    # are plausible candidates.
}
```

### Programmatic experiment expansion

Mirror the English notebook's `itertools.product` expansion (cells 22–24) for the cross-direction transfer experiments. Skeleton:

```python
# Enumerate well-attested (suffix, base_cell, derived_cell) tuples
well_attested = (
    all_cross_instances.query("type == 'true_friend' and not exclude_main")
    .groupby(["suffix_ipa", "base_cell", "derived_cell"])
    .size()
    .loc[lambda s: s >= MIN_INSTANCES_PER_DIRECTION]
    .reset_index()
)

# Within each suffix, generate cross-direction transfer experiments
for suffix, group in well_attested.groupby("suffix_ipa"):
    directions = list(zip(group.base_cell, group.derived_cell))
    for (b1, d1), (b2, d2) in itertools.combinations(directions, 2):
        name = f"cross_dir-{suffix}-{b1}→{d1}-to-{b2}→{d2}"
        experiments[name] = {
            "base_query": f"suffix_ipa == '{suffix}' and base_cell == '{b1}' and derived_cell == '{d1}'",
            "inflected_query": f"suffix_ipa == '{suffix}' and base_cell == '{b2}' and derived_cell == '{d2}'",
        }
```

`MIN_INSTANCES_PER_DIRECTION` is a notebook parameter; expect a small int (5–10) for the dev set, larger for a real run.

## Resolved questions

| Question | Resolution |
|---|---|
| What schema does `word_morph_detail` have? | `{upos, morph, csv_cell, lemma, alignment_confidence}` per token, confirmed on disk. |
| How are alignment failures represented? | `alignment_confidence == "missing"` → `csv_cell = null`. None observed on the dev set (all 405 tokens aligned, 378 exact + 27 fuzzy). |
| How exhaustive is the UD→CSV cell mapping? | `src/datasets/french_morph.py:ud_features_to_csv_cell` produces alphabetically-sorted pipe-joined strings; spot-checked that the format matches the CSV's. |
| How should we handle false_friend rows that don't appear in directions_report? | Decision (2026-05-21 brainstorm): separate join path — FF rows skip directions_report and require only per-token tag ∈ row's tag set. Implemented. |

## Open scientific question

**Will `false_friend_by_suffix` produce signal at train scale?** On `mls_french-dev` it produces 0 rows, and the FF spot-check in `prepare_inputs_french.ipynb` shows the rejection is **principled**: spaCy consistently resolves homophonous orth forms to the contextually-correct reading (e.g. *fasse* in "que je le **fasse** mourir" tagged as subjunctive verb), but the CSV's FF rows describe alternative readings (e.g. *fasse* as a noun) that rarely surface in actual narrated text.

If FF rows stay near-zero at train scale, the experiment is vestigial and should either be (a) dropped, (b) relaxed to per-orth-only without consulting the CSV's tag interpretation (which is a different scientific claim — comparing "any audio of orth pair" instead of "audio tokens with the FF-claimed morphology"), or (c) re-formulated as a different kind of contrast. **Do not run on train split before eyeballing the FF spot-check on a larger preprocessed dev set.**

## Implementation status

All steps completed on 2026-05-21:

| Step | Status | Commit / File |
|---|---|---|
| 1. Recreate outputs symlink on resume | — (resume step, not committed) | `ln -s /Users/jon/.superset/worktrees/e4c3a981-cd98-4323-81d0-358a6fc04641/wav2vec2-phoneme-models/outputs outputs` |
| 2. Verify `word_morph_detail` from #2 | ✓ verified (405 tokens, 0 missing) | — |
| 3. State-space spec for MLS French | ✓ committed | 4532942 (`scripts/generate_state_space_specs.py:compute_word_state_space_no_syllable`, `conf/dataset/mls_french.yaml:has_syllables`) |
| 4. `prepare_inputs_french.ipynb` | ✓ committed | 7bfeff3 |
| 5. `run_french.ipynb` | ✓ committed | 7bfeff3 + ec22478 |
| 6. Snakefile wiring | ✓ committed | 9464b70 |
| 7. End-to-end smoke test | ✓ passed | `snakemake --allowed-rules prepare_analogy_inputs_french run_analogy_experiment_french -- outputs/analogy/runs/mls_french-dev/w2v2_pc_fr_8/experiment_results.csv` |
| 8. Diagnostics in notebook | ✓ committed | 7bfeff3 (filter decomposition + FF spot-check cells) |

### Smoke-test result summary

- 3 true-friend rows survive end-to-end: (apporte, apportés, e::3sg.ind.pres→pp.masc.pl) ×1; (fut, furent, ʁ::hist.past.3sg→hist.past.3pl) ×2.
- 0 false-friend rows — see "Open scientific question" above.
- `experiment_results.csv` is empty (0 rows) because every `group_by` group has <2 distinct (base, inflected) pairs. The notebook handles this gracefully (guarded `correct` assignment).

### Re-running the smoke test (after worktree rebuild)

```bash
ln -s /Users/jon/.superset/worktrees/e4c3a981-cd98-4323-81d0-358a6fc04641/wav2vec2-phoneme-models/outputs outputs
mkdir -p data/french_morphology
gh api -H "Accept: application/vnd.github.v3.raw" repos/cbreiss/SpeechModelMorphology/contents/French/outputs/suffix_directions_report.csv > data/french_morphology/suffix_directions_report.csv
TOP10_SHA=$(gh api repos/cbreiss/SpeechModelMorphology/contents/French/outputs/suffix_pairs_top10.csv --jq .sha)
gh api repos/cbreiss/SpeechModelMorphology/git/blobs/$TOP10_SHA --jq .content | base64 -d > data/french_morphology/suffix_pairs_top10.csv
# Regenerate state-space spec for MLS French dev:
rm -f outputs/state_space_specs/mls_french-dev/w2v2_pc_fr_8/state_space_specs.h5
HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH=$(pwd) uv run python scripts/generate_state_space_specs.py \
  hydra.run.dir=outputs/state_space_specs/mls_french-dev/w2v2_pc_fr_8 \
  dataset=mls_french base_model=w2v2_pc_fr_8 \
  dataset.processed_data_dir=outputs/preprocessed_data/mls_french-dev \
  +base_model.hidden_state_path=outputs/hidden_states/w2v2_pc_fr_8/mls_french-dev.h5 \
  +analysis.state_space_specs_path=outputs/state_space_specs/mls_french-dev/w2v2_pc_fr_8/state_space_specs.h5
# Run the analogy rules (--allowed-rules bypasses the raw-audio re-extract that
# snakemake otherwise insists on chasing through the symlinked outputs):
uv run snakemake --cores 1 --rerun-incomplete \
  --allowed-rules prepare_analogy_inputs_french run_analogy_experiment_french \
  -- outputs/analogy/runs/mls_french-dev/w2v2_pc_fr_8/experiment_results.csv
```

## Code reuse

- `src/analysis/analogy.py:iter_equivalences` — **reuse unchanged**. Works with any `all_cross_instances` parquet that has the expected columns.
- `src/analysis/analogy.py:run_experiment_equiv_level` — **reuse unchanged**.
- `src/analysis/analogy.py:prepare_false_friends` — **not used** for French. Type is pre-labeled in the CSV.
- `src/analysis/analogy.py:get_inflection_df` — **not used** for French. Replaced by the suffix_pairs ⋈ directions_report join.
- `src/analysis/state_space.py:StateSpaceAnalysisSpec` — **reuse unchanged** once the French word computer produces a compatible spec.

## Pitfalls and notes

- **`exclude_main` semantics**: in English, `exclude_main = True` for false-friend and forced-choice rows so they're excluded from the basic accuracy headline. Mirror this for French.
- **`base` and `inflected` are orth strings, NOT lemmas**: state_space_spec.labels are orth forms from MLS. Don't accidentally key on `base_lemma`/`derived_lemma`.
- **String-quote escaping in query()**: `base_cell` values contain `|` (e.g. `'indicative|present|singular|third-person'`). Pandas `query()` handles this fine inside string literals, but watch for `'` in cell strings (none expected, but check before assuming).
- **The `→` character in `inflection`**: pandas `query()` works with it as long as it's only inside string literals. Don't try to put it in a column name.
- **Multiple morph cells per token**: if spaCy returns more than one possible cell (e.g. ambiguous between `indicative` and `subjunctive` for some present-tense forms), #2 should pick one deterministically — verify when #2 lands.
- **Dev set is 8 utterances**: the smoke test will produce very low-confidence numbers. The user knows this. Don't add error handling for "zero pairs survive" — let it raise and document it.
- **Don't refactor analogy.py for French**: keep changes additive. The English notebooks need to keep working.
