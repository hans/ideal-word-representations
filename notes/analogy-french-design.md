# French inflection analogy task — design notes (issue #1)

Working doc for the French analog of `notebooks/analogy/run.ipynb`. Captured before pausing #1 to land #2 (per-token morphological tagging of MLS French). When resuming #1, read this top to bottom, then start the implementation plan.

## Status

- Sketch approved by user via brainstorming Q&A on 2026-05-21.
- **Blocked on #2** — needs `word_morph_detail` (per-token UD morph features + CSV-cell strings) on the preprocessed MLS French dataset before the analogy can run.
- Branch state: `feat/issue-1` rebased onto `origin/wav2vec2-phoneme-models`; `outputs/` is a symlink to the wav2vec2-phoneme-models worktree's outputs (recreate on resume).

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

### Construction algorithm

1. **Pre-filter directions report**: keep rows with `n_stems >= 50 AND total_base_count >= 100 AND total_derived_count >= 100`. Yields ~700/1973 (suffix_ipa, base_cell, derived_cell) tuples.
2. **Join to suffix_pairs**: for each suffix_pairs row, split `base_tags` and `derived_tags` on ` ; ` to get the **set of possible cells** for each side. A directions-report tuple `(b, d)` is *consistent* with the row iff `b ∈ base_tag_set ∧ d ∈ derived_tag_set`. Emit one row per consistent direction.
3. **Filter to corpus-attested orth pairs**: drop rows where `base_orth` or `derived_orth` are not in `state_space_spec.labels`.
4. **Pull per-instance morph from `word_morph_detail` (from #2)**: for each `base_orth`, list its `(instance_idx, csv_cell)` pairs from the corpus. Same for `derived_orth`. Restrict to instances whose `csv_cell` equals the row's chosen `base_cell` (or `derived_cell`).
5. **Cartesian product over surviving instances**: every (base_instance × derived_instance) pair becomes a row. (This is identical to how the English notebook builds its `all_cross_instances`.)
6. **Attach diagnostic columns** (`base_phones` etc.) and assemble final parquet.

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

## Open questions resolved by #2's output

| Question | Resolution |
|---|---|
| What schema does `word_morph_detail` have? | #2's issue body proposes `{upos, morph, csv_cell, lemma, alignment_confidence}` per token. Confirm format matches when #2 lands. |
| How are alignment failures represented? | `alignment_confidence == "missing"` → `csv_cell = null`. We'll drop those instances in step 4 of `all_cross_instances` construction. |
| How exhaustive is the UD→CSV cell mapping? | #2 includes a mapping table + unit tests covering the ~92 distinct cells observed in `suffix_pairs_top10.csv`. Cross-check coverage after #2 lands. |

## Implementation order (when resuming #1)

1. **Recreate outputs symlink** to wav2vec2-phoneme-models worktree (see top of this doc).
2. **Verify #2's output**: load `outputs/preprocessed_data/mls_french-dev/`, confirm `word_morph_detail` is present, spot-check 10 tokens.
3. **State-space spec**: add `compute_word_state_space_french` to `scripts/generate_state_space_specs.py` (or a sister script). Run, write `outputs/state_space_specs/mls_french-dev/w2v2_pc_fr_8/state_space_specs.h5`.
4. **prepare_inputs_french.ipynb**: implement the all_cross_instances construction above. Write parquet to `outputs/analogy/inputs/mls_french-dev/w2v2_pc_fr_8/`.
5. **run_french.ipynb**: load all_cross_instances + state_space_spec + hidden states (no trained probe — use w2v2 hidden states directly via `embeddings_path = "ID"` branch). Build the `experiments` dict, run `analogy.run_experiment_equiv_level` over each, save `experiment_results.csv`.
6. **Snakefile wiring**: add rules to `Snakefile` for `analogy/inputs` and `analogy/runs` mirroring the English equivalents, with `mls_french-dev` + `w2v2_pc_fr_8` wildcards.
7. **Smoke test**: run end-to-end. Expect ~10 (base, derived) pairs to flow through; `experiment_results.csv` should have rows, accuracy will be ~chance.
8. **README/docs**: note in commit message + close-out comment that the dev set is intentionally tiny and the headline numbers come from a larger preprocessed split (future work).

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
