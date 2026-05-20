# MLS French scaffold — smoke-test handoff

This spec hands off the first end-to-end run of the MLS French scaffold
(commit `15f9dfc`). The goal is to **prove the pipeline composes correctly on
a tiny slice** before anyone commits prod-GPU time to the full ~1.1k-hour
training run. Everything here should run on a CPU-only sandbox in well under
an hour.

## Goal

Drive five points of data through the pipeline:

```
raw audio + transcripts
  -> MFA TextGrid alignments
  -> HF dataset (preprocessed_data/mls_french-<split>)
  -> hidden_states/.../mls_french-<split>.h5
  -> equivalence_datasets/.../{word_broad,phoneme}/equivalence.pkl
  -> train_decoder in no_train mode (verifies the train wiring loads end-to-end)
```

If all six artefacts build without manual intervention, the scaffold is good.

## Non-goals

- **No model training.** GPU training happens on prod; here we only verify the
  data + wiring. `train_decoder` runs in `trainer.mode=no_train` only.
- **No full corpus download.** ~250 GB on disk for the FLAC distribution.
- **No analogy experiments.** That pipeline is heavily English-specific and
  out of scope for this scaffold.
- **No benchmarking.** Wall-clock and memory aren't measured; correctness is.

## Prereqs you can assume

- Branch: `wav2vec2-phoneme-models` at or descended from commit `15f9dfc`
  ("scaffold mls french representation extraction + contrast model training").
- `uv sync` works in the sandbox (verified on linux/aarch64 + macOS).
- `.tools/micromamba` binary is present and executable (already in the repo).
- Network egress to:
  - `huggingface.co` (wav2vec2-FR-1K-base weights)
  - `conda-forge` (montreal-forced-aligner)
  - `mfa-models.readthedocs.io` (MFA model index)
  - wherever you choose to source test audio.

## Prereqs you must establish

### Test audio

You need 5–10 French utterances with ground-truth transcripts, in the MLS
layout. **Pick the cheapest reachable option from your sandbox:**

| Option | Cost | Notes |
|---|---|---|
| Download the full `mls_french.tar.gz` from openslr.org | ~37 GB compressed | The download rule (`snakemake data/mls_french/dev`) does this. Slow but real. |
| Subselect ~10 utterances from an existing local MLS mirror | ~free | If you find an MLS copy on the sandbox's filesystem, copy the relevant `audio/<spk>/<book>/*.flac` + the matching lines of `transcripts.txt` into a fresh tree. |
| Source 5–10 short French audio clips from Common Voice fr, Librivox samples, or recorded snippets | minutes | You'll need to hand-author a `transcripts.txt` and arrange them in MLS layout (see below). MFA will accept this just fine. |

**Required layout (whichever route you pick):**

```
data/mls_french_smoke/dev/
  transcripts.txt                            # <utt_id>\t<orthographic text>
  audio/
    <spk>/
      <book>/
        <utt_id>.wav                         # 16 kHz mono PCM
```

`<utt_id>` must be `<spk>_<book>_<seg>` (e.g. `1746_1093_000004`). Convert FLAC
to 16 kHz mono WAV with ffmpeg if needed.

**Transcript content matters.** Include utterances that exercise:

- a clitic (`l'`, `qu'`, `d'`, `n'`, `c'`) — tests apostrophe tokenization
- a hyphenated word (`peut-être`, `est-ce`) — tests hyphen tokenization
- multiple accented vowels (`é à è ç`) — tests Unicode normalization
- at least one short and one long utterance (variation in frame counts)

The advisor's open question was whether MFA's tokenization of clitics matches
what the HF builder assumes. **This is the single biggest silent-failure risk
in the scaffold.** Step 3 below is where you verify it.

### Path overrides

The default Snakefile rules use `data/mls_french/<split>`. For the smoke run,
either:

- Mirror your fixture under that exact path, **or**
- Invoke each rule with explicit input/output paths via `snakemake --config` /
  `-D`. Mirroring the path is simpler — recommend that.

## Steps

### Step 1 — Provision the MFA side-car env

```bash
.tools/mfa-env/run mfa version
```

Expected: prints an MFA version (≥ 3.0) and exits 0. First invocation creates
`.mamba/envs/mfa/` and prints `[mfa-env] Creating mfa env...`; subsequent
invocations are instant.

**On failure:**
- micromamba not executable → `chmod +x .tools/micromamba`.
- conda-forge resolution timeout → retry; if persistent, check sandbox egress.
- "command not found: mfa" → env created but MFA not in it; inspect
  `.tools/mfa-env/environment.yml` and re-run after deleting `.mamba/envs/mfa`.
- **`baumwelch` not installable (linux-aarch64)** → the conda backend cannot resolve
  MFA on ARM sandboxes. Use the Docker or Singularity backend instead:
  ```bash
  MFA_BACKEND=docker .tools/mfa-env/run mfa version      # Docker
  MFA_BACKEND=singularity .tools/mfa-env/run mfa version  # prod cluster
  ```
  All subsequent MFA steps accept the same `MFA_BACKEND=docker` prefix.
  See `.tools/mfa-env/run` for image override env vars.

### Step 2 — Download MFA French models

```bash
snakemake --cores 1 data/mfa/.french_mfa_models_downloaded
```

Expected: `data/mfa/.french_mfa_models_downloaded` (an empty sentinel) exists.
The acoustic model + IPA dictionary land under `.mamba/mfa/` (the
`MFA_ROOT_DIR` set in the run wrapper).

**On failure:**
- MFA model server down → retry; check `https://mfa-models.readthedocs.io/`.
- 404 on `french_mfa` → the model may have been renamed upstream. `.tools/mfa-env/run mfa model list acoustic` lists what's available; update
  `MFA_ACOUSTIC` / `MFA_DICTIONARY` constants in `workflows/mls_french/Snakefile`.

### Step 3 — Align the fixture (critical verification)

Assuming your fixture is at `data/mls_french/dev/`:

```bash
snakemake --cores 2 data/mfa_alignments/mls_french/dev
```

This runs `prepare_mfa_corpus` (materializes `.lab` files + audio symlinks at
`data/mfa_corpus/mls_french/dev/`) then `mfa_align` (writes TextGrids to
`data/mfa_alignments/mls_french/dev/`).

Expected: one `.TextGrid` per utterance under
`data/mfa_alignments/mls_french/dev/<spk>/<book>/<utt_id>.TextGrid`.

**Critical inspection** — pick one TextGrid containing a clitic-bearing
utterance and `cat` it. Verify on the `words` tier:

- Does MFA emit clitics (`l'`, `qu'`) as **separate** word intervals, or fused
  with the following word? Whatever MFA does is fine; the builder reads it
  verbatim. But note the answer.
- Hyphenated words: split or fused? Note the answer.
- On the `phones` tier, are accented vowels in IPA (e.g. `e`, `ɛ`, `ɛ̃`, `ə`)?
  Note the inventory — particularly whether nasal vowels are precomposed
  (`ɛ̃`) or use combining diacritics (`ɛ` + U+0303). The HF builder NFC-
  normalizes, so combining forms will collapse to precomposed.

**Report these observations back.** If MFA's tokenization surprises you,
escalate before continuing — downstream `word_broad` equivalence classes
depend on it being self-consistent.

**On failure:**
- "Pronunciation dictionary contains no entry for <word>" → your transcript
  has an OOV word; either drop that utterance or accept MFA's fallback OOV
  handling.
- "Could not align <utt_id>" on >50% of utterances → audio sample rate /
  encoding mismatch. Confirm 16 kHz mono PCM via `soxi` or `ffprobe`.

### Step 4 — Preprocess into HF dataset

```bash
snakemake --cores 1 outputs/preprocessed_data/mls_french-dev
```

This runs `notebooks/preprocessing/mls_french.ipynb` via papermill, which uses
the `src/datasets/huggingface_mls_french.py` builder.

Expected:
- `outputs/preprocessed_data/mls_french-dev/` exists, with `dataset_info.json`,
  `state.json`, and one or more arrow shards.
- `outputs/preprocessing/mls_french-dev/notebook.ipynb` (the executed copy)
  exists and ran without exceptions.

**Sanity check** — load the dataset and spot-check one item:

```bash
uv run python -c "
import datasets
ds = datasets.load_from_disk('outputs/preprocessed_data/mls_french-dev')
print(ds)
item = ds[0]
print('text:', repr(item['text']))
print('word_detail utts:', item['word_detail']['utterance'][:8])
print('phonetic_detail utts:', item['phonetic_detail']['utterance'][:12])
print('word_phonemic_detail[0]:', item['word_phonemic_detail'][0] if item['word_phonemic_detail'] else None)
print('input_values len:', len(item['input_values']))
"
```

Expect:
- `text` is NFC-normalized lowercase French.
- `word_detail.utterance` matches MFA's word tier (including whatever clitic
  treatment you observed in step 3).
- `phonetic_detail.utterance` contains IPA strings.
- `word_phonemic_detail[0]` is a list of `{phone, start, stop}` dicts grouped
  under the first word. **If this is empty or all `None`, the grouping in the
  notebook silently failed** — bisect.

**On failure:**
- `KeyError: 'phonemic_detail'` → the notebook's `add_phonemic_detail` cell
  didn't run; check the executed papermill notebook for an earlier exception.
- `feature_extractor` errors on first audio item → confirm audio is 16 kHz
  mono float32-compatible.

### Step 5 — Extract hidden states

```bash
snakemake --cores 1 outputs/hidden_states/w2v2_pc_fr_8/mls_french-dev.h5
```

This downloads `LeBenchmark/wav2vec2-FR-1K-base` (≈ 360 MB) on first run, then
runs `scripts/extract_hidden_states.py` in pseudocausal mode at layer 8.

Expected: an HDF5 file containing per-frame layer-8 hidden states for every
utterance in the preprocessed dataset.

**Sanity check:**

```bash
uv run python -c "
import h5py, numpy as np
with h5py.File('outputs/hidden_states/w2v2_pc_fr_8/mls_french-dev.h5','r') as f:
    print('keys:', list(f.keys()))
    states = f['hidden_states'][:]
    print('hidden_states shape:', states.shape)
    print('dtype:', states.dtype)
    print('finite fraction:', np.isfinite(states).mean())
"
```

Expect a non-trivial `(num_frames, 1, 768)` array (768 = wav2vec2-base hidden
size), finite fraction = 1.0.

**On failure:**
- `NotImplementedError` from `prepare_processor` → the model_ref didn't pass
  the relaxed check in `src/models/transformer.py`. Re-read the file.
- Pseudocausal extraction OOMs on CPU → reduce `batch_size` in
  `scripts/extract_hidden_states.py:extract_representations_pseudo_causal`
  for the smoke run; do **not** commit that change.

### Step 6 — Build equivalence datasets

```bash
snakemake --cores 1 \
  outputs/equivalence_datasets/mls_french-dev/w2v2_pc_fr_8/word_broad_10frames_fixedlen25/equivalence.pkl \
  outputs/equivalence_datasets/mls_french-dev/w2v2_pc_fr_8/phoneme_10frames_fixedlen25/equivalence.pkl
```

Expected: two pickle files exist.

**Sanity check:**

```bash
uv run python -c "
import pickle
for path in [
    'outputs/equivalence_datasets/mls_french-dev/w2v2_pc_fr_8/word_broad_10frames_fixedlen25/equivalence.pkl',
    'outputs/equivalence_datasets/mls_french-dev/w2v2_pc_fr_8/phoneme_10frames_fixedlen25/equivalence.pkl',
]:
    eq = pickle.load(open(path,'rb'))
    print(path)
    print('  type:', type(eq).__name__)
    print('  num classes:', len(set(eq.class_labels)) if hasattr(eq,'class_labels') else 'n/a')
"
```

Expect: at least a handful of distinct word-broad classes (one per orthographic
word form in the fixture) and a handful of phoneme classes (one per IPA segment
attested).

**On failure:**
- Zero classes → the equivalence classer didn't find the field it expects on
  the dataset. Re-read `src/datasets/speech_equivalence.py:equivalence_classers`
  to confirm `word_broad` and `phoneme` field names match the HF builder
  output.

### Step 7 — Run train_decoder in no-train mode

```bash
snakemake --cores 1 \
  outputs/models/mls_french-dev/w2v2_pc_fr_8/randomff_32/random
```

This invokes `train_decoder.py trainer.mode=no_train` for the random baseline,
which constructs the model and dumps initial weights without actually
optimizing. CPU-friendly.

Expected: a model directory with the usual hydra job artefacts and an
initial-weights checkpoint.

**Do not attempt** the `ffff_32-pc-mAP1` / `ffff_32-pc-phon_mAP1` rules on this
sandbox — they'd actually train and are not in scope for the smoke test.

**On failure:**
- Hydra config interpolation errors → likely a path/typo in
  `conf/base_model/w2v2_pc_fr_8.yaml` or `conf/dataset/mls_french.yaml`.

## Known pre-existing issues

- `snakemake --list` fails at parse time because
  `workflows/librispeech/Snakefile:1` uses `storage: provider = "http"`, which
  requires `snakemake-storage-plugin-http` (not in `pyproject.toml`). The
  mls_french rules above all target specific output files, which doesn't
  require `--list`, so this doesn't block the smoke test. Don't try to fix it
  in this PR.
- `src/datasets/huggingface_librispeech.py` imports `datasets.tasks` which is
  gone in current `datasets`. Same dead import the MLS builder originally had,
  removed there. The LibriSpeech preprocessing notebook would also fail until
  the same fix is applied; out of scope for this smoke test.

## Acceptance — what to report back

A successful smoke test produces a short report covering:

1. Which test-data option you took, and what utterances you used (IDs +
   transcripts).
2. The clitic / hyphenation / accent observations from step 3 (this is the
   most important deliverable).
3. Confirmation that steps 4–7 each produced their expected artefacts.
4. Any sandbox-specific workarounds you applied that aren't already in the
   repo.

If anything between steps 3 and 7 surprises you, **stop and surface it**
before continuing — silent passes are worse than loud failures here.
