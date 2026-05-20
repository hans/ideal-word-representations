# MLS French Smoke Test Report

**Date:** 2026-05-20  
**Branch:** `wav2vec2-phoneme-models` at `15f9dfc`  
**Sandbox:** linux/aarch64 (Docker)

---

## Executive summary

The smoke test hit a hard blocker at **Step 1** (MFA env creation). `baumwelch` — the
Python bindings for Kaldi used by all MFA ≥ 3.0 releases — is not packaged for
`linux-aarch64` on conda-forge. The MFA env cannot be created on this platform. Steps
2–7 were not reached.

Steps 4–7 (HF dataset, hidden states, equivalence datasets, `train_decoder`) are
independent of platform and are likely to work once MFA alignment output exists. The
blocker is entirely in the MFA sidecar.

---

## Pre-flight results

| Check | Result |
|---|---|
| Network egress to `huggingface.co` | ✅ OK |
| Network egress to `conda-forge` | ✅ OK |
| `.tools/micromamba` binary executable | ✅ v2.6.2 |
| `ffmpeg` | ❌ not installed, not installable without root |
| `soxi`/`ffprobe` | ❌ not available |
| `uv sync` / Python stack | ✅ OK |

`ffmpeg` is absent but this is not on the critical path: the fixture audio can be
generated directly with `scipy` as 16 kHz mono PCM WAV files (no FLAC-to-WAV
conversion needed when synthesising from scratch).

---

## Step 1 — MFA env creation: **BLOCKED**

```
error    libmamba Could not solve for environment specs
    The following package could not be installed
    └─ montreal-forced-aligner >=3.0 * is not installable because it requires
       └─ baumwelch =* *, which does not exist (perhaps a missing channel).
```

### Root cause

`baumwelch` is the conda-forge package that bundles the `_kalpy` Python bindings for
Kaldi (pybind11 wrappers around Kaldi C++ libs). It is not compiled for
`linux-aarch64`. Confirmed via:

```bash
.tools/micromamba search -c conda-forge baumwelch   # → "No entries matching"
```

All MFA ≥ 3.0 releases (checked: 3.0.7, 3.1.0, 3.3.9) list `baumwelch` as a
hard dependency. MFA 2.x also requires `baumwelch`.

Installing MFA via pip succeeds but fails at runtime:

```
ModuleNotFoundError: No module named '_kalpy'
```

The Kaldi C++ shared libraries (`libkaldi-*.so`) **are** available for
`linux-aarch64` on conda-forge (kaldi 5.5.1172 installs cleanly) — only the Python
bindings are missing.

### linux-64 is unblocked

Confirmed via dry-run:

```bash
.tools/micromamba create -n probe -c conda-forge montreal-forced-aligner \
    --platform linux-64 --dry-run
# → "Install: 268 packages, Total download: 379MB — Dry run."
```

The issue is strictly an aarch64 packaging gap.

---

## Audio fixture identified (not yet materialised)

While the MFA blocker prevents running the smoke test end-to-end, the MLS French dev
set was streamed from HuggingFace to identify fixtures that cover all required
transcript properties. The following 8 utterances from speaker **1591**, chapter
**1028** were verified:

| utt_id | Covers | Transcript (excerpt) |
|---|---|---|
| `1591_1028_000000` | clitic + hyphen + accent + long | "cependant il envoya chercher le pêcheur à l'heure même… cria-t-il apporte-moi…" |
| `1591_1028_000004` | clitic + hyphen + accent | "il fut saisi… la prétendue princesse… cria-t-elle au prince qui êtes-vous" |
| `1591_1028_000006` | clitic + hyphen + accent | "en même temps elle poursuivit… vous vizir ajouta-t-il…" |
| `1591_1028_000009` | clitic + accent | "le poisson n'ayant rien répondu… très distinctement oui oui…" |
| `1591_1028_000011` | clitic + hyphen + accent + short | "je suis lui répondit-elle la fille d'un roi des indes…" |
| `1591_1028_000016` | clitic + hyphen + accent | "le pêcheur ne lui dit pas… ce jour-là les poissons…" |
| `1591_1028_000017` | clitic + hyphen | "ne suffit-il pas qu'on l'accuse… quand il s'agit d'assurer les jours d'un roi" |
| `1591_1028_000019` | clitic + hyphen + accent + long | "mais ô prodige inouï… la cuisine s'entrouvrit… elle était habillée d'une étoffe" |

All utterances exercise:
- ✅ Clitics: `l'`, `qu'`, `d'`, `n'`, `s'`, `c'`
- ✅ Hyphenated forms: `cria-t-elle`, `ajouta-t-il`, `ne suffit-il`, `répondit-elle`, `ce jour-là`
- ✅ Accented vowels: `é`, `è`, `ê`, `â`, `ô`, `ç`, `î`
- ✅ Length variation: `1591_1028_000011` is short (one sentence); `1591_1028_000019` is long

These can be materialised via `datasets.load_dataset('facebook/multilingual_librispeech',
'french', split='dev', streaming=True)` once the MFA blocker is resolved.

---

## Steps 4–7: wiring assessment (not run)

These steps do not depend on MFA. Their primary risk factors are:

- **Step 4 (HF dataset):** The preprocessing notebook uses `src/datasets/huggingface_mls_french.py`. The known risk from the spec — clitic tokenization matching between MFA and the HF builder — cannot be validated until MFA alignment output exists.
- **Step 5 (hidden states):** `LeBenchmark/wav2vec2-FR-1K-base` is accessible from this sandbox (HuggingFace egress confirmed). `src/models/transformer.py:prepare_processor` supports `LeBenchmark/wav2vec2` prefix. Should work.
- **Step 6 (equivalence datasets):** Depends on HF dataset output. `word_broad` and `phoneme` classers are registered in `src/datasets/speech_equivalence.py`.
- **Step 7 (train_decoder no_train):** The `run_no_train` rule uses `phoneme_10frames` equivalence and `randomff_32` model. No GPU needed. Should work once equivalence datasets exist.

---

## How to unblock

### Chosen path: MFA_BACKEND=docker (or singularity on prod)

`.tools/mfa-env/run` now supports a `MFA_BACKEND` env var with three values:

| `MFA_BACKEND` | When to use |
|---|---|
| `conda` (default) | x86_64 Linux / macOS (Rosetta) |
| `docker` | local aarch64 sandbox — run manually from host |
| `singularity` | prod cluster — same image, `singularity exec` / `apptainer exec` |

The official MFA image (`mmcauliffe/montreal-forced-aligner:latest`) is linux/amd64 and
bundles kaldi + all Python bindings. The run script mounts `REPO_ROOT` and `MFA_ROOT_DIR`
so model downloads persist and all relative paths in the Snakefile rules work unchanged.

### Commands to run from the host (outside this sandbox)

```bash
# Clone path on host — adjust to wherever the repo lives
REPO=/path/to/ideal-word-representations

# Step 1 — verify MFA docker works
MFA_BACKEND=docker ${REPO}/.tools/mfa-env/run mfa version

# Step 2 — download French acoustic model + dictionary
MFA_BACKEND=docker snakemake --snakefile ${REPO}/Snakefile \
    --directory ${REPO} --cores 1 \
    data/mfa/.french_mfa_models_downloaded

# Step 3 — align (after test data is in data/mls_french/dev/)
MFA_BACKEND=docker snakemake --snakefile ${REPO}/Snakefile \
    --directory ${REPO} --cores 2 \
    data/mfa_alignments/mls_french/dev
```

Or if you want to run the Snakemake steps from inside this sandbox but have docker
available on the host daemon:

```bash
# Inside sandbox — docker socket must be reachable (e.g. Docker Desktop socket mount)
MFA_BACKEND=docker snakemake --cores 1 data/mfa/.french_mfa_models_downloaded
MFA_BACKEND=docker snakemake --cores 2 data/mfa_alignments/mls_french/dev
```

### Prod (Singularity / Apptainer)

```bash
MFA_BACKEND=singularity snakemake --cores 2 data/mfa_alignments/mls_french/dev
```

`apptainer` is preferred over `singularity` if both are in PATH. Override the image with
`MFA_SINGULARITY_IMAGE=docker://mmcauliffe/montreal-forced-aligner:v3.3.9` to pin a
version.

### Alternative: x86_64 sandbox

Any `linux-x86_64` machine can use the default `conda` backend without changes —
`baumwelch` resolves cleanly there (268 packages, confirmed by dry-run).

---

## Changes made to the repo during this run

- None committed. Exploratory changes (adding `librosa` to `pyproject.toml`, creating
  a throwaway `mfa2` conda env) were reverted / not committed.
