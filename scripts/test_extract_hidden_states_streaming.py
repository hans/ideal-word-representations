"""
Test that the streaming (out_path) path of extract_hidden_states produces output
bit-equivalent to the in-memory path.

Usage:
    uv run python scripts/test_extract_hidden_states_streaming.py

Exits 0 on PASS or if the model is not cached (graceful skip).
Exits 1 on FAIL.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
from datasets import Dataset

rng = np.random.default_rng(0)
items = [
    {"input_values": rng.standard_normal(rng.integers(8000, 48000)).astype(np.float32)}
    for _ in range(20)
]
ds = Dataset.from_list(items).with_format("torch")

# ---------------------------------------------------------------------------
# Load model (skip gracefully if not cached)
# ---------------------------------------------------------------------------
import transformers  # noqa: E402

MODEL_ID = "facebook/wav2vec2-base"
PROCESSOR_ID = "facebook/wav2vec2-base-960h"

try:
    model = transformers.Wav2Vec2Model.from_pretrained(MODEL_ID, local_files_only=True)
except OSError:
    print(
        f"Model '{MODEL_ID}' is not cached locally. "
        "Run `huggingface-cli download facebook/wav2vec2-base` to cache it. "
        "Skipping test."
    )
    sys.exit(0)

try:
    fe = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
        PROCESSOR_ID, local_files_only=True
    )
    tok = transformers.Wav2Vec2CTCTokenizer.from_pretrained(
        PROCESSOR_ID, local_files_only=True
    )
    processor = transformers.Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
except OSError:
    print(
        f"Processor '{PROCESSOR_ID}' is not cached locally. "
        "Run `huggingface-cli download facebook/wav2vec2-base-960h` to cache it. "
        "Skipping test."
    )
    sys.exit(0)

# ---------------------------------------------------------------------------
# Import extract_hidden_states after confirming dependencies are available
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_hidden_states import extract_hidden_states  # noqa: E402

VALIDATE_SCRIPT = Path(__file__).parent / "validate_extract_hidden_states.py"
LAYER = 1
failures = []

# ---------------------------------------------------------------------------
# Run tests for both pseudo_causal settings
# ---------------------------------------------------------------------------
for pseudo_causal in [False, True]:
    label = f"pseudo_causal={pseudo_causal}"
    print(f"\n--- Testing {label} ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = str(Path(tmpdir) / "baseline.h5")
        streaming_path = str(Path(tmpdir) / "streaming.h5")

        # Old path: in-memory → to_hdf5
        print("  Running in-memory path...")
        mem_result = extract_hidden_states(
            ds,
            model,
            processor,
            layer=LAYER,
            pseudo_causal=pseudo_causal,
            out_path=None,
        )
        mem_result.to_hdf5(baseline_path)
        print(
            f"  In-memory: states={mem_result.states.shape}, frames={mem_result.num_frames}"
        )

        # New path: streaming
        print("  Running streaming path...")
        stream_result = extract_hidden_states(
            ds,
            model,
            processor,
            layer=LAYER,
            pseudo_causal=pseudo_causal,
            out_path=streaming_path,
        )
        print(
            f"  Streaming: states={stream_result.states.shape}, frames={stream_result.num_frames}"
        )

        # Validate with external script
        print("  Validating...")
        proc = subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT), baseline_path, streaming_path],
            capture_output=True,
            text=True,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            print(f"  FAIL [{label}]: validator returned {proc.returncode}")
            if stdout:
                print(f"    stdout: {stdout}")
            if stderr:
                print(f"    stderr: {stderr}")
            failures.append(label)
        else:
            print(f"  PASS [{label}]: {stdout}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if failures:
    print(f"FAIL — {len(failures)} test(s) failed: {failures}")
    sys.exit(1)
else:
    print("PASS — all tests passed")
    sys.exit(0)
