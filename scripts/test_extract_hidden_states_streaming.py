"""Verify streaming (out_path) path produces bit-exact output vs in-memory path.

Usage:
    uv run python scripts/test_extract_hidden_states_streaming.py

Exits 0 on PASS or graceful skip (model not cached). Exits 1 on FAIL.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    import numpy as np
    from datasets import Dataset

    rng = np.random.default_rng(0)
    items = [
        {"input_values": rng.standard_normal(rng.integers(8000, 48000)).astype(np.float32)}
        for _ in range(20)
    ]
    ds = Dataset.from_list(items).with_format("torch")

    import transformers

    MODEL_ID = "facebook/wav2vec2-base"
    PROCESSOR_ID = "facebook/wav2vec2-base-960h"

    try:
        model = transformers.Wav2Vec2Model.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        print(f"Model '{MODEL_ID}' not cached locally — skipping test.")
        return 0

    try:
        fe = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
            PROCESSOR_ID, local_files_only=True
        )
        tok = transformers.Wav2Vec2CTCTokenizer.from_pretrained(
            PROCESSOR_ID, local_files_only=True
        )
        processor = transformers.Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
    except OSError:
        print(f"Processor '{PROCESSOR_ID}' not cached locally — skipping test.")
        return 0

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.extract_hidden_states import extract_hidden_states

    validate_script = Path(__file__).parent / "validate_extract_hidden_states.py"
    failures = []

    for pseudo_causal in [False, True]:
        label = f"pseudo_causal={pseudo_causal}"
        print(f"\n--- Testing {label} ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = str(Path(tmpdir) / "baseline.h5")
            streaming_path = str(Path(tmpdir) / "streaming.h5")

            print("  Running in-memory path...")
            mem_result = extract_hidden_states(
                ds, model, processor, layer=1, pseudo_causal=pseudo_causal, out_path=None
            )
            mem_result.to_hdf5(baseline_path)
            print(f"  In-memory: {mem_result.num_frames} frames")

            print("  Running streaming path...")
            stream_result = extract_hidden_states(
                ds, model, processor, layer=1, pseudo_causal=pseudo_causal,
                out_path=streaming_path,
            )
            print(f"  Streaming: {stream_result.num_frames} frames")

            proc = subprocess.run(
                [sys.executable, str(validate_script), baseline_path, streaming_path],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(f"  FAIL [{label}]: {proc.stdout.strip()}")
                if proc.stderr.strip():
                    print(f"    stderr: {proc.stderr.strip()}")
                failures.append(label)
            else:
                print(f"  PASS [{label}]: {proc.stdout.strip()}")

    print()
    if failures:
        print(f"FAIL — {len(failures)} test(s) failed: {failures}")
        return 1
    print("PASS — all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
