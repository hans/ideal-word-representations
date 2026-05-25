"""Validate that two HDF5 files produced by extract_hidden_states.py are equivalent.

Usage:
    python scripts/validate_extract_hidden_states.py baseline.h5 new.h5
"""

import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.speech_equivalence import SpeechHiddenStateDataset


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} baseline.h5 new.h5")
        sys.exit(1)

    baseline_path, new_path = sys.argv[1], sys.argv[2]

    print(f"Loading baseline: {baseline_path}")
    baseline = SpeechHiddenStateDataset.from_hdf5(baseline_path)

    print(f"Loading new:      {new_path}")
    new = SpeechHiddenStateDataset.from_hdf5(new_path)

    failures: list[str] = []

    # ------------------------------------------------------------------
    # 1. flat_idxs: must be array-equal (same order, same values)
    # ------------------------------------------------------------------
    baseline_fi = np.array(baseline.flat_idxs)
    new_fi = np.array(new.flat_idxs)

    if baseline_fi.shape != new_fi.shape:
        failures.append(
            f"flat_idxs shape mismatch: baseline={baseline_fi.shape}, new={new_fi.shape}"
        )
    elif not np.array_equal(baseline_fi, new_fi):
        n_diff = int(np.sum(np.any(baseline_fi != new_fi, axis=1)))
        failures.append(
            f"flat_idxs not equal: {n_diff}/{len(baseline_fi)} rows differ"
        )
    else:
        print("  flat_idxs:          OK")

    # ------------------------------------------------------------------
    # 2. compression_ratios: keys and values must match (rtol=1e-5)
    # ------------------------------------------------------------------
    b_keys = set(baseline.compression_ratios.keys())
    n_keys = set(new.compression_ratios.keys())

    if b_keys != n_keys:
        failures.append(
            f"compression_ratios key sets differ: "
            f"baseline has {len(b_keys)} items, new has {len(n_keys)} items"
        )
    else:
        cr_baseline = np.array(
            [baseline.compression_ratios[k] for k in sorted(b_keys)], dtype=np.float64
        )
        cr_new = np.array(
            [new.compression_ratios[k] for k in sorted(n_keys)], dtype=np.float64
        )
        if not np.allclose(cr_baseline, cr_new, rtol=1e-5):
            max_rdiff = np.max(np.abs(cr_baseline - cr_new) / (np.abs(cr_baseline) + 1e-12))
            failures.append(
                f"compression_ratios not close (rtol=1e-5): max relative diff = {max_rdiff:.2e}"
            )
        else:
            print("  compression_ratios: OK")

    # ------------------------------------------------------------------
    # 3. states: shape and values (rtol=1e-5, atol=1e-6)
    # ------------------------------------------------------------------
    b_shape = tuple(baseline.states.shape)
    n_shape = tuple(new.states.shape)

    if b_shape != n_shape:
        failures.append(
            f"states shape mismatch: baseline={b_shape}, new={n_shape}"
        )
    else:
        # Load in chunks to avoid OOM on large files
        chunk = 4096
        n_rows = b_shape[0]
        states_ok = True
        max_abs_diff = 0.0
        max_rel_diff = 0.0

        for start in range(0, n_rows, chunk):
            end = min(start + chunk, n_rows)
            b_chunk = baseline.states[start:end][()]
            n_chunk = new.states[start:end][()]
            if not np.allclose(b_chunk, n_chunk, rtol=1e-5, atol=1e-6):
                abs_diff = np.abs(b_chunk - n_chunk)
                rel_diff = abs_diff / (np.abs(b_chunk) + 1e-12)
                max_abs_diff = max(max_abs_diff, float(abs_diff.max()))
                max_rel_diff = max(max_rel_diff, float(rel_diff.max()))
                states_ok = False

        if not states_ok:
            failures.append(
                f"states not close (rtol=1e-5, atol=1e-6): "
                f"max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e}"
            )
        else:
            print("  states:             OK")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
