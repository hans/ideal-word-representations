"""
Validate that two SpeechHiddenStateDataset HDF5 files are equivalent.

Usage:
    uv run python scripts/validate_extract_hidden_states.py baseline.h5 new.h5

Exits 0 if the files are equivalent, 1 otherwise.
"""

import sys

import h5py
import numpy as np


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline.h5> <new.h5>", file=sys.stderr)
        sys.exit(1)

    baseline_path = sys.argv[1]
    new_path = sys.argv[2]

    errors = []

    with h5py.File(baseline_path, "r") as base, h5py.File(new_path, "r") as new:
        # Check model_name attribute
        base_name = base.attrs.get("model_name", "")
        new_name = new.attrs.get("model_name", "")
        if base_name != new_name:
            errors.append(f"model_name mismatch: {base_name!r} vs {new_name!r}")

        # Check states
        base_states = base["states"][:]
        new_states = new["states"][:]
        if base_states.shape != new_states.shape:
            errors.append(
                f"states shape mismatch: {base_states.shape} vs {new_states.shape}"
            )
        elif not np.allclose(base_states, new_states, rtol=1e-5, atol=1e-6):
            max_diff = np.abs(base_states - new_states).max()
            errors.append(f"states values differ: max absolute diff = {max_diff:.2e}")

        # Check flat_idxs
        base_flat = base["flat_idxs"][:]
        new_flat = new["flat_idxs"][:]
        if base_flat.shape != new_flat.shape:
            errors.append(
                f"flat_idxs shape mismatch: {base_flat.shape} vs {new_flat.shape}"
            )
        elif not np.array_equal(base_flat, new_flat):
            n_diff = (base_flat != new_flat).any(axis=1).sum()
            errors.append(f"flat_idxs values differ: {n_diff} rows differ")

        # Check compression_ratios
        base_cr = base["compression_ratios"][:]
        new_cr = new["compression_ratios"][:]
        if base_cr.shape != new_cr.shape:
            errors.append(
                f"compression_ratios shape mismatch: {base_cr.shape} vs {new_cr.shape}"
            )
        elif not np.allclose(base_cr, new_cr, rtol=1e-5, atol=1e-7):
            max_diff = np.abs(base_cr - new_cr).max()
            errors.append(
                f"compression_ratios values differ: max diff = {max_diff:.2e}"
            )

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED: files are equivalent")
        sys.exit(0)


if __name__ == "__main__":
    main()
