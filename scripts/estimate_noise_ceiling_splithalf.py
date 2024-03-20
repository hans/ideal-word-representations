"""
Estimate brain encoding noise ceiling performance using split-half correlation.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from src.encoding.ecog import get_electrode_df


L = logging.getLogger(__name__)


def estimate_subject_noise_ceiling(config, data_spec) -> pd.DataFrame:
    electrodes_df = get_electrode_df(config, data_spec.subject)

    data_dir = Path(config.corpus.paths.data_path) / data_spec.subject / config.corpus.name / "block_z"
    outfile = list(data_dir.glob(f"{data_spec.subject}_{data_spec.block}_{config.corpus.paths.out_file_glob}"))
    assert len(outfile) == 1

    cout = loadmat(outfile[0], simplify_cells=True)
    out = cout['out']

    repeat_idxs = [idx for idx, data in enumerate(out) if data["resp"].ndim == 3]
    all_repeats = [out[idx]["resp"].transpose(2, 0, 1) for idx in repeat_idxs]
    L.info(f"Loaded {len(all_repeats)} repeats for {data_spec.subject} {data_spec.block}")

    num_electrodes = all_repeats[0].shape[1]
    dataf = out[0]["dataf"]

    # Evaluate split-half correlation on each repeat item
    splithalf_corrs = np.zeros((len(all_repeats), num_electrodes), dtype=float)
    for idx, repeat in enumerate(all_repeats):
        # Cut off padding
        before_pad, after_pad = out[0]["befaft"]
        before_pad_samples, after_pad_samples = int(before_pad * dataf), int(after_pad * dataf)
        repeat = repeat[:, :, before_pad_samples:-after_pad_samples]

        for electrode in range(num_electrodes):
            split1, split2 = train_test_split(repeat[:, electrode], test_size=0.5)
            splithalf_corrs[idx, electrode] = np.corrcoef(split1, split2)[0, 1]

    corrs_df = pd.DataFrame(splithalf_corrs, index=pd.Index(repeat_idxs, name="trial_idx")) \
        .reset_index().melt(id_vars="trial_idx", var_name="electrode_idx", value_name="correlation")
    corrs_df["electrode_name"] = corrs_df.electrode_idx.map(dict(enumerate(electrodes_df.index)))
    corrs_df["subject"] = data_spec.subject
    corrs_df["block"] = data_spec.block

    return corrs_df


@hydra.main(config_path="../conf_encoder", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    all_corrs = pd.concat([
        estimate_subject_noise_ceiling(config, data_spec)
        for data_spec in config.data
    ])

    all_corrs.to_csv(Path(HydraConfig.get().runtime.output_dir) / "splithalf_corrs.csv", index=False)


if __name__ == "__main__":
    main()