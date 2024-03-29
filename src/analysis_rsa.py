from dataclasses import dataclass
import logging
from pathlib import Path
from typing import cast, TypeAlias

import datasets
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from jaxtyping import Float
import numpy as np
import mat73
from omegaconf import DictConfig
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr
import torch
from tqdm.auto import tqdm, trange

from src.analysis.state_space import StateSpaceAnalysisSpec
from src.datasets.speech_equivalence import SpeechEquivalenceDataset, SpeechHiddenStateDataset
from src.encoding import ecog
from src.encoding.ecog import timit as timit_encoding
from src.encoding.ecog.timit import OutFileWithAnnotations, OutFile


L = logging.getLogger(__name__)


def normalize(data: Float[np.ndarray, "num_epochs num_channels num_timepoints"], method: str) -> Float[np.ndarray, "num_epochs num_channels num_timepoints"]:
    if method == "zscore":
        return (data - data.mean(axis=2, keepdims=True)) / data.std(axis=2, keepdims=True)
    elif method == "minmax":
        return (data - data.min(axis=2, keepdims=True)) / (data.max(axis=2, keepdims=True) - data.min(axis=2, keepdims=True))
    elif method == "none" or method is None:
        return data
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def do_ecog_rsa(aligned: ecog.AlignedECoGDataset, config: DictConfig):
    # epochs: (n_epochs, n_electrodes, n_timepoints)
    epochs, epoch_info = ecog.epoch_by_state_space(
        aligned, config.analysis.state_space, config.analysis.epoch_window.ecog,
        baseline_window=config.analysis.epoch_baseline_window.ecog
    )
    epochs = cast(np.ndarray, epochs)
    epochs = normalize(epochs, config.analysis.normalize.ecog)
    
    all_distances = [
        pdist(epochs[:, electrode_idx, :], metric=config.analysis.distance_metric.ecog)
        for electrode_idx in trange(epochs.shape[1], desc="Computing ECoG RSA", unit="electrode")
    ]
    return np.array(all_distances), cast(pd.DataFrame, epoch_info)


def do_model_rsa(epoch_info: pd.DataFrame, aligned: ecog.AlignedECoGDataset, config: DictConfig):
    model_epochs = []

    assert config.analysis.epoch_baseline_window.model is None, "Baseline subtraction not supported for model data"

    model_epoch_window = config.analysis.epoch_window.model
    # convert to model samples. TODO magic number
    model_epoch_window = (int(model_epoch_window[0] * 50), int(model_epoch_window[1] * 50))
    for _, epoch in tqdm(epoch_info.iterrows(), total=len(epoch_info), desc="Computing model RSA", unit="epoch"):
        start_frame, end_frame = np.array(epoch.span_model_frames) + epoch.item_start_frame

        epoch_start = max(0, start_frame + model_epoch_window[0])
        epoch_end = min(aligned._snapshot.hidden_states.num_frames,
                        start_frame + model_epoch_window[1])
        
        epoch_data = aligned._snapshot.embeddings[epoch_start:epoch_end].T
        model_epochs.append(epoch_data)

    model_epoch_data = np.stack(model_epochs)
    model_epoch_data = normalize(model_epoch_data, config.analysis.normalize.model)

    model_epoch_data = model_epoch_data.reshape(model_epoch_data.shape[0], -1)
    model_dists = pdist(model_epoch_data, metric=config.analysis.distance_metric.model)

    return model_dists


def main(config):
    """
    Model--brain comparison by RSA.
    """

    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # All data should be from the same subject
    assert len(config.data) == 1
    data_spec = config.data[0]

    assert len(config.feature_sets.model_features) == 1
    feature_spec = next(iter(config.feature_sets.model_features.values()))

    # load model embeddings etc.
    ce_model = ecog.ContrastiveModelSnapshot.from_config(config, feature_spec)

    # prepare aligned ECoG d ata
    aligned = ecog.AlignedECoGDataset(ce_model, timit_encoding.prepare_out_file(config, data_spec))

    # Prepare electrode metadata
    electrode_df = ecog.get_electrode_df(config, data_spec.subject)

    if len(electrode_df) != aligned.out[0]["resp"].shape[0]:
        L.warning(f"Electrode count mismatch: {len(electrode_df)} electrodes in electrode_df, {aligned.out[0]['resp'].shape[0]} electrodes in aligned data. Will subset electrode_df to match data")
        electrode_df = electrode_df.loc[:aligned.out[0]["resp"].shape[0] - 1]

    ecog_distances, epoch_info = do_ecog_rsa(aligned, config)
    model_distances = do_model_rsa(epoch_info, aligned, config)

    model_electrode_dists = np.array([
        spearmanr(model_distances, ecog_distances_i).statistic
        for ecog_distances_i in tqdm(ecog_distances, desc="Computing model--electrode correlations", unit="electrode")
    ])

    electrode_df["model_electrode_dist"] = model_electrode_dists
    electrode_df.to_csv(out_dir / "model_electrode_dists.csv")