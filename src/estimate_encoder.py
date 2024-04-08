from dataclasses import dataclass
import logging
from pathlib import Path
from typing import cast, TypeAlias

import datasets
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
import mat73
from omegaconf import DictConfig
import pandas as pd
from scipy.io import loadmat
import torch
from tqdm.auto import tqdm

from src.analysis.state_space import StateSpaceAnalysisSpec
from src.datasets.speech_equivalence import SpeechEquivalenceDataset, SpeechHiddenStateDataset
from src.encoding.ecog import timit as timit_encoding
from src.encoding.ecog.timit import prepare_xy
from src.encoding.ecog import get_electrode_df


L = logging.getLogger(__name__)


def main(config):
    """
    Estimate a single TRF encoder by concatenating one or more blocks of data.
    """

    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # All data should be from the same subject
    all_subjects = set(data_spec.subject for data_spec in config.data)
    assert len(all_subjects) == 1, f"All data should be from the same subject. Got: {all_subjects}"
    subject = all_subjects.pop()

    # Prepare electrode metadata
    electrode_df = get_electrode_df(config, subject)
    electrode_df.to_csv(out_dir / "electrodes.csv")

    all_xy = [prepare_xy(config, data_spec) for data_spec in config.data]
    X, Y, feature_names, feature_shapes, trial_onsets = timit_encoding.concat_xy(all_xy)

    cv_outer = instantiate(config.cv)
    cv_inner = instantiate(config.cv)

    # TODO check match between model sfreq and dataset sfreq

    best_model, preds, scores, coefs, best_hparams = timit_encoding.strf_nested_cv(
        X, Y, feature_names, feature_shapes,
        trf_kwargs=config.model,
        sfreq=config.model.sfreq, cv_outer=cv_outer, cv_inner=cv_inner)
    
    if len(scores) == 0:
        # No models converged. Save dummy outputs.
        scores_df = pd.DataFrame(
            [(fold, output_dim, np.nan)
             for fold in range(cv_outer.get_n_splits())
             for output_dim in range(Y.shape[1])],
            columns=["fold", "output_dim", "score"]
        )
        best_model = None
        coefs = []
        preds = np.zeros(Y.shape)
    else:
        scores_df = pd.DataFrame(
            np.array(scores),
            index=pd.Index(list(range(cv_outer.get_n_splits())), name="fold"),
            columns=pd.Index(list(range(scores[0].shape[0])), name="output_dim"))
        scores_df = scores_df.reset_index().melt(id_vars="fold", var_name="output_dim", value_name="score")

    scores_df["output_name"] = scores_df.output_dim.map(dict(enumerate(electrode_df.index)))
    scores_df.to_csv(out_dir / "scores.csv", index=False)

    # save best model
    torch.save(best_model, out_dir / "model.pkl")

    # save coef estimates per fold
    torch.save(coefs, out_dir / "coefs.pkl")

    preds = np.concatenate(preds, axis=0)
    np.save(out_dir / "predictions.npy", preds)