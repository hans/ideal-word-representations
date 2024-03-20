from dataclasses import dataclass
import logging
from pathlib import Path
from typing import cast

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
from src.encoding.ecog import get_electrode_df


L = logging.getLogger(__name__)


@dataclass
class ContrastiveModelSnapshot:
    """
    Snapshot of the training environment and output of a contrastive model used
    for brain encoding.
    """
    # TODO would be nicer to scaffold with Hydra instantiate :)

    feature_spec: DictConfig
    dataset: datasets.Dataset
    hidden_states: SpeechHiddenStateDataset
    equiv_dataset: SpeechEquivalenceDataset
    state_space: StateSpaceAnalysisSpec
    embeddings: np.ndarray

    @classmethod
    def from_config(cls, config: DictConfig, feature_spec: DictConfig):
        dataset = cast(datasets.Dataset, datasets.load_from_disk(config.dataset_path))

        with open(feature_spec.equivalence_path, "rb") as f:
            equiv_dataset: SpeechEquivalenceDataset = torch.load(f)
        with open(feature_spec.hidden_state_path, "rb") as f:
            hidden_states: SpeechHiddenStateDataset = torch.load(f)
        with open(feature_spec.state_space_path, "rb") as f:
            state_space: StateSpaceAnalysisSpec = torch.load(f)[feature_spec.state_space]
        embeddings = np.load(feature_spec.embeddings_path)

        return cls(
            feature_spec=feature_spec,
            dataset=dataset,
            hidden_states=hidden_states,
            equiv_dataset=equiv_dataset,
            state_space=state_space,
            embeddings=embeddings
        )

    def __post_init__(self):
        assert self.embeddings.shape[0] == self.hidden_states.num_frames


def load_and_align_model_embeddings(config, out):
    """
    Load model embeddings of TIMIT stimuli and align them with the
    corresponding ECoG responses / baseline features.
    """

    if len(config.feature_sets.model_features) != 1:
        raise NotImplementedError("Only one model feature set is supported")
    feature_spec = next(iter(config.feature_sets.model_features.values()))
    model = ContrastiveModelSnapshot.from_config(config, feature_spec)

    out_all_names = {out_i["name"] for out_i in out}
    name_to_item_idx, name_to_frame_bounds, compression_ratios = {}, {}, {}
    def process_item(item):
        name = Path(item["file"]).parent.stem.lower() + "_" + item["id"].lower()
        if name in out_all_names:
            name_to_item_idx[name] = item["idx"]

            frame_start, frame_end = model.hidden_states.frames_by_item[item["idx"]]
            name_to_frame_bounds[name] = (frame_start, frame_end)
            compression_ratios[name] = (frame_end - frame_start) / len(item["input_values"])
    model.dataset.map(process_item)

    # Make sure that ECoG data and model embeddings are of approximately the same length,
    # modulo sampling differences. Compute length of each sentence in seconds according
    # to two sources:
    comparisons = [(out_i["resp"].shape[1] / 100 - 1, # remove padding
                    (name_to_frame_bounds[out_i['name']][1] - name_to_frame_bounds[out_i['name']][0]) / compression_ratios[out_i["name"]] / 16000)
                   for out_i in out if out_i["name"] in name_to_frame_bounds]
    np.testing.assert_allclose(*zip(*comparisons), atol=0.05,
                               err_msg="ECoG data and model embeddings should be of approximately the same length")

    # Scatter model embeddings into a time series aligned with ECoG data.
    n_model_dims = model.embeddings.shape[1]
    embedding_misses = 0
    embedding_scatter_hits = 0
    for i, out_i in enumerate(tqdm(out, desc="Aligning model embeddings")):
        name = out_i["name"]

        n_ecog_samples = out_i["resp"].shape[1]
        embedding_data = np.zeros((n_model_dims, n_ecog_samples))
        out_i["model_embedding"] = embedding_data

        try:
            item_idx = name_to_item_idx[name]
            frame_start, frame_end = name_to_frame_bounds[name]
        except KeyError:
            # TODO how to handle this -- we shouldn't score on sentences not included
            # in the embedding data
            L.warning(f"Skipping {name} as it is not in the embedding set")
            embedding_misses += 1
            continue

        # Now scatter model embeddings according to the spans in the state
        # space specification
        unit_spans = model.state_space.get_trajectories_in_span(frame_start, frame_end)
        for unit_start, unit_end, _, _ in unit_spans:
            # Compute aligned ECoG sample
            # magic numbers: 16 KHz audio sampling rate
            unit_start_secs = (unit_start - frame_start) / compression_ratios[name] / 16000
            # magic numbers: 0.5 seconds of before-padding; 100 Hz sampling rate
            unit_start_ecog = int((0.5 + unit_start_secs) * 100)

            # Scatter an impulse predictor at the sample aligned to the onset of the unit
            unit_embedding = None
            if feature_spec.featurization == "last_frame":
                unit_embedding = model.embeddings[unit_end]
            elif feature_spec.featurization == "mean":
                unit_embedding = model.embeddings[unit_start:unit_end].mean(axis=0)
            else:
                raise ValueError(f"Unknown featurization {feature_spec.featurization}")

            embedding_data[:, unit_start_ecog] = unit_embedding
            embedding_scatter_hits += 1

        out_i["model_embedding"] = embedding_data

    if embedding_misses > 0:
        L.warning(f"Skipped {embedding_misses} sentences ({embedding_misses / len(out) * 100}%) as they were not in the embedding set")
    L.info(f"Scattered model embeddings for {embedding_scatter_hits} {feature_spec.state_space} units")

    return out


def load_out_file(path):
    """
    Load preprocessed ECoG data, accounting for different Matlab export
    formats.
    """
    try:
        return loadmat(path, simplify_cells=True)
    except NotImplementedError:
        # Matlab >= 7.3 format. Load using `mat73` and simulate the
        # simplify_cells functionality for the fields we care about
        data = mat73.loadmat(path)

        # Simplify cell representation
        target_cells = ["name", "resp", "dataf", "befaft"]
        ret = []
        for i in range(len(data["out"]["resp"])):
            ret.append({
                cell: data["out"][cell][i] for cell in target_cells
            })

        return {"out": ret}


def prepare_xy(config: DictConfig, data_spec: DictConfig) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int]]]:
    data_dir = Path(config.corpus.paths.data_path) / data_spec.subject / config.corpus.name / "block_z"

    if data_spec.block is None:
        full_glob = f"{data_spec.subject}_*_{config.corpus.paths.out_file_glob}"
    else:
        full_glob = f"{data_spec.subject}_{data_spec.block}_{config.corpus.paths.out_file_glob}"
    outfile = list(data_dir.glob(full_glob))
    assert len(outfile) == 1

    cout = load_out_file(outfile[0])
    out = cout["out"]

    # add sentence details to out
    out = timit_encoding.add_details_to_out(out, config.corpus.sentdetV,
                                            config.corpus.paths.info_path,
                                            config.corpus.name,
                                            data_spec.subject)
    feature_sets = config.feature_sets.baseline_features[:]

    center_features = [True] * len(feature_sets)
    scale_features = [True] * len(feature_sets)

    # add model embeddings to out
    if getattr(config.feature_sets, "model_features", None):
        out = load_and_align_model_embeddings(config, out)
        feature_sets.append("model_embedding")

        center_features.append(False)
        scale_features.append(False)

    X, Y, feature_names, feature_shapes = timit_encoding.prepare_strf_xy(
        out, feature_sets, data_spec.subject,
        center_features=np.array(center_features),
        scale_features=np.array(scale_features))
    
    return X, Y, feature_names, feature_shapes


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
    X, Y, feature_names, feature_shapes = timit_encoding.concat_xy(all_xy)

    cv_outer = instantiate(config.cv)
    cv_inner = instantiate(config.cv)

    # HACK
    sfreq = 100

    best_model, preds, scores, coefs, best_hparams = timit_encoding.strf_nested_cv(
        X, Y, feature_names, feature_shapes,
        sfreq=sfreq, cv_outer=cv_outer, cv_inner=cv_inner)
    
    scores_df = pd.DataFrame(
        np.array(scores),
        index=pd.Index(list(range(cv_outer.get_n_splits())), name="fold"),
        columns=pd.Index(list(range(scores[0].shape[0])), name="output_dim"))
    scores_df = scores_df.reset_index().melt(id_vars="fold", var_name="output_dim", value_name="score")
    scores_df["output_name"] = scores_df.output_dim.map(dict(enumerate(electrode_df.index)))
    scores_df.to_csv(out_dir / "scores.csv", index=False)

    coef_df = timit_encoding.trf_grid_to_df(best_model, coefs,
                                            output_names=electrode_df.index)
    coef_df.to_csv(out_dir / "coefs.csv", index=False)

    preds = np.concatenate(preds, axis=0)
    np.save(out_dir / "predictions.npy", preds)