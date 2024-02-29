import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import pickle
from scipy.io import loadmat
from sklearn.model_selection import KFold, GridSearchCV
from tqdm.auto import tqdm
import yaml

from src.datasets.speech_equivalence import SpeechEquivalenceDataset
from src.encoding.ecog import timit as timit_encoding
from src.models.integrator import load_or_compute_embeddings


L = logging.getLogger(__name__)


def load_model_embeddings(config, data_spec):
    assert len(config.feature_sets.model_features) == 1
    model_feature_spec = config.feature_sets.model_features[0]
    model_dir = Path(f"outputs/models/{model_feature_spec.model}")
    with open(model_dir / ".hydra" / "config.yaml") as f:
        model_config = yaml.safe_load(f)

    # TODO clean up
    equiv_dataset_path = f"data/timit_equivalence_{model_config['model']['base_model_ref'].replace('/', '-')}_{model_config['model']['base_model_layer']}-phoneme-{model_config['equivalence']['num_frames_per_phoneme']}.pkl"
    with open(equiv_dataset_path, "rb") as f:
        equiv_dataset: SpeechEquivalenceDataset = pickle.load(f)

    model_embeddings = load_or_compute_embeddings(
        None, equiv_dataset,
        model_dir=f"outputs/models/{model_feature_spec.model}",
        equiv_dataset_path=equiv_dataset_path)
    return model_config, model_embeddings, equiv_dataset


def load_and_align_model_embeddings(config, data_spec, out):
    """
    Load model embeddings of TIMIT stimuli and align them with the
    corresponding ECoG responses / baseline features.
    """
    model_config, model_embeddings, equiv_dataset = load_model_embeddings(config, data_spec)
    # Load originating dataset
    dataset = instantiate(model_config['dataset'], processor=None)["train"]

    out_all_names = {out_i["name"] for out_i in out}

    name_to_item_idx, name_to_frame_bounds, compression_ratios = {}, {}, {}
    def process_item(item, idx):
        name = Path(item["file"]).parent.stem.lower() + "_" + item["id"].lower()
        if name in out_all_names:
            name_to_item_idx[name] = idx

            frame_start, frame_end = equiv_dataset.hidden_state_dataset.frames_by_item[idx]
            name_to_frame_bounds[name] = (frame_start, frame_end)
            compression_ratios[name] = (frame_end - frame_start) / len(item["input_values"])

    dataset.map(process_item, with_indices=True)

    # Make sure that ECoG data and model embeddings are of approximately the same length,
    # modulo sampling differences. Compute length of each sentence in seconds according
    # to two sources:
    comparisons = [(out_i["resp"].shape[1] / 100 - 1, # remove padding
                    (name_to_frame_bounds[out_i['name']][1] - name_to_frame_bounds[out_i['name']][0]) / compression_ratios[out_i["name"]] / 16000)
                   for out_i in out if out_i["name"] in name_to_frame_bounds]
    np.testing.assert_allclose(*zip(*comparisons), atol=0.05,
                               err_msg="ECoG data and model embeddings should be of approximately the same length")
    
    # Resample and align model embeddings to be the same size
    # as the ECoG data
    n_model_dims = model_embeddings.shape[1]
    for i, out_i in enumerate(out):
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
            continue

        # TODO make this logic customizable

        # Find annotated unit ends relative to start frame
        unit_ends = (equiv_dataset.Q[frame_start:frame_end] != -1).nonzero(as_tuple=True)[0]
        # Find unit starts. NB this only works assuming 1 annotated frame per unit (phoneme)
        unit_starts = np.concatenate([[0], unit_ends[:-1] + 1])

        for unit_start, unit_end in zip(unit_starts, unit_ends):
            # Compute aligned ECoG sample
            # magic numbers: 16 KHz audio sampling rate
            unit_start_secs = unit_start / compression_ratios[name] / 16000
            # magic numbers: 0.5 seconds of before-padding; 100 Hz sampling rate
            unit_start_ecog = int((0.5 + unit_start_secs) * 100)

            embedding_data[:, unit_start_ecog] = model_embeddings[frame_start + unit_end]

        out_i["model_embedding"] = embedding_data

    return out
    


def prepare_xy(config, data_spec) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int]]]:
    data_dir = Path(config.corpus.paths.data_path) / data_spec.subject / config.corpus.name / "block_z"
    outfile = list(data_dir.glob(f"{data_spec.subject}_{data_spec.block}_{config.corpus.paths.out_file_glob}"))
    assert len(outfile) == 1

    cout = loadmat(outfile[0], simplify_cells=True)
    out = cout["out"]

    # add sentence details to out
    out = timit_encoding.add_details_to_out(out, config.corpus.sentdetV,
                                            config.corpus.paths.info_path,
                                            config.corpus.name,
                                            data_spec.subject)
    feature_sets = config.feature_sets.baseline_features[:]

    # add model embeddings to out
    if getattr(config.feature_sets, "model_features", None):
        out = load_and_align_model_embeddings(config, data_spec, out)
        feature_sets.append("model_embedding")

    X, Y, feature_names, feature_shapes = timit_encoding.prepare_strf_xy(
        out, feature_sets, data_spec.subject)
    
    return X, Y, feature_names, feature_shapes


def get_electrode_df(config, subject):
    electrode_path = Path(config.corpus.paths.data_path) / subject / "BilingVowel" / "imaging" / "elecs" / "TDT_elecs_all.mat"
    elecs = loadmat(electrode_path, simplify_cells=True)["anatomy"]
    ret = pd.DataFrame(elecs, columns=["label", "long_name", "type", "roi"]).set_index("label")
    return ret


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

    all_xy = [prepare_xy(config, data_spec) for data_spec in tqdm(config.data, desc="Prepare design matrix")]
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