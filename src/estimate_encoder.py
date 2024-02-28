from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import KFold, GridSearchCV
from tqdm.auto import tqdm

from src.encoding.ecog import timit as timit_encoding


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

    baseline_feature_sets = config.feature_sets.baseline_features
    X, Y, feature_names, feature_shapes = timit_encoding.prepare_strf_xy(
        out, baseline_feature_sets, data_spec.subject)
    
    # TODO extract feature representations from model embeddings if provided,
    # with some aggregation strategy
    # TODO align with TIMIT Xy representation and do sanity checks

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