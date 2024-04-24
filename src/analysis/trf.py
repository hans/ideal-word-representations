from dataclasses import dataclass
from typing import Optional, Union

from mne.decoding import ReceptiveField
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold


def trf_to_df(trf: ReceptiveField, output_names, return_pl=False) -> Union[pd.DataFrame, pl.DataFrame]:
    trf_df = []
    for input_dim, name in enumerate(trf.feature_names):
        for output_dim, output_name in enumerate(output_names):
            coefs = trf.coef_[output_dim, input_dim]
            for delay, coef in zip(trf.delays_, coefs):
                trf_df.append({
                    "feature": name,
                    "output_name": output_name,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "lag": delay,
                    "time": delay / trf.sfreq,
                    "coef": coef,
                })

    return pd.DataFrame(trf_df) if not return_pl else pl.DataFrame(trf_df)


def coefs_to_df(coefs, feature_names, output_names, sfreq) -> pd.DataFrame:
    """
    Convert the saved coefs array from `estimate_encoder` to a TRF coefficient dataframe.
    """
    trf_df = []

    num_folds = len(coefs)
    num_electrodes, num_features, num_lags = coefs[0].shape
    assert num_electrodes == len(output_names)
    assert num_features == len(feature_names)

    for fold, fold_coefs in enumerate(coefs):
        for electrode_idx, feature_idx, lag_idx in np.ndindex(fold_coefs.shape):
            trf_df.append({
                "fold": fold,
                "feature": feature_names[feature_idx],
                "output_name": output_names[electrode_idx],
                "input_dim": feature_idx,
                "output_dim": electrode_idx,
                "lag": lag_idx,
                "time": lag_idx / sfreq,
                "coef": fold_coefs[electrode_idx, feature_idx, lag_idx],
            })

    return pd.DataFrame(trf_df)


@dataclass
class CVTRFResult:
    coefs: pd.DataFrame
    scores: pd.DataFrame
    predictions: Optional[np.ndarray] = None


def estimate_trf_cv(X, y, output_names, n_splits=5,
                    return_predictions=False, **kwargs) -> CVTRFResult:
    # K-fold estimation over contiguous sections of the data
    kf = KFold(n_splits=n_splits, shuffle=False)

    trf = ReceptiveField(**kwargs)
    coefs, scores, predictions = [], [], []

    for train_idx, test_idx in kf.split(X):
        trf.fit(X[train_idx], y[train_idx])
        coefs.append(trf_to_df(trf, output_names))
        scores.append(trf.score(X[test_idx], y[test_idx]))

        if return_predictions:
            predictions.append(trf.predict(X[test_idx]))

    coef_df = pd.concat(coefs, names=["fold"], keys=list(range(n_splits))).reset_index()
    scores_df = pd.DataFrame(np.array(scores), columns=output_names,
                             index=pd.Index(range(n_splits), name="fold"))

    return CVTRFResult(coefs=coef_df, scores=scores_df,
                       predictions=np.concatenate(predictions) if return_predictions else None)