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
    num_folds = len(coefs)
    num_electrodes, num_features, num_lags = coefs[0].shape

    # Precompute arrays for fold, feature, electrode, lag, and time
    folds = np.repeat(np.arange(num_folds), num_electrodes * num_features * num_lags)
    electrodes = np.tile(np.repeat(np.arange(num_electrodes), num_features * num_lags), num_folds)
    features = np.tile(np.repeat(np.arange(num_features), num_lags), num_folds * num_electrodes)
    lags = np.tile(np.arange(num_lags), num_folds * num_electrodes * num_features)
    times = lags / sfreq

    # Flatten all coefficient arrays
    coef_flat = np.concatenate([fold_coefs.flatten() for fold_coefs in coefs])

    # Create the DataFrame at once
    trf_df = pd.DataFrame({
        "fold": folds,
        "feature_idx": features,
        "input_dim": features,
        "output_dim": electrodes,
        "lag": lags,
        "time": times,
        "coef": coef_flat
    })
    trf_df["feature"] = trf_df.feature_idx.map(dict(enumerate(feature_names)))
    trf_df["output_name"] = trf_df.output_dim.map(dict(enumerate(output_names)))

    return trf_df


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