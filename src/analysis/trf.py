from mne.decoding import ReceptiveField
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def trf_to_df(trf: ReceptiveField, output_names) -> pd.DataFrame:
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

    return pd.DataFrame(trf_df)


def estimate_trf_cv(X, y, output_names, n_splits=5, **kwargs):
    # K-fold estimation over contiguous sections of the data
    kf = KFold(n_splits=n_splits, shuffle=False)

    trf = ReceptiveField(**kwargs)
    trf_df = []

    for train_idx, test_idx in kf.split(X):
        trf.fit(X[train_idx], y[train_idx])
        trf_df.append(trf_to_df(trf, output_names))

    return pd.concat(trf_df, names=["fold"], keys=list(range(n_splits))).reset_index()