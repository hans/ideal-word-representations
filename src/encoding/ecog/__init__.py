from pathlib import Path

from mne.decoding import ReceptiveField
import pandas as pd
from scipy.io import loadmat


def get_electrode_df(config, subject: str):
    electrode_path = Path(config.corpus.paths.data_path) / subject / "BilingVowel" / "imaging" / "elecs" / "TDT_elecs_all.mat"
    elecs = loadmat(electrode_path, simplify_cells=True)["anatomy"]
    ret = pd.DataFrame(elecs, columns=["label", "long_name", "type", "roi"]).set_index("label")
    return ret


class TemporalReceptiveField(ReceptiveField):

    def score(self, X, y):
        # parent class returns one score per output
        scores = super().score(X, y)
        return scores.mean()
    
    def score_multidimensional(self, X, y):
        return super().score(X, y)