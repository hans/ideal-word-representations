from pathlib import Path
import re

import pandas as pd
import soundfile as sf
import torch

from models import Vocabulary


class TimitCorpus:
    
    def __init__(self, df_path="timit_merged.csv", corpus_path="data", dtype="float32"):
        self._df = pd.read_csv(df_path, index_col=["dialect", "speaker", "sentence_idx", "word_idx", "phone_idx"])

        self.phone_vocab = Vocabulary("phones")
        for _, row in self._df.iterrows():
            self.phone_vocab.add_token(row.phone)

        self.word_df: pd.DataFrame = self._df.groupby(["speaker", "sentence_idx", "word_idx"]) \
            .apply(lambda xs: xs.iloc[0].word)
        self.word_df.index = self.word_df.index.set_levels(self.word_df.index.levels[2].astype(int), level=2)

        self.sentences_df = self.word_df.groupby(["speaker", "sentence_idx"]) \
            .apply(lambda xs: xs.str.cat(sep=" "))

        self._load_sounds(corpus_path, dtype=dtype)

    def _load_sounds(self, corpus_path, dtype, sample_rate=16000):
        files = list(Path(f"{corpus_path}/TRAIN").glob("*/*/*.WAV"))

        # tuples (dialect, speaker, sentence_idx)
        labels = [re.findall(r"([^/]+)/([^/]+)/([^/]+)\.WAV", str(file))[0] for file in files]

        sounds = []
        for file in files:
            sound_i, sfreq_i = sf.read(file, dtype="float32")
            assert sfreq_i == sample_rate, "Mismatched sample rate"
            sounds.append(sound_i)
        sounds = [torch.from_numpy(sound) for sound in sounds]

        self.sounds = dict(zip(labels, sounds))

    @property
    def num_word_instances(self):
        return len(self.word_df)
    
    @property
    def num_word_tokens(self):
        return len(self.word_df.groupby(["sentence_idx", "word_idx"]))
    
    @property
    def num_phone_types(self):
        return len(self.phone_vocab)

    @property
    def num_speakers(self):
        return len(self.sentences_df.groupby("speaker"))