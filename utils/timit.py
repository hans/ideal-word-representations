from pathlib import Path
import re

from datasets import Dataset, GeneratorBasedBuilder
import pandas as pd
import soundfile as sf
import torch

from models import Vocabulary


import numpy as np
test_item = {
    'file': '/data/TRAIN/DR4/MMDM0/SI681.WAV',
    'audio': {'path': '/data/TRAIN/DR4/MMDM0/SI681.WAV',
      		  'array': np.array([-0.00048828, -0.00018311, -0.00137329, 0.00079346, 0.00091553,  0.00085449], dtype=np.float32),
      		  'sampling_rate': 16000},
    'text': 'Would such an act of refusal be useful?',
    'phonetic_detail': [{'start': '0', 'stop': '1960', 'utterance': 'h#'},
                        {'start': '1960', 'stop': '2466', 'utterance': 'w'},
                        {'start': '2466', 'stop': '3480', 'utterance': 'ix'},
                        {'start': '3480', 'stop': '4000', 'utterance': 'dcl'},
                        {'start': '4000', 'stop': '5960', 'utterance': 's'},
                        {'start': '5960', 'stop': '7480', 'utterance': 'ah'},
                        {'start': '7480', 'stop': '7880', 'utterance': 'tcl'},
                        {'start': '7880', 'stop': '9400', 'utterance': 'ch'},
                        {'start': '9400', 'stop': '9960', 'utterance': 'ix'},
                        {'start': '9960', 'stop': '10680', 'utterance': 'n'},
                        {'start': '10680', 'stop': '13480', 'utterance': 'ae'},
                        {'start': '13480', 'stop': '15680', 'utterance': 'kcl'},
                        {'start': '15680', 'stop': '15880', 'utterance': 't'},
                        {'start': '15880', 'stop': '16920', 'utterance': 'ix'},
                        {'start': '16920', 'stop': '18297', 'utterance': 'v'},
                        {'start': '18297', 'stop': '18882', 'utterance': 'r'},
                        {'start': '18882', 'stop': '19480', 'utterance': 'ix'},
                        {'start': '19480', 'stop': '21723', 'utterance': 'f'},
                        {'start': '21723', 'stop': '22516', 'utterance': 'y'},
                        {'start': '22516', 'stop': '24040', 'utterance': 'ux'},
                        {'start': '24040', 'stop': '25190', 'utterance': 'zh'},
                        {'start': '25190', 'stop': '27080', 'utterance': 'el'},
                        {'start': '27080', 'stop': '28160', 'utterance': 'bcl'},
                        {'start': '28160', 'stop': '28560', 'utterance': 'b'},
                        {'start': '28560', 'stop': '30120', 'utterance': 'iy'},
                        {'start': '30120', 'stop': '31832', 'utterance': 'y'},
                        {'start': '31832', 'stop': '33240', 'utterance': 'ux'},
                        {'start': '33240', 'stop': '34640', 'utterance': 's'},
                        {'start': '34640', 'stop': '35968', 'utterance': 'f'},
                        {'start': '35968', 'stop': '37720', 'utterance': 'el'},
                        {'start': '37720', 'stop': '39920', 'utterance': 'h#'}],
    'word_detail': [{'start': '1960', 'stop': '4000', 'utterance': 'would'},
                    {'start': '4000', 'stop': '9400', 'utterance': 'such'},
                    {'start': '9400', 'stop': '10680', 'utterance': 'an'},
                    {'start': '10680', 'stop': '15880', 'utterance': 'act'},
                    {'start': '15880', 'stop': '18297', 'utterance': 'of'},
                    {'start': '18297', 'stop': '27080', 'utterance': 'refusal'},
                    {'start': '27080', 'stop': '30120', 'utterance': 'be'},
                    {'start': '30120', 'stop': '37720', 'utterance': 'useful'}],

    'dialect_region': 'DR4',
    'sentence_type': 'SI',
    'speaker_id': 'MMDM0',
    'id': 'SI681'
}

def group_phonetic_detail(item, drop_phones=None, key="phonetic_detail"):
    """
    Group phonetic_detail entries according to the containing word.
    """
    phonetic_detail = item[key]
    word_detail = item["word_detail"]

    word_phonetic_detail = []
    for start, stop in zip(word_detail["start"], word_detail["stop"]):
        word_phonetic_detail.append([])
        for phon_start, phon_stop, phon in zip(phonetic_detail["start"], phonetic_detail["stop"], phonetic_detail["utterance"]):
            if drop_phones is not None and phon in drop_phones:
                continue
            if phon_start >= start and phon_stop <= stop:
                word_phonetic_detail[-1].append({"phone": phon, "start": phon_start, "stop": phon_stop})

    item[f"word_{key}"] = word_phonetic_detail
    return item


class TimitCorpus:
    
    def __init__(self, df_path="timit_merged.csv", corpus_path="data", dtype="float32"):
        super().__init__(name="timit", version="0.0.1")

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
    def speakers(self) -> list[str]:
        return sorted(self.sentences_df.index.get_level_values("speaker").unique())

    @property
    def num_speakers(self):
        return len(self.sentences_df.groupby("speaker"))
    
    def _compute_word_features(self, phon_rows):
            return {
                "onset": phon_rows["onset"].min(),
                "offset": phon_rows["offset"].max(),
                "phones": [{"phone": phon_row.phone,
                            "onset": phon_row.onset,
                            "offset": phon_row.offset}
                           for phon_row in phon_rows.itertuples()]
            }

    def _generate_examples(self):
        all_speakers = self.speakers

        for (dialect, speaker, sentence_idx), audio in self.sounds.items():
            phones = self._df.loc[(dialect, speaker, sentence_idx)]

            yield (dialect, speaker, sentence_idx), {
                "input_values": audio,

                "testing": "herpderp",
                "words": phones.groupby("word_idx").apply(self._compute_word_features).tolist(),

                "speaker": speaker,
                "speaker_idx": all_speakers.index(speaker),
                "dialect": dialect,
                "sentence_idx": sentence_idx,
            }

    def to_dataset(self) -> Dataset:
        return Dataset.from_generator(self._yield_dataset)