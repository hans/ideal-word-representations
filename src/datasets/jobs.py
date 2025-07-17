from collections import defaultdict
import json
from pathlib import Path
import re

import datasets
import numpy as np
import soundfile
import textgrid

_CITATION = """Your dataset citation here."""

class JobsDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    SAMPLE_RATE = 16000

    # maximum stimulus length
    CHUNK_SIZE = 20
    # stride between chunks
    CHUNK_STRIDE = 4

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=None, mono=False),
                    "text": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "word_detail": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "stop": datasets.Value("int64"),
                            "utterance": datasets.Value("string"),
                        }
                    ),
                    "phonetic_detail": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "stop": datasets.Value("int64"),
                            "utterance": datasets.Value("string"),
                        }
                    ),
                }
            ),
            supervised_keys=("file", "text"),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = Path(dl_manager.manual_dir)
        wav_files = sorted(data_dir.glob("*.wav"))
        json_files = {tg.stem: tg for tg in data_dir.glob("*.json")}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"wav_files": wav_files, "json_files": json_files},
            )
        ]

    def _generate_examples(self, wav_files, json_files):
        for wav_path in wav_files:
            file_id = wav_path.stem
            json_path = json_files[file_id]
            
            if not json_path.exists():
                continue  # Skip files without a corresponding JSON

            with open(json_path) as f:
                json_data = json.load(f)

            assert len(json_data["Word"]) == len(json_data["Start"])
            assert len(json_data["Word"]) == len(json_data["End"])

            # we will generate multiple overlapping chunked items
            # from this file
            stimulus, fs = soundfile.read(wav_path)
            assert fs == self.SAMPLE_RATE
            stimulus_length = len(stimulus) / fs

            for i, start in enumerate(np.arange(0, stimulus_length, self.CHUNK_STRIDE)):
                start_sample = int(start * self.SAMPLE_RATE)
                end = start + self.CHUNK_SIZE
                end_sample = int(end * self.SAMPLE_RATE)

                chunk = stimulus[start_sample:end_sample]
                if len(chunk) < self.SAMPLE_RATE * self.CHUNK_SIZE:
                    # right-pad with zeros
                    chunk = np.pad(
                        chunk, (0, self.SAMPLE_RATE * self.CHUNK_SIZE - len(chunk))
                    )

                words = []
                for word_start, word_end, word in zip(
                    json_data["Start"], json_data["End"], json_data["Word"]
                ):
                    word_start_sample = int(word_start * self.SAMPLE_RATE)
                    word_end_sample = int(word_end * self.SAMPLE_RATE)

                    # check if the word is within the chunk
                    if word_start_sample < start_sample or word_start_sample > end_sample:
                        continue
                    word_end_sample = min(word_end_sample, end_sample)

                    words.append({
                        "start": word_start_sample - start_sample,
                        "stop": word_end_sample - start_sample,
                        "utterance": word,
                    })

                # Create dummy phonemes, one per word
                phonemes = [
                    {"start": word["start"] + 1, "stop": word["stop"] - 1, "utterance": "ɑ"}
                    for word in words
                ]

                yield f"{file_id}_{i}", {
                    "file": str(wav_path),
                    "audio": {"array": chunk, "sampling_rate": self.SAMPLE_RATE},
                    # "text": transcript,
                    "id": f"{file_id}_{i}",
                    "word_detail": words,
                    "phonetic_detail": phonemes,
                }