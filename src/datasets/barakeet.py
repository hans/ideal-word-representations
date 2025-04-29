from collections import defaultdict
from pathlib import Path
import re

import datasets
import textgrid

_CITATION = """Your dataset citation here."""

class BarakeetDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    SAMPLE_RATE = 16000

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
                    "word_raw_detail": datasets.Sequence(
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
            task_templates=[
                datasets.tasks.AutomaticSpeechRecognition(
                    audio_column="audio", transcription_column="text"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        data_dir = Path(dl_manager.manual_dir)
        wav_files = sorted(data_dir.glob("*.wav"))
        textgrid_files = {tg.stem: tg for tg in data_dir.glob("*.TextGrid")}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"wav_files": wav_files, "textgrid_files": textgrid_files},
            )
        ]

    def _generate_examples(self, wav_files, textgrid_files):
        # Get continuum for each lexical item
        continua = defaultdict(list)
        # extract lexical item and spectrum step from wav file name
        wav_file_re = r"\d+_([^_]+)_.+_(\d+)"
        for wav_path in wav_files:
            lexical_item, grade = re.findall(wav_file_re, wav_path.stem)[0]
            continua[lexical_item].append(int(grade.lstrip("0")))

        for wav_path in wav_files:
            file_id = wav_path.stem
            textgrid_path = textgrid_files.get(file_id)
            
            if not textgrid_path:
                continue  # Skip files without a corresponding TextGrid

            lexical_item, grade = re.findall(r"\d+_([^_]+)_.+_(\d+)", file_id)[0]
            grade = int(grade.lstrip("0"))
            grade_adjusted = grade - min(continua[lexical_item]) + 1
            
            tg = textgrid.TextGrid()
            tg.read(str(textgrid_path))
            
            words, words_raw, phonemes = None, None, []
            transcript = ""

            for tier in tg:
                if "word" in tier.name.lower():
                    words_raw = [
                        {"start": int(i.minTime * self.SAMPLE_RATE),
                         "stop": int(i.maxTime * self.SAMPLE_RATE),
                         "utterance": i.mark}
                        for i in tier if i.mark.strip()
                    ]

                    words = [
                        {"start": int(i.minTime * self.SAMPLE_RATE),
                         "stop": int(i.maxTime * self.SAMPLE_RATE),
                         "utterance": f"{lexical_item}-{grade_adjusted}"}
                        for i in tier if i.mark.strip()
                    ]
                elif "phone" in tier.name.lower():
                    phonemes = [
                        {"start": int(i.minTime * self.SAMPLE_RATE),
                         "stop": int(i.maxTime * self.SAMPLE_RATE),
                         "utterance": i.mark}
                        for i in tier if i.mark.strip()
                    ]
                elif "transcript" in tier.name.lower():
                    transcript = " ".join(i.mark.strip() for i in tier if i.mark.strip())

            if words is None:
                # induce single word from the phoneme span. later annotations didn't explicitly
                # include the word
                word_start = min(p["start"] for p in phonemes)
                word_stop = max(p["stop"] for p in phonemes)
                words = [
                    {
                        "start": word_start,
                        "stop": word_stop,
                        "utterance": f"{lexical_item}-{grade_adjusted}",
                    }
                ]
                words_raw = [
                    {
                        "start": word_start,
                        "stop": word_stop,
                        "utterance": f"{lexical_item}",
                    }
                ]

            yield file_id, {
                "file": str(wav_path),
                "audio": str(wav_path),
                "text": transcript,
                "id": file_id,
                "word_raw_detail": words_raw,
                "word_detail": words,
                "phonetic_detail": phonemes,
            }
