# MLS (Multilingual LibriSpeech) French builder, modeled on
# src/datasets/huggingface_librispeech.py.
#
# Joins the MLS distribution layout (per-split transcripts.txt + audio/{spk}/{book}/*.wav)
# with Montreal Forced Aligner TextGrid output to yield items in the same schema
# as the LibriSpeech builder (audio, text, speaker_id, word_detail, phonetic_detail).
#
# All word and phone strings are NFC-normalized so combining-diacritic IPA forms
# (e.g. "ɛ" + combining tilde vs precomposed "ɛ̃") compare equal downstream.

import os
import re
import unicodedata
from pathlib import Path

import datasets
import textgrid


_CITATION = """\
@inproceedings{Pratap2020MLSAL,
  title={MLS: A Large-Scale Multilingual Dataset for Speech Research},
  author={Vineel Pratap and Qiantong Xu and Anuroop Sriram and Gabriel Synnaeve and Ronan Collobert},
  booktitle={Interspeech},
  year={2020}
}
"""

_DESCRIPTION = """\
Multilingual LibriSpeech (MLS) French split. Read speech derived from LibriVox
audiobooks at 16 kHz, segmented into ~10s utterances. Phoneme/word alignments
are not distributed with MLS; we expect them as MFA TextGrid output joined in
at builder time.
"""


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


class MLSFrenchConfig(datasets.BuilderConfig):
    """BuilderConfig for MLS French."""

    def __init__(self, alignment_dir, **kwargs):
        """
        Args:
          alignment_dir: directory containing MFA TextGrid output, with one
            `.TextGrid` per utterance. The builder accepts either a flat layout
            (any `*.TextGrid` under the directory) or the `<spk>/<book>/<id>.TextGrid`
            tree MFA produces by default.
        """
        super().__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.alignment_dir = alignment_dir
        self.audio_sample_rate = 16000


class MLSFrenchASR(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 256
    BUILDER_CONFIG_CLASS = MLSFrenchConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "file": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=None, mono=False),
                "text": datasets.Value("string"),
                "speaker_id": datasets.Value("string"),
                "chapter_id": datasets.Value("string"),
                "id": datasets.Value("string"),

                "word_detail": datasets.Sequence({
                    "start": datasets.Value("int64"),
                    "stop": datasets.Value("int64"),
                    "utterance": datasets.Value("string"),
                }),
                "phonetic_detail": datasets.Sequence({
                    "start": datasets.Value("int64"),
                    "stop": datasets.Value("int64"),
                    "utterance": datasets.Value("string"),
                }),
            }),
            supervised_keys=("file", "text"),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"{data_dir} does not exist. Pass an MLS split directory "
                f"(containing transcripts.txt and audio/) via `data_dir=`.")

        split_name = re.sub(r"[^\w]+", ".", os.path.basename(data_dir))
        # rough count for the SplitInfo; the generator skips utts without alignments
        num_examples = sum(1 for _ in open(Path(data_dir) / "transcripts.txt"))

        gen = datasets.SplitGenerator(
            name=split_name,
            gen_kwargs={"data_dir": data_dir})
        gen.split_info.num_examples = num_examples
        return [gen]

    def _index_alignments(self):
        """Map utterance_id -> Path of its TextGrid file."""
        align_root = Path(self.config.alignment_dir)
        index = {}
        for tg_path in align_root.rglob("*.TextGrid"):
            index[tg_path.stem] = tg_path
        return index

    def _parse_textgrid(self, tg_path: Path) -> dict:
        """Extract word and phone tiers from an MFA TextGrid, in sample frames."""
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        rate = self.config.audio_sample_rate

        out = {}
        for tier_name, target in [("words", "word_detail"),
                                  ("phones", "phonetic_detail")]:
            tier = tg.getFirst(tier_name)
            starts, stops, utts = [], [], []
            for interval in tier:
                starts.append(int(interval.minTime * rate))
                stops.append(int(interval.maxTime * rate))
                # MFA emits empty marks for silence intervals; keep them so frame
                # coverage is faithful, but NFC-normalize the non-empty ones.
                mark = interval.mark or ""
                utts.append(_nfc(mark))
            out[target] = {"start": starts, "stop": stops, "utterance": utts}
        return out

    def _generate_examples(self, data_dir):
        data_dir = Path(data_dir)
        audio_root = data_dir / "audio"

        # transcripts.txt: <id>\t<orthographic text> per line
        transcripts = {}
        with (data_dir / "transcripts.txt").open() as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                utt_id, text = line.split("\t", 1)
                transcripts[utt_id] = text

        alignments = self._index_alignments()

        for utt_id, text in transcripts.items():
            if utt_id not in alignments:
                continue

            # ID format: <spk>_<book>_<seg> -> audio/<spk>/<book>/<utt_id>.wav
            spk, book, _seg = utt_id.split("_", 2)
            audio_path = audio_root / spk / book / f"{utt_id}.wav"
            if not audio_path.exists():
                # Fall back to flac if the download rule didn't convert.
                flac_path = audio_path.with_suffix(".flac")
                if flac_path.exists():
                    audio_path = flac_path
                else:
                    continue

            tg_info = self._parse_textgrid(alignments[utt_id])

            yield utt_id, {
                "id": utt_id,
                "speaker_id": spk,
                "chapter_id": book,
                "file": str(audio_path),
                "text": _nfc(text),
                "audio": {
                    "path": str(audio_path),
                    "bytes": audio_path.read_bytes(),
                },
                **tg_info,
            }
