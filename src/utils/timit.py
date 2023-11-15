from pathlib import Path
import re

from datasets import load_dataset, load_from_disk
import transformers
import pandas as pd
import soundfile as sf
import torch

from models.transformer import TilingWordFeatureExtractor2


import numpy as np


# TODO cross-check resulting mappings with cmudict
TIMIT_MAPPING = {
    'ax': 'AH',
    'ax-h': 'AH',
    'axr': 'ER',
    'dx': 'T',
    'el': ['AH', 'L'],
    'em': ['AH', 'M'],
    'en': ['AH', 'N'],
    'eng': ['IH', 'NG'],
    'hv': 'HH',
    'ix': 'IH',
    'nx': ['N', 'T'],
    'q': 'T',
    'pau': '[SIL]',
    'epi': '[SIL]',
    'ux': 'UW'
}


TIMIT_IGNORE = ["h#"]
TIMIT_JOIN_CLOSURES = ['bcl', 'dcl', 'gcl', 'kcl', 'pcl', 'tcl']


def map_timit_phone_to_cmudict_phoneme(phone, phone_vocab):
    if phone in TIMIT_IGNORE:
        return []
    elif phone in TIMIT_MAPPING:
        ret = TIMIT_MAPPING[phone]
        if isinstance(ret, str):
            ret = [ret]
        return ret
    elif not phone.upper() in phone_vocab:
        raise ValueError(f"Invalid phone {phone.upper()}")
    return [phone.upper()]


def map_timit_to_cmudict(timit_item, phone_vocab):
    phonemes = []

    i = 0
    phonetic_detail = timit_item["phonetic_detail"]
    num_phones = len(phonetic_detail["start"])
    while i < num_phones:
        phone = phonetic_detail["utterance"][i]
        start = phonetic_detail["start"][i]
        stop = phonetic_detail["stop"][i]

        if phone in TIMIT_IGNORE:
            i += 1
        elif phone in TIMIT_JOIN_CLOSURES:
            release_phone = phone[:-len("cl")]
            if phonetic_detail["utterance"][i + 1] == release_phone:
                phoneme_start = start
                phoneme_end = phonetic_detail["stop"][i + 1]
                phoneme_label = map_timit_phone_to_cmudict_phoneme(release_phone, phone_vocab)

                for phoneme in phoneme_label:
                    phonemes.append((phoneme_start, phoneme_end, phoneme))
                i += 2
            else:
                phoneme_label = map_timit_phone_to_cmudict_phoneme(release_phone, phone_vocab)
                for phoneme in phoneme_label:
                    phonemes.append((start, stop, phoneme))
                i += 1
        else:
            for phoneme in map_timit_phone_to_cmudict_phoneme(phone, phone_vocab):
                phonemes.append((start, stop, phoneme))
            i += 1

    return phonemes


def add_phonemic_detail(item, phone_vocab):
    phonemes = map_timit_to_cmudict(item, phone_vocab)

    starts, stops, utterances = zip(*phonemes)
    item["phonemic_detail"] = {
        "start": starts,
        "stop": stops,
        "utterance": utterances
    }

    return item


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


def prepare_timit_corpus(data_dir,
                         processor: transformers.Wav2Vec2Processor,
                         drop_phones=None):
    """
    Load and prepare TIMIT corpus for training.
    """

    corpus = load_dataset("timit_asr", data_dir=data_dir)

    phone_vocab = set(processor.tokenizer.get_vocab().keys())
    # Sanity check: all TIMIT mapped to CMU targets should be in the vocab
    for src, tgt in TIMIT_MAPPING.items():
        if isinstance(tgt, str):
            tgt = [tgt]
        for t in tgt:
            assert t in phone_vocab

    corpus = corpus.map(add_phonemic_detail, batched=False,
                        fn_kwargs=dict(phone_vocab=phone_vocab))

    # Compute phonetic and phonemic details grouped by word span
    corpus = corpus.map(group_phonetic_detail, batched=False,
                        fn_kwargs=dict(drop_phones=drop_phones))
    corpus = corpus.map(group_phonetic_detail, batched=False,
                        fn_kwargs=dict(key="phonemic_detail"))
    
    def prepare_audio(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        return batch
    corpus = corpus.map(prepare_audio)

    twfe = TilingWordFeatureExtractor2(processor.tokenizer, item_key="word_phonemic_detail")
    def add_features(example):
        example["phone_targets"] = twfe(example)
        return example
    corpus = corpus.map(add_features, load_from_cache_file=False)
    
    return corpus


def load_or_prepare_timit_corpus(processed_data_dir,
                                 raw_data_dir,
                                 processor: transformers.Wav2Vec2Processor,
                                 drop_phones=None):
    """
    Load preprocessed TIMIT corpus if it exists, or compute and save to
    the preprocessed directory.
    """
    if Path(processed_data_dir).exists():
        corpus = load_from_disk(processed_data_dir)
    else:
        corpus = prepare_timit_corpus(raw_data_dir, processor, drop_phones)
        corpus.save_to_disk(processed_data_dir)
    return corpus