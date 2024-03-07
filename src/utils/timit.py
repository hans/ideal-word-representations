import itertools
import logging
from pathlib import Path
import re

from datasets import load_dataset, load_from_disk, DatasetDict
import transformers
import pandas as pd
import soundfile as sf
import torch

from src.models.transformer import TilingWordFeatureExtractor2
from src.utils import syllabifier


import numpy as np

L = logging.getLogger(__name__)


# TODO cross-check resulting mappings with cmudict
TIMIT_MAPPING = {
    'ax': 'AH',
    'ax-h': 'AH',
    'axr': 'ER',

    # tap/flap
    'dx': 'T',

    # lateral/nasal release
    # TODO we should retain this detail for the syllabification somehow
    'el': "L",
    'em': "M",
    'en': "N",
    'eng': "NG",
    'hv': 'HH',
    'ix': 'IH',

    # nasal flap
    'nx': "N",
    'pau': '[SIL]',
    'epi': '[SIL]',
    'ux': 'UW'
}


TIMIT_IGNORE = ["h#"]

# If we see `key` followed by a phone matching regex `value`, join these into a single
# phone with label equivalent to the label of the second phone
TIMIT_JOINS = {
    # join glottal stop with following vowel
    "q": "^(iy|ih|eh|ey|ae|aa|aw|ay|ah|ao|oy|ow|uh|uw|ux|er|ax|ix|axr|ax-h)$",

    # these can precede both stops and affricates
    "dcl": "^jh$",
    "tcl": "^ch$",

    # join plosive closure with release
    **{f"{k}cl": f"^{k}$" for k in ['b', 'g', 'k', 'p']},
}


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

    phonetic_detail = timit_item["phonetic_detail"]
    num_phones = len(phonetic_detail["start"])

    def handle_join_exception(join_key, position) -> tuple[int, list[tuple[int, int, str]]]:
        """
        This function is called when we fail to find a partner for a join key -- for example,
        a closure `dcl` without an accompanying release `d`.
        
        Returns:
            increment: the delta that should be applied to TIMIT phone cursor
            phonemes: a list of tuples (start, stop, phoneme) that should be appended to the phonemes list
        """
        start = phonetic_detail["start"][position]
        stop = phonetic_detail["stop"][position]
        if join_key.endswith("cl"):
            # unreleased stop -- map to full phoneme
            phonemes = [(start, stop, phoneme)
                        for phoneme in map_timit_phone_to_cmudict_phoneme(join_key[:-len("cl")], phone_vocab)]
            return -1, phonemes
        elif join_key == "q":
            # treat this as a glottalized /t/
            # this is often wrong in the annotation, i.e. we'll see a glottal stop preceding a vowel
            # actually outside of the span of the word containing that vowel.
            # this code will mark that glottal stop ars a /t/, but that marking
            # will be disposed later in the pipeline since it's outside of the span
            # of any word in the sentence
            L.info(f"Interpreting glottal stop as /t/ in item {timit_item['text']}: "
                   f"{' '.join(timit_item['phonetic_detail']['utterance'][max(0, position - 2):position+3])}")
            
            phonemes = [(start, stop, phoneme)
                        for phoneme in map_timit_phone_to_cmudict_phoneme("t", phone_vocab)]
            return -1, phonemes
        else:
            L.error(timit_item['text'])
            L.error(timit_item['phonetic_detail']['utterance'][max(0, position - 2):position+3])
            L.error(timit_item['phonetic_detail']['utterance'])
            raise ValueError("error processing join candidate")

    # Process backwards to handle join candidates easily
    i = num_phones - 1
    while i >= 0:
        phone = phonetic_detail["utterance"][i]
        start = phonetic_detail["start"][i]
        stop = phonetic_detail["stop"][i]

        prev_phone = phonetic_detail["utterance"][i - 1] if i > 0 else None

        if phone in TIMIT_IGNORE:
            i -= 1
        elif phone in TIMIT_JOINS:
            # first element of a join found but we hadn't seen the second element previously
            increment, join_phonemes = handle_join_exception(phone, i)
            phonemes.extend(join_phonemes)
            i += increment
        elif prev_phone in TIMIT_JOINS:
            # is the previous phone annotated as being in the same word as the current phone?
            prev_phone_start = phonetic_detail["start"][i - 1]
            cur_word_start = max(word_start for word_start in timit_item["word_detail"]["start"]
                                 if word_start <= start)

            if prev_phone_start >= cur_word_start and re.match(TIMIT_JOINS[prev_phone], phone):
                phoneme_start = phonetic_detail["start"][i - 1]
                phoneme_end = stop
                phoneme_label = map_timit_phone_to_cmudict_phoneme(phone, phone_vocab)
                for phoneme in phoneme_label:
                    phonemes.append((phoneme_start, phoneme_end, phoneme))

                i -= 2
            else:
                for phoneme in map_timit_phone_to_cmudict_phoneme(phone, phone_vocab):
                    phonemes.append((start, stop, phoneme))
                i -= 1
        else:
            for phoneme in map_timit_phone_to_cmudict_phoneme(phone, phone_vocab):
                phonemes.append((start, stop, phoneme))
            i -= 1

    return phonemes[::-1]


def add_phonemic_detail(item, phone_vocab):
    phonemes = map_timit_to_cmudict(item, phone_vocab)

    starts, stops, utterances = zip(*phonemes)
    item["phonemic_detail"] = {
        "start": starts,
        "stop": stops,
        "utterance": utterances
    }

    return item


def group_phonetic_detail(item, idx, drop_phones=None, key="phonetic_detail"):
    """
    Group phonetic_detail entries according to the containing word.
    """
    phonetic_detail = item[key]
    word_detail = item["word_detail"]

    # Assure that each phone gets mapped to exactly one word. We'll arbitrarily map to the
    # first word that contains the phone; this seems to most frequently match TIMIT annotation standards
    phone_mask = np.zeros(len(phonetic_detail["start"]), dtype=bool)
    # Note that we also assign phonemes which span words to the leftmost word, consistent
    # with this strategy

    word_phonetic_detail = []
    for start, stop, word in zip(word_detail["start"], word_detail["stop"], word_detail["utterance"]):
        word_phonetic_detail.append([])
        for j, (phon_start, phon_stop, phon) in enumerate(zip(phonetic_detail["start"], phonetic_detail["stop"], phonetic_detail["utterance"])):
            if phone_mask[j]:
                continue
            elif drop_phones is not None and phon in drop_phones:
                phone_mask[j] = True
                continue
            
            # if the phoneme has start in this word, assign it to this word
            if phon_start >= start and phon_start < stop:
                phone_mask[j] = True
                word_phonetic_detail[-1].append({"phone": phon, "start": phon_start, "stop": phon_stop})

    for unused_phone in np.flatnonzero(~phone_mask):
        preceding_phones = " ".join(phonetic_detail["utterance"][max(0, unused_phone - 3):unused_phone])
        following_phones = " ".join(phonetic_detail["utterance"][unused_phone + 1:min(len(phonetic_detail["utterance"]), unused_phone + 4)])
        unused_phone_str = phonetic_detail["utterance"][unused_phone]
        L.warning(f"Unused phone {unused_phone_str} in item {idx} ({item['text']}) (preceding: {preceding_phones}, following: {following_phones})")

    item[f"word_{key}"] = word_phonetic_detail
    return item


def add_syllabic_detail(item):
    word_syllables = []
    for word in item["word_phonemic_detail"]:
        phones = [ph["phone"] for ph in word if ph["phone"] != "[SIL]"]
        if len(phones) > 0:
            syllables = syllabifier.syllabify(syllabifier.English, phones)

            assert phones == list(itertools.chain.from_iterable(
                [tuple(onset) + tuple(nucleus) + tuple(coda) for stress, onset, nucleus, coda in syllables]))
            # print(syllables)
            # word["syllables"] = syllables

            phoneme_idx, syllable_idx = 0, 0
            syllable_dicts = []
            for stress, onset, nucleus, coda in syllables:
                syllable_phones = tuple(onset + nucleus + coda)
                syllable_dict = {
                    "phones": syllable_phones,
                    "idx": syllable_idx,
                    "phoneme_start_idx": phoneme_idx,
                    "phoneme_end_idx": phoneme_idx + len(syllable_phones), # exclusive
                    "stress": stress,

                    "start": word[phoneme_idx]["start"],
                    "stop": word[phoneme_idx + len(syllable_phones) - 1]["stop"],
                }

                # Add cross-reference data in word_phonemic_detail
                for j, ph in enumerate(syllable_phones):
                    word[phoneme_idx + j]["syllable_idx"] = syllable_idx
                    word[phoneme_idx + j]["idx_in_syllable"] = j
                    word[phoneme_idx + j]["syllable_phones"] = tuple(syllable_phones)
                    word[phoneme_idx + j]["stress"] = stress
                    word[phoneme_idx + j]["syllable_start"] = syllable_dict["start"]
                    word[phoneme_idx + j]["syllable_stop"] = syllable_dict["stop"]

                syllable_dicts.append(syllable_dict)
                phoneme_idx += len(syllable_phones)
                syllable_idx += 1
        else:
            syllable_dicts = []

        word_syllables.append(syllable_dicts)
    
    item["word_syllable_detail"] = word_syllables
    return item


def check_item(item, idx, drop_phones=None):
    grouped_phonemic_detail = item["word_phonemic_detail"]
    grouped_syllable_detail = item["word_syllable_detail"]
    assert len(grouped_phonemic_detail) == len(item["word_detail"]["utterance"])
    assert len(grouped_syllable_detail) == len(item["word_detail"]["utterance"])

    all_phonemes = [phon["phone"] for word in grouped_phonemic_detail for phon in word]
    all_phonemes_syll = [phone for word in item["word_syllable_detail"] for syllable in word for phone in syllable["phones"]]
    assert len(all_phonemes) == len(all_phonemes_syll)
    assert all_phonemes == all_phonemes_syll, "phonemic detail does not match phonemes within syllable detail"

    # NB we do expect a mismatch here since some phonemes in the flat representation
    # won't appear in the word grouped representation, if they are outside the span of a word
    # all_phonemes_flat = [ph for ph in item["phonemic_detail"]["utterance"] if ph not in (drop_phones or [])]
    # assert all_phonemes == all_phonemes_flat, \
    #     f"grouped phonemic detail does not match non-grouped phonemic detail in item {idx}:" \
    #     f"\n{item['text']}\n{all_phonemes}\n{all_phonemes_flat}"


def prepare_timit_corpus(data_dir,
                         processor: transformers.Wav2Vec2Processor,
                         add_phoneme_targets=False,
                         drop_phones=("[SIL]",)):
    """
    Load and prepare TIMIT corpus for training.
    """

    if (Path(data_dir) / "dataset_dict.json").exists():
        corpus = load_dataset(data_dir)
    else:
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
    corpus = corpus.map(group_phonetic_detail, batched=False, with_indices=True,
                        fn_kwargs=dict(drop_phones=drop_phones))
    corpus = corpus.map(group_phonetic_detail, batched=False, with_indices=True,
                        fn_kwargs=dict(key="phonemic_detail",
                                       drop_phones=drop_phones))
    
    # Add syllabic detail
    corpus = corpus.map(add_syllabic_detail, batched=False)

    # Run sanity checks on updated annotations
    corpus.map(check_item, batched=False, with_indices=True,
               fn_kwargs=dict(drop_phones=drop_phones))
    
    def prepare_audio(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        return batch
    corpus = corpus.map(prepare_audio)
    
    return corpus


def load_or_prepare_timit_corpus(processed_data_dir,
                                 raw_data_dir,
                                 processor: transformers.Wav2Vec2Processor,
                                 drop_phones=("[SIL]",)) -> DatasetDict:
    """
    Load preprocessed TIMIT corpus if it exists, or compute and save to
    the preprocessed directory.
    """
    if Path(processed_data_dir).exists():
        corpus = load_from_disk(processed_data_dir)
    else:
        corpus = prepare_timit_corpus(raw_data_dir, processor, drop_phones=drop_phones)
        corpus.save_to_disk(processed_data_dir)
    return corpus