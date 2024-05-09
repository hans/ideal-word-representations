import itertools
import logging
from pathlib import Path
import re

from datasets import load_dataset, load_from_disk, DatasetDict
import transformers
import pandas as pd
import soundfile as sf
import torch

from src.utils import syllabifier


import numpy as np

L = logging.getLogger(__name__)


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
    "dcl": "^(d|jh)$",
    "tcl": "^(t|ch)$",

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

        if len(word_phonetic_detail[-1]) == 0:
            preceding_word_phones = " ".join(phone["phone"] for phone in word_phonetic_detail[-2]) if len(word_phonetic_detail) > 1 else ""
            L.warning(f"No phones found for word {word} in item {idx} ({item['text']}) (preceding word: {preceding_word_phones})")

    for unused_phone in np.flatnonzero(~phone_mask):
        preceding_phones = " ".join(phonetic_detail["utterance"][max(0, unused_phone - 3):unused_phone])
        following_phones = " ".join(phonetic_detail["utterance"][unused_phone + 1:min(len(phonetic_detail["utterance"]), unused_phone + 4)])
        unused_phone_str = phonetic_detail["utterance"][unused_phone]
        L.warning(f"Unused phone {unused_phone_str} in item {idx} ({item['text']}) (preceding: {preceding_phones}, following: {following_phones})")

    # from pprint import pprint
    # pprint(list(zip(word_detail["start"], word_detail["stop"], word_detail["utterance"])))
    # pprint(list(zip(phonetic_detail["start"], phonetic_detail["stop"], phonetic_detail["utterance"])))
    # pprint(word_phonetic_detail)

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
    try:
        grouped_phonemic_detail = item["word_phonemic_detail"]
        grouped_syllable_detail = item["word_syllable_detail"]
        assert len(grouped_phonemic_detail) == len(item["word_detail"]["utterance"])
        assert len(grouped_syllable_detail) == len(item["word_detail"]["utterance"])

        # Our processing strategy does create empty words.
        # We assign phones to the earliest word whose span contains the phoneme onset,
        # and not to later words. Because TIMIT phones can span words, this can create
        # empty words (e.g. where a single phoneme spans the offset of word 1 and the
        # entirety of word 2).
        # # No empty words
        # assert all(len(word) > 0 for word in grouped_phonemic_detail)
        # # No empty syllables
        # assert all(len(syllables) > 0 for syllables in grouped_syllable_detail), \
        #     f"Empty syllables in item {idx} ({item['text']})"

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
    except Exception as e:
        L.error(f"Error in item {idx} ({item['text']})")
        raise e


def prepare_corpus(corpus,
                   processor: transformers.Wav2Vec2Processor,
                   drop_phones=("[SIL]",)):
    """
    Load and prepare TIMIT corpus for training.
    """

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

    # Remove original audio
    corpus = corpus.remove_columns("audio")
    
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
        if (Path(raw_data_dir) / "dataset_dict.json").exists():
            # Huggingface-style raw TIMIT
            raw_corpus = load_dataset(raw_data_dir)
        else:
            # real raw TIMIT
            raw_corpus = load_dataset("timit_asr", data_dir=raw_data_dir)

        corpus = prepare_corpus(raw_corpus, processor, drop_phones=drop_phones)
        corpus.save_to_disk(processed_data_dir)
    return corpus


def get_word_metadata(word_state_space):
    """
    Augment the given word state space with linguistic data.
    """
    word_freq_df = pd.read_csv("data/WorldLex_Eng_US.Freq.2.txt", sep="\t", index_col="Word")
    # compute weighted average frequency across domains
    word_freq_df["BlogFreq_rel"] = word_freq_df.BlogFreq / word_freq_df.BlogFreq.sum()
    word_freq_df["TwitterFreq_rel"] = word_freq_df.TwitterFreq / word_freq_df.TwitterFreq.sum()
    word_freq_df["NewsFreq_rel"] = word_freq_df.NewsFreq / word_freq_df.NewsFreq.sum()
    word_freq_df["Freq"] = word_freq_df[["BlogFreq", "TwitterFreq", "NewsFreq"]].mean(axis=1) \
        * word_freq_df[["BlogFreq", "TwitterFreq", "NewsFreq"]].sum().mean()
    
    # Fixes for TIMIT
    word_freq_df.loc["'em"] = word_freq_df.loc["them"]
    word_freq_df.loc["cap'n"] = word_freq_df.loc["captain"]
    word_freq_df.loc["playin'"] = word_freq_df.loc["playing"]
    word_freq_df.loc["goin'"] = word_freq_df.loc["going"]
    word_freq_df.loc["takin'"] = word_freq_df.loc["taking"]
    word_freq_df.loc["givin'"] = word_freq_df.loc["giving"]
    word_freq_df.loc["doin'"] = word_freq_df.loc["doing"]
    word_freq_df.loc["y'all"] = word_freq_df.loc["you"]
    word_freq_df.loc["c'mon"] = word_freq_df.loc["come"]
    word_freq_df.loc["ma'am"] = word_freq_df.loc["madam"]
    word_freq_df.loc["herdin'"] = word_freq_df.loc["herding"]

    word_metadata = word_state_space.cuts.xs("syllable", level="level") \
        .groupby(["label", "instance_idx"]).description.count().rename("num_syllables").to_frame()
    word_metadata["monosyllabic"] = word_metadata.num_syllables == 1
    word_metadata["word_freq_lookup"] = word_metadata.index.get_level_values("label") \
        .str.replace("'([std]|ll|re)$", "", regex=True) \
        .str.replace("(you|i|we|they)'(ve|m)$", "\\2", regex=True) \
        .str.replace("(could|would)'ve$", "\\1", regex=True)
    word_metadata["word_frequency"] = word_metadata.word_freq_lookup.map(word_freq_df.Freq.to_dict())
    missing_words = word_metadata[word_metadata.word_frequency.isna()].index.get_level_values("label").unique()
    print("Missing words: ", missing_words.to_list())
    word_metadata["word_frequency"] = word_metadata.word_frequency.fillna(np.percentile(word_freq_df.Freq, 2))
    print("Word frequency tertile split:\n", word_metadata.word_frequency.quantile([0.33, 0.66]))
    word_metadata["word_frequency_quantile"] = pd.qcut(word_metadata.word_frequency, 3, labels=["low", "med", "high"])
    print(word_metadata.groupby("word_frequency_quantile").apply(lambda xs: xs.sample(5).index.get_level_values("label").to_list()))

    word_phoneme_metadata = word_state_space.cuts.xs("phoneme", level="level") \
        .groupby(["label", "instance_idx", "item_idx"]).apply(
            lambda xs: pd.Series({"onset_phoneme": xs.iloc[0].description,
                                "onset_biphone": xs.description.iloc[:2].str.cat(sep=" ")}))

    return pd.merge(word_metadata, word_phoneme_metadata, left_index=True, right_index=True,
                    how="left", validate="one_to_one")