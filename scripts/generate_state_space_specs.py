from collections import defaultdict
from typing import TypeAlias, cast

import datasets
import hydra
from omegaconf import DictConfig
import pandas as pd

import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

from src.analysis.state_space import StateSpaceAnalysisSpec
from src.datasets.speech_equivalence import SpeechHiddenStateDataset, SpeechEquivalenceDataset


SpecGroup: TypeAlias = dict[str, StateSpaceAnalysisSpec]

def compute_word_state_space(dataset: datasets.Dataset,
                             hidden_state_dataset: SpeechHiddenStateDataset,
                             ) -> SpecGroup:
    frames_by_item = hidden_state_dataset.frames_by_item
    frame_spans_by_word = defaultdict(list)
    cuts_df = []

    def process_item(item):
        # How many frames do we have stored for this item?
        start_frame, stop_frame = frames_by_item[item["idx"]]
        num_frames = stop_frame - start_frame

        compression_ratio = num_frames / len(item["input_values"])

        for i, word_detail in enumerate(item["word_syllable_detail"]):
            if not word_detail:
                continue

            word_start_frame = start_frame + int(word_detail[0]["start"] * compression_ratio)
            word_stop_frame = start_frame + int(word_detail[-1]["stop"] * compression_ratio)
            word = item["word_detail"]["utterance"][i]

            instance_idx = len(frame_spans_by_word[word])
            frame_spans_by_word[word].append((word_start_frame, word_stop_frame))

            for syllable in word_detail:
                cuts_df.append({
                    "label": word,
                    "instance_idx": instance_idx,
                    "level": "syllable",
                    "description": tuple(syllable["phones"]),
                    "onset_frame_idx": start_frame + int(syllable["start"] * compression_ratio),
                    "offset_frame_idx": start_frame + int(syllable["stop"] * compression_ratio),
                    "item_idx": item["idx"],
                })

            for phoneme in item["word_phonemic_detail"][i]:
                cuts_df.append({
                    "label": word,
                    "instance_idx": instance_idx,
                    "level": "phoneme",
                    "description": phoneme["phone"],
                    "onset_frame_idx": start_frame + int(phoneme["start"] * compression_ratio),
                    "offset_frame_idx": start_frame + int(phoneme["stop"] * compression_ratio),
                    "item_idx": item["idx"],
                })

    dataset.map(process_item, batched=False)

    words = list(frame_spans_by_word.keys())
    spans = list(frame_spans_by_word.values())

    spec = StateSpaceAnalysisSpec(
        total_num_frames=hidden_state_dataset.num_frames,
        labels=words,
        target_frame_spans=spans,
        cuts=pd.DataFrame(cuts_df).set_index(["label", "instance_idx", "level"]).sort_index(),
    )

    return {"word": spec}


def compute_word_state_space_no_syllable(
        dataset: datasets.Dataset,
        hidden_state_dataset: SpeechHiddenStateDataset,
        ) -> SpecGroup:
    """Word-level state space for datasets that lack `word_syllable_detail`.

    Word frame spans come from `word_detail.start/stop`; cuts contain only
    `level=phoneme` rows (no syllable level). Downstream code that does
    `cuts.xs("phoneme", level="level")` works unchanged.
    """
    frames_by_item = hidden_state_dataset.frames_by_item
    frame_spans_by_word = defaultdict(list)
    cuts_df = []

    def process_item(item):
        start_frame, stop_frame = frames_by_item[item["idx"]]
        num_frames = stop_frame - start_frame

        compression_ratio = num_frames / len(item["input_values"])

        word_starts = item["word_detail"]["start"]
        word_stops = item["word_detail"]["stop"]
        word_utts = item["word_detail"]["utterance"]
        for i, word in enumerate(word_utts):
            if not word:
                continue
            phonemes = item["word_phonemic_detail"][i]
            if not phonemes:
                continue

            word_start_frame = start_frame + int(word_starts[i] * compression_ratio)
            word_stop_frame = start_frame + int(word_stops[i] * compression_ratio)

            instance_idx = len(frame_spans_by_word[word])
            frame_spans_by_word[word].append((word_start_frame, word_stop_frame))

            for phoneme in phonemes:
                cuts_df.append({
                    "label": word,
                    "instance_idx": instance_idx,
                    "level": "phoneme",
                    "description": phoneme["phone"],
                    "onset_frame_idx": start_frame + int(phoneme["start"] * compression_ratio),
                    "offset_frame_idx": start_frame + int(phoneme["stop"] * compression_ratio),
                    "item_idx": item["idx"],
                })

    dataset.map(process_item, batched=False)

    words = list(frame_spans_by_word.keys())
    spans = list(frame_spans_by_word.values())

    spec = StateSpaceAnalysisSpec(
        total_num_frames=hidden_state_dataset.num_frames,
        labels=words,
        target_frame_spans=spans,
        cuts=pd.DataFrame(cuts_df).set_index(["label", "instance_idx", "level"]).sort_index(),
    )

    return {"word": spec}


def compute_syllable_state_space(dataset: datasets.Dataset,
                                 hidden_state_dataset: SpeechHiddenStateDataset,
                                 ) -> SpecGroup:
    frames_by_item = hidden_state_dataset.frames_by_item
    
    frame_spans_by_syllable = defaultdict(list)
    frame_spans_by_syllable_ordinal = defaultdict(list)
    frame_spans_by_syllable_and_ordinal = defaultdict(list)

    # aggregate cuts just for frame_spans_by_syllable
    cuts_df = []

    def process_item(item):
        # How many frames do we have stored for this item?
        start_frame, stop_frame = frames_by_item[item["idx"]]
        num_frames = stop_frame - start_frame

        compression_ratio = num_frames / len(item["input_values"])

        for i, word in enumerate(item["word_syllable_detail"]):
            word_phones = item["word_phonemic_detail"][i]
        
            for syllable in word:
                syllable_start_frame = start_frame + int(syllable["start"] * compression_ratio)
                syllable_stop_frame = start_frame + int(syllable["stop"] * compression_ratio)

                syllable_phones = tuple(syllable["phones"])
                instance_idx = len(frame_spans_by_syllable[syllable_phones])
                span = (syllable_start_frame, syllable_stop_frame)
                
                frame_spans_by_syllable[syllable_phones].append(span)
                frame_spans_by_syllable_ordinal[syllable["idx"]].append(span)
                frame_spans_by_syllable_and_ordinal[(syllable_phones, syllable["idx"])].append(span)

                syllable_phone_detail = word_phones[syllable["phoneme_start_idx"] : syllable["phoneme_end_idx"]]
                for phone_detail in syllable_phone_detail:
                    cuts_df.append({
                        "label": syllable_phones,
                        "instance_idx": instance_idx,
                        "level": "phoneme",
                        "description": phone_detail["phone"],
                        "onset_frame_idx": start_frame + int(phone_detail["start"] * compression_ratio),
                        "offset_frame_idx": start_frame + int(phone_detail["stop"] * compression_ratio),
                        "item_idx": item["idx"],
                    })

    dataset.map(process_item, batched=False)

    return {
        "syllable": StateSpaceAnalysisSpec(
            total_num_frames=hidden_state_dataset.num_frames,
            labels=list(frame_spans_by_syllable.keys()),
            target_frame_spans=list(frame_spans_by_syllable.values()),
            cuts=pd.DataFrame(cuts_df).set_index(["label", "instance_idx", "level"]).sort_index(),
        ),

        "syllable_by_ordinal": StateSpaceAnalysisSpec(
            total_num_frames=hidden_state_dataset.num_frames,
            labels=list(frame_spans_by_syllable_ordinal.keys()),
            target_frame_spans=list(frame_spans_by_syllable_ordinal.values()),
        ),

        "syllable_by_identity_and_ordinal": StateSpaceAnalysisSpec(
            total_num_frames=hidden_state_dataset.num_frames,
            labels=list(frame_spans_by_syllable_and_ordinal.keys()),
            target_frame_spans=list(frame_spans_by_syllable_and_ordinal.values()),
        )
    }


def compute_phoneme_state_space(dataset: datasets.Dataset,
                                hidden_state_dataset: SpeechHiddenStateDataset,
                                ) -> SpecGroup:
    frames_by_item = hidden_state_dataset.frames_by_item

    frame_spans_by_phoneme = defaultdict(list)
    frame_spans_by_phoneme_position = defaultdict(list)
    frame_spans_by_syllable_index = defaultdict(list)
    frame_spans_by_phoneme_and_syllable_index = defaultdict(list)

    def process_item(item):
        # How many frames do we have stored for this item?
        start_frame, stop_frame = frames_by_item[item["idx"]]
        num_frames = stop_frame - start_frame

        compression_ratio = num_frames / len(item["input_values"])

        for word in item["word_phonemic_detail"]:
            for i, phone in enumerate(word):
                phone_start_frame = start_frame + int(phone["start"] * compression_ratio)
                phone_stop_frame = start_frame + int(phone["stop"] * compression_ratio)

                frame_spans_by_phoneme[phone["phone"]].append((phone_start_frame, phone_stop_frame))
                frame_spans_by_phoneme_position[i].append((phone_start_frame, phone_stop_frame))
                frame_spans_by_syllable_index[phone["syllable_idx"]].append((phone_start_frame, phone_stop_frame))
                frame_spans_by_phoneme_and_syllable_index[(phone["phone"], phone["syllable_idx"])].append((phone_start_frame, phone_stop_frame))

    dataset.map(process_item, batched=False)

    phoneme_positions = sorted(frame_spans_by_phoneme_position.keys())
    phonemes = sorted(frame_spans_by_phoneme.keys())

    return {
        "phoneme": StateSpaceAnalysisSpec(
            total_num_frames=hidden_state_dataset.num_frames,
            labels=phonemes,
            target_frame_spans=[frame_spans_by_phoneme[phone] for phone in phonemes],
        ),

        "phoneme_by_syllable_position": StateSpaceAnalysisSpec(
            total_num_frames=hidden_state_dataset.num_frames,
            labels=phoneme_positions,
            target_frame_spans=[frame_spans_by_phoneme_position[i] for i in phoneme_positions],
        ),

        "phoneme_by_syllable_position_and_identity": StateSpaceAnalysisSpec(
            total_num_frames=hidden_state_dataset.num_frames,
            labels=list(frame_spans_by_phoneme_and_syllable_index.keys()),
            target_frame_spans=list(frame_spans_by_phoneme_and_syllable_index.values()),
        ),
    }


def compute_biphone_state_space(dataset: datasets.Dataset,
                                hidden_state_dataset: SpeechHiddenStateDataset,
                                ):
    frames_by_item = hidden_state_dataset.frames_by_item

    from collections import defaultdict
    frame_spans_by_biphone = defaultdict(list)
    cuts_df = []

    def process_item(item):
        # How many frames do we have stored for this item?
        start_frame, stop_frame = frames_by_item[item["idx"]]
        num_frames = stop_frame - start_frame

        compression_ratio = num_frames / len(item["input_values"])

        for word in item["word_phonemic_detail"]:
            if len(word) == 0:
                continue

            start_dummy = {"phone": "#", "start": word[0]["start"], "stop": word[0]["stop"]}
            end_dummy = {"phone": "#", "start": word[-1]["start"], "stop": word[-1]["stop"]}
            word = [start_dummy] + word + [end_dummy]
                                     
            for i, (p1, p2) in enumerate(zip(word, word[1:])):
                biphone_start_frame = start_frame + int(p1["start"] * compression_ratio)
                biphone_stop_frame = start_frame + int(p2["stop"] * compression_ratio)

                biphone_label = (p1["phone"], p2["phone"])
                instance_idx = len(frame_spans_by_biphone[biphone_label])
                frame_spans_by_biphone[biphone_label].append((biphone_start_frame, biphone_stop_frame))

                # add constituent phonemes to cuts
                cuts_df.append({
                    "label": biphone_label,
                    "instance_idx": instance_idx,
                    "level": "phoneme",
                    "description": p1["phone"],
                    "onset_frame_idx": start_frame + int(p1["start"] * compression_ratio),
                    "offset_frame_idx": start_frame + int(p1["stop"] * compression_ratio),
                    "item_idx": item["idx"],
                })
                cuts_df.append({
                    "label": biphone_label,
                    "instance_idx": instance_idx,
                    "level": "phoneme",
                    "description": p2["phone"],
                    "onset_frame_idx": start_frame + int(p2["start"] * compression_ratio),
                    "offset_frame_idx": start_frame + int(p2["stop"] * compression_ratio),
                    "item_idx": item["idx"],
                })

    dataset.map(process_item, batched=False)

    biphones = sorted(frame_spans_by_biphone.keys())
    return {
        "biphone": StateSpaceAnalysisSpec(
            total_num_frames=hidden_state_dataset.num_frames,
            labels=biphones,
            target_frame_spans = [frame_spans_by_biphone[b] for b in biphones],
            cuts=pd.DataFrame(cuts_df).set_index(["label", "instance_idx", "level"]).sort_index(),
        )
    }


STATE_SPACE_COMPUTERS = {
    "word": compute_word_state_space,
    "syllable": compute_syllable_state_space,
    "biphone": compute_biphone_state_space,
    "phoneme": compute_phoneme_state_space,
}

# Subset used when the dataset has no syllabification (e.g. MLS French — no
# French syllabifier in the preprocessing pipeline). The phoneme state space
# still works because `compute_phoneme_state_space` reads `word_phonemic_detail`
# directly, not `word_syllable_detail` — but `syllable_idx` will be missing,
# which the by_syllable_* variants depend on, so they're dropped too.
STATE_SPACE_COMPUTERS_NO_SYLLABLE = {
    "word": compute_word_state_space_no_syllable,
}

@delayed
def compute_state_space_spec(name: str, computer, dataset_path, hidden_state_path):
    dataset = cast(datasets.Dataset, datasets.load_from_disk(dataset_path))
    hidden_states = SpeechHiddenStateDataset.from_hdf5(hidden_state_path)

    return name, computer(dataset, hidden_states)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    dask_client = Client(LocalCluster())

    has_syllables = config.dataset.get("has_syllables", True)
    computers = STATE_SPACE_COMPUTERS if has_syllables else STATE_SPACE_COMPUTERS_NO_SYLLABLE

    # compute in parallel
    dask_results = dask.compute(*[
        compute_state_space_spec(name, computer,
                                 config.dataset.processed_data_dir,
                                 config.base_model.hidden_state_path)
        for name, computer in computers.items()
    ])

    # ensure no name collisions
    all_state_space_specs = {}
    for name, new_specs in dask_results:
        assert not set(new_specs.keys()) & set(all_state_space_specs.keys())
        all_state_space_specs.update(new_specs)

    # save
    for name, spec in all_state_space_specs.items():
        spec.to_hdf5(config.analysis.state_space_specs_path, key=name)


if __name__ == "__main__":
    main()