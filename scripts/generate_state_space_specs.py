from collections import defaultdict
from typing import TypeAlias, cast

import datasets
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

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


STATE_SPACE_COMPUTERS = {
    "word": compute_word_state_space,
    "syllable": compute_syllable_state_space,
    "phoneme": compute_phoneme_state_space,
}

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    dataset = cast(datasets.Dataset, datasets.load_from_disk(config.dataset.processed_data_dir))

    hidden_state_path = config.base_model.hidden_state_path
    with open(hidden_state_path, "rb") as f:
        hidden_states: SpeechHiddenStateDataset = torch.load(f)

    all_state_space_specs = {}
    for name, computer in STATE_SPACE_COMPUTERS.items():
        new_specs = computer(dataset, hidden_states)
        assert not set(new_specs.keys()) & set(all_state_space_specs.keys())
        all_state_space_specs.update(new_specs)

    with open(config.analysis.state_space_specs_path, "wb") as f:
        torch.save(all_state_space_specs, f)    


if __name__ == "__main__":
    main()