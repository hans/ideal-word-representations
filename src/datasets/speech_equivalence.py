from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias, Callable, Any, Hashable

import datasets
import numpy as np
import torch
import transformers


# defines the critical logic by which frames are equivalence-classed.
# Functions accept word annotation and frame index, and return arbitrary
# grouping value.
EquivalenceClasser: TypeAlias = Callable[[Any, int], Hashable]
equivalence_classers: dict[str, EquivalenceClasser] = {
    "phoneme_within_word_prefix": 
        lambda word, i: tuple(phone["phone"] for phone in word[:i+1]),
}

# Each equivalence classer also defines a function to compute the start of the
# event which is a sufficient statistic for the class.
start_references = {
    "phoneme_within_word_prefix": "word",
}


@dataclass
class SpeechHiddenStateDataset:
    model_name: str

    # num_frames * num_layers * hidden_size
    states: torch.Tensor

    # mapping from flattened frame index to (item index, frame index)
    flat_idxs: list[tuple[int, int]]

    def __post_init__(self):
        assert self.states.ndim == 3
        assert len(self.flat_idxs) == self.states.shape[0]

    def get_layer(self, layer: int) -> torch.Tensor:
        return self.states[:, layer, :]

    def __repr__(self):
        return f"SpeechHiddenStateDataset({self.model_name}, {self.num_items} items, {self.num_frames} frames, {self.num_layers} layers, {self.states.shape[2]} hidden size)"
    __str__ = __repr__

    @property
    def num_frames(self) -> int:
        return len(self.flat_idxs)
    
    @property
    def num_items(self) -> int:
        return len(self.frames_by_item)
    
    @property
    def num_layers(self) -> int:
        return self.states.shape[1]
    
    @property
    def hidden_size(self) -> int:
        return self.states.shape[2]

    @cached_property
    def frames_by_item(self) -> dict[int, tuple[int, int]]:
        """Mapping from item number to flat idx frame start, end (exclusive)"""
        item_idxs, flat_idx_starts = np.unique(np.array(self.flat_idxs)[:, 0], return_index=True)
        flat_idx_ends = np.concatenate([flat_idx_starts[1:], [len(self.flat_idxs)]])
        return {item: (start, end) for item, start, end in zip(item_idxs, flat_idx_starts, flat_idx_ends)}


@dataclass
class SpeechEquivalenceDataset:
    """
    Represents an equivalence classing over a `SpeechHiddenStateDataset`.

    Each frame is annotated by
        1) a class, and
        2) a start frame. An ideal model should be able to aggregrate hidden states from the
           start frame to the current frame and predict this class. In this sense the start
           frame defines a sufficient statistic for the class on an instance level.

    These annotations are represented in the vectors `Q` and `S`, respectively.
    """

    name: str

    hidden_state_dataset: SpeechHiddenStateDataset
    Q: torch.Tensor
    S: torch.Tensor

    class_labels: list[Any]
    """
    For each equivalence class, a description of its content.
    """

    def __post_init__(self):
        assert self.Q.shape[0] == self.hidden_state_dataset.num_frames
        assert self.S.shape[0] == self.hidden_state_dataset.num_frames

        assert self.Q.max() < len(self.class_labels)
        assert self.Q.min() >= -1

        # If Q is set, then S should be set
        assert (self.Q == -1).logical_or(self.S != -1).all()

        # Sanity check: no crazy long events
        insane_length = 100
        evident_lengths = self.lengths
        evident_lengths = evident_lengths[evident_lengths != -1]
        assert evident_lengths.max() < insane_length

    def __repr__(self):
        return f"SpeechEquivalenceDataset({self.name}, {self.num_classes} classes, {self.num_instances} instances, with {self.hidden_state_dataset})"

    @property
    def num_instances(self) -> int:
        return (self.Q != -1).long().sum().item()

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)
    
    @cached_property
    def lengths(self):
        lengths = torch.arange(self.S.shape[0]) - self.S
        lengths[self.S == -1] = -1
        return lengths
    

def make_timit_equivalence_dataset(name: str,
                                   dataset: datasets.Dataset,
                                   model: transformers.PreTrainedModel,
                                   classer: str,
                                   num_frames_per_phoneme=None) -> SpeechEquivalenceDataset:
    """
    TIMIT-specific function to prepare an equivalence-classed frame dataset
    from a TIMIT dataset and a speech model.

    NB that equivalence classing is not specific to models and could be
    decoupled in principle.
    """
    assert classer in equivalence_classers

    flat_idxs = []
    frames_by_item = {}
    frame_states_list = []
    frame_groups = defaultdict(list)

    def process_item(item, idx):
        with torch.no_grad():
            output = model(output_hidden_states=True,
                           input_values=torch.tensor(item["input_values"]).unsqueeze(0).to(model.device))
            
        # num_layers * sequence_length * hidden_size
        batch_hidden = torch.stack(output.hidden_states).squeeze(1).cpu()

        flat_idx_offset = len(flat_idxs)
        flat_idxs.extend([(idx, i) for i in range(batch_hidden.shape[1])])
        frames_by_item[idx] = (flat_idx_offset, len(flat_idxs))
        frame_states_list.append(batch_hidden)

        # Now align and store frame metadata
        compression_ratio = batch_hidden.shape[1] / len(item["input_values"])
        for word in item["word_phonemic_detail"]:
            if len(word) == 0:
                continue

            word_str = tuple(phone["phone"] for phone in word)
            word_start = int(word[0]["start"] * compression_ratio)
            word_end = int(word[-1]["stop"] * compression_ratio)

            for j, phone in enumerate(word):
                word_prefix = word_str[:j + 1]

                phone_str = phone["phone"]
                phone_start = int(phone["start"] * compression_ratio)
                phone_end = int(phone["stop"] * compression_ratio)

                ks = list(range(phone_start, phone_end + 1))
                if num_frames_per_phoneme is not None and len(ks) > num_frames_per_phoneme:
                    ks = ks[-num_frames_per_phoneme:]
                for k in ks:
                    for classer, fn in equivalence_classers.items():
                        class_key = fn(word, j)
                        frame_groups[class_key].append((idx, k))

    dataset.map(process_item, with_indices=True, desc="Extracting hidden states")

    frame_states = torch.cat(frame_states_list, dim=1)
    frame_states = frame_states.transpose(0, 1).contiguous()

    # Now run equivalence classing.
    Q = torch.zeros(len(flat_idxs), dtype=torch.long) - 1
    class_labels = list(frame_groups.keys())
    class_label_to_idx = {class_key: idx for idx, class_key in enumerate(class_labels)}
    flat_idx_rev = {idx: i for i, idx in enumerate(flat_idxs)}
    for class_key, group in frame_groups.items():
        for idx, frame in group:
            Q[flat_idx_rev[idx, frame]] = class_label_to_idx[class_key]

    # Compute start frames.
    S = torch.zeros(len(flat_idxs), dtype=torch.long) - 1
    start_reference = start_references[classer]
    if start_reference == "word":
        def compute_start(item, idx):
            flat_idx_offset, flat_idx_end = frames_by_item[idx]
            num_frames = flat_idx_end - flat_idx_offset
            compression_ratio = num_frames / len(item["input_values"])

            for word in item["word_phonemic_detail"]:
                word_str = tuple(phone["phone"] for phone in word)
                word_start = int(word[0]["start"] * compression_ratio)
                word_end = int(word[-1]["stop"] * compression_ratio)

                for j in range(word_start, word_end + 1):
                    S[flat_idx_offset + j] = flat_idx_offset + word_start
    elif start_reference == "phoneme":
        def compute_start(item, idx):
            flat_idx_offset, flat_idx_end = frames_by_item[idx]
            num_frames = flat_idx_end - flat_idx_offset
            compression_ratio = num_frames / len(item["input_values"])

            for word in item["word_phonemic_detail"]:
                for phone in word:
                    phone_str = phone["phone"]
                    phone_start = int(phone["start"] * compression_ratio)
                    phone_end = int(phone["stop"] * compression_ratio)

                    for j in range(phone_start, phone_end + 1):
                        S[flat_idx_offset + j] = flat_idx_offset + phone_start
    else:
        raise ValueError(f"Unknown start reference {start_reference}")
    
    dataset.map(compute_start, with_indices=True, desc="Computing start frames")

    # Prepare return datasets
    hidden_state_dataset = SpeechHiddenStateDataset(model_name=model.name_or_path,
                                                    states=frame_states,
                                                    flat_idxs=flat_idxs)
    return SpeechEquivalenceDataset(name=name,
                                    hidden_state_dataset=hidden_state_dataset,
                                    Q=Q,
                                    S=S,
                                    class_labels=class_labels)