from collections import defaultdict
from beartype import beartype
from dataclasses import dataclass, replace
from functools import cached_property
import itertools
from pathlib import Path
from typing import TypeAlias, Callable, Any, Hashable, Optional, Union

import datasets
import h5py
from jaxtyping import Float, Int64
import numpy as np
import torch
from torch import Tensor as T
from tqdm.auto import trange, tqdm


# defines the critical logic by which frames are equivalence-classed.
# Functions accept grouped word phonemic annotations, word utterances and
# frame index, and return arbitrary grouping value.
EquivalenceClasser: TypeAlias = Callable[[Any, str, int], Hashable]
equivalence_classers: dict[str, EquivalenceClasser] = {
    "phoneme_within_word_prefix": 
        lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail[:i+1]),
    "phoneme": lambda word_phonemic_detail, word_str, i: word_phonemic_detail[i]["phone"],
    "next_phoneme": lambda word_phonemic_detail, word_str, i: word_phonemic_detail[i+1]["phone"] if i + 1 < len(word_phonemic_detail) else None,
    "phoneme_within_word_suffix": lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail[i:]),
    "word_suffix": lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail[i + 1:]),
    "word": lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail),
    "word_broad": lambda word_phonemic_detail, word_str, i: word_str,

    "biphone_recon": lambda word_phonemic_detail, word_str, i: (word_phonemic_detail[i-1]["phone"] if i > 0 else "#", word_phonemic_detail[i]["phone"]),
    "biphone_pred": lambda word_phonemic_detail, word_str, i: (word_phonemic_detail[i]["phone"], word_phonemic_detail[i+1]["phone"] if i + 1 < len(word_phonemic_detail) else "#"),

    "phoneme_fixed": lambda word_phonemic_detail, word_str, i: word_phonemic_detail[i]["phone"],

    "syllable": lambda word_phonemic_detail, word_str, i: tuple(word_phonemic_detail[i]["syllable_phones"]) if word_phonemic_detail[i]["syllable_phones"] else None,
}

def syllable_anti_grouper(word_phonemic_detail, i):
    # mismatch: current syllable
    mismatch_key = tuple(word_phonemic_detail[i]["syllable_phones"]) if word_phonemic_detail[i]["syllable_phones"] else None
    if mismatch_key is None:
        return None, None

    # match: preceding syllable
    current_syllable_idx = word_phonemic_detail[i]["syllable_idx"]
    if current_syllable_idx == 0:
        return None, None
    for j in range(i - 1, -1, -1):
        if word_phonemic_detail[j]["syllable_idx"] < current_syllable_idx:
            return tuple(word_phonemic_detail[j]["syllable_phones"]) if word_phonemic_detail[j]["syllable_phones"] else None, None

    return None, None


# Define subequivalence classing logic. Subequivalence classes define the conjunction of 1) a mismatch
# on an equivalence class and 2) a match on a second equivalence class. This can be used to define
# hard negative samples, for example -- phonemes which mismatch but appear in the same context,
# or syllables which mismatch but contain a substantial number of similar segments.
#
# For a given word annotation and phoneme index, return a grouping value (on which a frame should be
# mismatched with other frames) and an anti-grouping value (on which a frame should match with other
# frames).
subequivalence_classers: dict[str, Callable] = {
    "phoneme": lambda word, i: (word[i]["phone"], word[i - 1]["phone"] if i > 0 else "#"),

    # anti-grouping: preceding syllable
    "syllable": syllable_anti_grouper,
}

# Each equivalence classer also defines a function to compute the start of the
# event which is a sufficient statistic for the class.
start_references = {
    "phoneme_within_word_prefix": "word",
    "phoneme": "word",
    "next_phoneme": "word",
    "phoneme_within_word_suffix": "word",
    "word_suffix": "word",
    "word": "word",
    "word_broad": "word",

    "biphone_recon": "word",
    "biphone_pred": "word",

    "phoneme_fixed": "fixed",

    "syllable": "word",
}


@dataclass
class SpeechHiddenStateDataset:
    model_name: str

    # num_frames * num_layers * hidden_size
    # states: Float[T, "num_frames num_layers hidden_size"]
    states: h5py.Dataset

    # mapping from flattened frame index to (item index, frame index)
    flat_idxs: list[tuple[int, int]]

    _file_handle: Optional[h5py.File] = None

    def __post_init__(self):
        assert self.states.ndim == 3
        assert len(self.flat_idxs) == self.states.shape[0]

    def get_layer(self, layer: int) -> torch.Tensor:
        return torch.tensor(self.states[:, layer, :][()])

    def __repr__(self):
        return f"SpeechHiddenStateDataset({self.model_name}, {self.num_items} items, {self.num_frames} frames, {self.num_layers} layers, {self.states.shape[2]} hidden size)"
    __str__ = __repr__

    def to_hdf5(self, path: str):
        with h5py.File(path, "w") as f:
            f.attrs["model_name"] = self.model_name
            f.create_dataset("states", data=self.states.numpy())
            f.create_dataset("flat_idxs", data=self.flat_idxs, dtype=np.int32)
    
    @classmethod
    def from_hdf5(cls, path: Union[str, Path]):
        f = h5py.File(path, "r")
        model_name = f.attrs["model_name"]
        states = f["states"]  # NB not loading into memory
        flat_idxs = f["flat_idxs"][:]

        return cls(model_name=model_name, states=states, flat_idxs=flat_idxs,
                    _file_handle=f)

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


@beartype
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

    Q: Int64[T, "num_frames"]
    S: Int64[T, "num_frames"]

    class_labels: list[Any]
    """
    For each equivalence class, a description of its content.
    """

    def __post_init__(self):
        assert self.Q.shape[0] == self.S.shape[0]

        assert self.Q.max() < len(self.class_labels)
        assert self.Q.min() >= -1

        # If Q is set, then S should be set
        assert (self.Q == -1).logical_or(self.S != -1).all()

    def __repr__(self):
        return f"SpeechEquivalenceDataset({self.name}, {self.num_classes} classes, {self.num_instances} instances)"

    def is_compatible_with(self, dataset: SpeechHiddenStateDataset):
        return self.Q.shape[0] == dataset.num_frames

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
    
    @cached_property
    def class_to_frames(self):
        chunksize = 100000
        equiv_class_to_idxs = defaultdict(list)
        for start in trange(0, self.Q.shape[0], chunksize, desc="Preparing equiv class mappings"):
            matches = (self.Q[start:start + chunksize] == torch.arange(self.num_classes)[:, None]).nonzero().numpy()
            for class_idx, frame_idxs in itertools.groupby(matches, key=lambda x: x[0]):
                frame_idxs = start + np.array(list(frame_idxs))[:, 1]
                equiv_class_to_idxs[class_idx].extend(frame_idxs.tolist())

        return equiv_class_to_idxs
    
    def drop_lengths(self, max_length: int) -> "SpeechEquivalenceDataset":
        """
        Drop all frames with length greater than `max_length`.
        """
        mask = self.lengths > max_length

        new_Q = self.Q.clone()
        new_Q[mask] = -1
        new_S = self.S.clone()
        new_S[mask] = -1

        return replace(self, Q=new_Q, S=new_S)
    

def make_timit_equivalence_dataset(name: str,
                                   dataset: datasets.Dataset,
                                   hidden_states: SpeechHiddenStateDataset,
                                   equivalence_classer: str,
                                   minimum_frequency_percentile: float = 0.,
                                   max_length: Optional[int] = 100,
                                   num_frames_per_phoneme=None) -> SpeechEquivalenceDataset:
    """
    TIMIT-specific function to prepare an equivalence-classed frame dataset
    from a TIMIT dataset and a speech model.

    NB that equivalence classing is not specific to models and could be
    decoupled in principle.
    """
    assert equivalence_classer in equivalence_classers

    frame_groups = defaultdict(list)
    frames_by_item = hidden_states.frames_by_item

    # Align with TIMIT annotations
    def process_item(item, idx):
        # Now align and store frame metadata
        num_frames = frames_by_item[idx][1] - frames_by_item[idx][0]
        compression_ratio = num_frames / len(item["input_values"])
        for i, word in enumerate(item["word_phonemic_detail"]):
            if len(word) == 0:
                continue

            word_str = item["word_detail"]["utterance"][i]
            word_start = int(word[0]["start"] * compression_ratio)
            word_end = int(word[-1]["stop"] * compression_ratio)

            for j, phone in enumerate(word):
                phone_str = phone["phone"]
                phone_start = int(phone["start"] * compression_ratio)
                phone_end = int(phone["stop"] * compression_ratio)

                ks = list(range(phone_start, phone_end + 1))
                if num_frames_per_phoneme is not None and len(ks) > num_frames_per_phoneme:
                    # Sample uniformly spaced frames within the span of the phoneme
                    ks = np.linspace(phone_start, phone_end, num_frames_per_phoneme).round().astype(int)
                for k in ks:
                    class_label = equivalence_classers[equivalence_classer](word, word_str, j)
                    if class_label is not None:
                        frame_groups[class_label].append((idx, k))
    dataset.map(process_item, with_indices=True, desc="Aligning metadata")

    if minimum_frequency_percentile > 0:
        # Filter out low-frequency classes
        class_counts = {class_key: len(group) for class_key, group in frame_groups.items()}
        class_counts = np.array(list(class_counts.values()))
        min_count = np.percentile(class_counts, minimum_frequency_percentile)

        len_before = len(frame_groups)
        frame_groups = {class_key: group for class_key, group in frame_groups.items() if len(group) >= min_count}
        len_after = len(frame_groups)
        print(f"Filtered out {len_before - len_after} classes with fewer than {min_count} frames")
        print(f"Remaining classes: {len_after}")

    # Now run equivalence classing.
    Q = torch.zeros(len(hidden_states.flat_idxs), dtype=torch.long) - 1
    class_labels = list(frame_groups.keys())
    class_label_to_idx = {class_key: idx for idx, class_key in enumerate(class_labels)}
    flat_idx_rev = {tuple(idx): i for i, idx in enumerate(hidden_states.flat_idxs)}
    for class_key, group in frame_groups.items():
        for idx, frame in group:
            Q[flat_idx_rev[idx, frame]] = class_label_to_idx[class_key]

    # Compute start frames.
    S = torch.zeros(len(hidden_states.flat_idxs), dtype=torch.long) - 1
    start_reference = start_references[equivalence_classer]
    if start_reference == "word":
        def compute_start(item, idx):
            flat_idx_offset, flat_idx_end = frames_by_item[idx]
            num_frames = flat_idx_end - flat_idx_offset
            compression_ratio = num_frames / len(item["input_values"])

            for word in item["word_phonemic_detail"]:
                if len(word) == 0:
                    continue
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
    elif start_reference == "fixed":
        # TODO magic number: fixed number of frames over which to integrate
        fixed_length = 20
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
                        S[flat_idx_offset + j] = max(flat_idx_offset, flat_idx_offset + j - fixed_length)
    else:
        raise ValueError(f"Unknown start reference {start_reference}")
    
    dataset.map(compute_start, with_indices=True, desc="Computing start frames")

    ret = SpeechEquivalenceDataset(name=name,
                                   Q=Q,
                                   S=S,
                                   class_labels=class_labels)
    if max_length is not None:
        ret = ret.drop_lengths(max_length)
    return ret