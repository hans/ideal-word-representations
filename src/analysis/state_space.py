"""
State space analysis tools for integrator models.
"""

from functools import cached_property
from dataclasses import dataclass
from typing import Optional, Union, Any, Callable

import numpy as np
import pandas as pd
import torch

from src.datasets.speech_equivalence import SpeechHiddenStateDataset, SpeechEquivalenceDataset



@dataclass
class StateSpaceAnalysisSpec:

    # Number of frames in the dataset associated with this state space spec.
    # Used for validation.
    total_num_frames: int

    labels: list[str]

    # Analyze K categories of N state space trajectories.
    # Tuples are start and end indices, inclusive.
    target_frame_spans: list[list[tuple[int, int]]]

    # Optional representation of frame subdivisions at lower levels of representation.
    # For example, a word-level state space trajectory may retain information here
    # about phoneme-level subdivisions.
    # 
    # DataFrame with index levels (label, instance_idx, level)
    # and columns (description, frame_idx, onset_frame_idx, offset_frame_idx).
    # This refers to the frame span `target_frame_spans[labels.index(label)][instance_idx]`
    cuts: Optional[pd.DataFrame] = None

    def __post_init__(self):
        assert len(self.target_frame_spans) == len(self.labels)

        if self.cuts is not None:
            assert set(self.cuts.index.names) == {"level", "label", "instance_idx"}
            assert set(self.cuts.columns) >= {"description", "onset_frame_idx", "offset_frame_idx"}
            assert set(self.cuts.index.get_level_values("label")) <= set(self.labels)

            assert (self.cuts.onset_frame_idx < self.total_num_frames).all()
            assert (self.cuts.offset_frame_idx < self.total_num_frames).all()

            # Make sure onset and offset idxs are within the span of the instance
            for (label, instance_idx), cuts_group in self.cuts.groupby(["label", "instance_idx"]):
                label_idx = self.labels.index(label)
                start, end = self.target_frame_spans[label_idx][instance_idx]

                assert (cuts_group.onset_frame_idx >= start).all()
                assert (cuts_group.offset_frame_idx <= end).all()

    def is_compatible_with(self, dataset: Union[SpeechHiddenStateDataset, np.ndarray]) -> bool:
        if isinstance(dataset, SpeechHiddenStateDataset):
            return self.total_num_frames == dataset.num_frames
        else:
            return self.total_num_frames == dataset.shape[0]
    
    def drop_labels(self, drop_idxs=None, drop_names=None):
        if drop_idxs is None and drop_names is None:
            raise ValueError("Must provide either drop_idxs or drop_names")
        
        if drop_idxs is None:
            drop_idxs = [i for i, label in enumerate(self.labels) if label in drop_names]
        
        labels = [label for i, label in enumerate(self.labels) if i not in drop_idxs]
        target_frame_spans = [span for i, span in enumerate(self.target_frame_spans) if i not in drop_idxs]

        new_cuts = None
        if self.cuts is not None:
            mask = self.cuts.index.get_level_values("label").isin(labels)
            new_cuts = self.cuts.loc[mask]

        return StateSpaceAnalysisSpec(
            total_num_frames=self.total_num_frames,
            labels=labels,
            target_frame_spans=target_frame_spans,
            cuts=new_cuts,
        )
    
    def expand_by_cut_index(self, cut_level: str) -> "StateSpaceAnalysisSpec":
        """
        Expand the state space analysis spec to include information about
        the given cut index within each class instance.
        """
        if self.cuts is None:
            raise ValueError("No cuts available to expand")

        cuts_df = self.cuts.xs(cut_level, level="level")
        cuts_df["idx_in_level"] = cuts_df.groupby(["label", "instance_idx"]).cumcount()
        new_target_frame_spans = []
        new_labels = []

        for (label, idx_in_level), cuts_group in cuts_df.groupby(["label", "idx_in_level"]):
            new_labels.append((label, idx_in_level))
            new_target_frame_spans.append(list(zip(cuts_group.onset_frame_idx, cuts_group.offset_frame_idx)))

        return StateSpaceAnalysisSpec(
            total_num_frames=self.total_num_frames,
            labels=new_labels,
            target_frame_spans=new_target_frame_spans,
            cuts=None,
        )
    
    @cached_property
    def flat(self) -> np.ndarray:
        """
        Return a "flat" representation indexing into state space trajectories
        by frame index rather than by label and instance index.

        Returns a `total_num_frames` x 4 array, where each row is a reference
        to the start of a state trajectory instance, with columns:
        - start frame index
        - end frame index
        - label index
        - instance index
        """
        flat_references = []
        for i, (label, frame_spans) in enumerate(zip(self.labels, self.target_frame_spans)):
            for j, (start, end) in enumerate(frame_spans):
                flat_references.append((start, end, i, j))

        return np.array(sorted(flat_references))

    def get_trajectories_in_span(self, span_left, span_right) -> np.ndarray:
        """
        Return the state space trajectories that intersect with the given
        frame span (inclusive).
        """
        return self.flat[
            (self.flat[:, 0] <= span_right) & (self.flat[:, 1] >= span_left)
        ]


def prepare_word_trajectory_spec(
        dataset: SpeechEquivalenceDataset,
        target_words: list[tuple[str, ...]],
) -> StateSpaceAnalysisSpec:
    """
    Retrieve a list of frame spans describing a speech perception
    trajectory matching the given target words.
    """

    # We expect the dataset class labels to correspond to words
    assert all(type(label) == tuple for label in dataset.class_labels)

    target_labels = [dataset.class_labels.index(word) for word in target_words]
    frame_bounds = []
    for label_idx in target_labels:
        final_frames = torch.where(dataset.Q == label_idx)[0]
        start_frames = dataset.S[final_frames]
        frame_bounds.append(list(zip(start_frames.numpy(), final_frames.numpy())))

    return StateSpaceAnalysisSpec(
        target_frame_spans=frame_bounds,
        labels=target_words,
        total_num_frames=dataset.hidden_state_dataset.num_frames,
    )



def prepare_state_trajectory(
        embeddings: np.ndarray,
        spec: StateSpaceAnalysisSpec,
        expand_window: Optional[tuple[int, int]] = None,
        pad="last",
) -> list[np.ndarray]:
    """
    Prepare the state trajectory for the given dataset and model embeddings.

    If `expand_window` is not None, add `expand_window[0]` frames to the left
    of each trajectory and `expand_window[1]` frames to the right.
    """
    max_num_frames = max(max(end - start + 1 for start, end in trajectory_spec)
                         for trajectory_spec in spec.target_frame_spans)
    if expand_window is not None:
        max_num_frames += expand_window[0] + expand_window[1]
    ret = []

    for i, frame_spec in enumerate(spec.target_frame_spans):
        num_instances = len(frame_spec)
        trajectory_frames = np.zeros((num_instances, max_num_frames, embeddings.shape[1]))
        for j, (start, end) in enumerate(frame_spec):
            if expand_window is not None:
                start = max(0, start - expand_window[0])
                end = min(spec.total_num_frames - 1, end + expand_window[1])

            trajectory_frames[j, :end - start + 1] = embeddings[start:end + 1]

            # Pad on right
            if pad == "last":
                pad_value = embeddings[end]
            elif isinstance(pad, str):
                raise ValueError(f"Invalid pad value {pad}; use `last` or a float")
            else:
                pad_value = pad

            if end - start + 1 < max_num_frames:
                trajectory_frames[j, end - start + 1:] = pad_value

        ret.append(trajectory_frames)

    return ret


def make_agg_fn_mean_last_k(k):
    def agg_fn(xs):
        nan_onset = np.isnan(xs[:, :, 0]).argmax(axis=1)
        # if there are no nans, set nan_onset to len
        nan_onset[~np.isnan(xs[:, :, 0]).any(axis=1)] = xs.shape[1]
        return np.stack([
            np.mean(xs[i, np.maximum(0, nan_onset[i] - k) : nan_onset[i]], axis=0, keepdims=True)
            for i in range(xs.shape[0])
        ])
    return agg_fn

TRAJECTORY_AGG_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda xs: np.nanmean(xs, axis=1, keepdims=True),
    "max": lambda xs: np.nanmax(xs, axis=1, keepdims=True),
    "last_frame": lambda xs: xs[np.arange(xs.shape[0]), np.isnan(xs[:, :, 0]).argmax(axis=1) - 1][:, None, :],
}

TRAJECTORY_META_AGG_FNS: dict[str, Callable[[Any], Callable[[np.ndarray], np.ndarray]]] = {
    "mean_last_k": make_agg_fn_mean_last_k,
}


def aggregate_state_trajectory(trajectory: list[np.ndarray],
                               agg_fn_spec: Union[str, tuple[str, Any]],
                               keepdims=False) -> list[np.ndarray]:
    """
    Aggregate over time in the state trajectories returned by `prepare_state_trajectory`.
    """
    if isinstance(agg_fn_spec, tuple):
        agg_fn_name, agg_fn_args = agg_fn_spec
        agg_fn = TRAJECTORY_META_AGG_FNS[agg_fn_name](agg_fn_args)
    else:
        agg_fn = TRAJECTORY_AGG_FNS[agg_fn_spec]

    ret = [agg_fn(traj) for traj in trajectory]
    if not keepdims:
        ret = [traj.squeeze(1) for traj in ret]

    return ret