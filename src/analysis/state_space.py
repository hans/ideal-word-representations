"""
State space analysis tools for integrator models.
"""

from copy import deepcopy
from functools import cached_property, wraps
from dataclasses import dataclass
from typing import Optional, Union, Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

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
            assert self.cuts.index.names == ["label", "instance_idx", "level"]
            assert set(self.cuts.columns) >= {"description", "onset_frame_idx", "offset_frame_idx"}
            assert set(self.cuts.index.get_level_values("label")) <= set(self.labels)

            assert (self.cuts.onset_frame_idx < self.total_num_frames).all()
            assert (self.cuts.offset_frame_idx < self.total_num_frames).all()

            # check consistency of cuts + target frame spans by merge-and-compare. faster than a for loop! :)
            cuts_validity_check = pd.merge(self.cuts.reset_index("level", drop=True), self.target_frame_spans_df,
                                           left_index=True, right_on=["label", "instance_idx"])
            assert (cuts_validity_check.onset_frame_idx >= cuts_validity_check.start_frame).all()
            assert (cuts_validity_check.offset_frame_idx <= cuts_validity_check.end_frame).all()

    @property
    def target_frame_spans_df(self) -> pd.DataFrame:
        """
        Return a dataframe representation of target frame spans. The keys `label` and `instance_idx`
        are comparable to the keys in `cuts`.
        """
        return pd.DataFrame([
            (label, instance_idx, start, end)
            for label, frame_spans in zip(self.labels, self.target_frame_spans)
            for instance_idx, (start, end) in enumerate(frame_spans)
        ], columns=["label", "instance_idx", "start_frame", "end_frame"])

    @property
    def label_counts(self):
        return pd.Series([len(spans) for spans in self.target_frame_spans], index=self.labels)

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
    
    def subsample_instances(self, max_instances_per_label: int, random=True):
        """
        Return a copy of the current state space analysis spec with at most
        `max_instances_per_label` instances per label.
        """
        new_target_frame_spans, new_cuts = [], {}
        for label, frame_spans in zip(self.labels, self.target_frame_spans):
            if len(frame_spans) <= max_instances_per_label:
                keep_idxs = np.arange(len(frame_spans))
            else:
                if random:
                    keep_idxs = np.random.choice(len(frame_spans), max_instances_per_label, replace=False)
                else:
                    keep_idxs = np.arange(max_instances_per_label)

            new_target_frame_spans.append([frame_spans[i] for i in keep_idxs])
            if self.cuts is not None:
                new_cuts_i = self.cuts.loc[label]
                new_cuts_i = new_cuts_i[new_cuts_i.index.get_level_values("instance_idx").isin(keep_idxs)]

                # Relabel instance_idx
                new_cuts_i = new_cuts_i.reset_index("instance_idx")
                new_cuts_i["instance_idx"] = new_cuts_i.instance_idx.map({old_idx: new_idx for new_idx, old_idx in enumerate(keep_idxs)})
                new_cuts_i = new_cuts_i.set_index("instance_idx", append=True).reorder_levels(["instance_idx", "level"])
                new_cuts[label] = new_cuts_i

        new_cuts = pd.concat(new_cuts, names=["label"]).sort_index() if self.cuts is not None else None

        return StateSpaceAnalysisSpec(
            total_num_frames=self.total_num_frames,
            labels=self.labels,
            target_frame_spans=new_target_frame_spans,
            cuts=new_cuts,
        )

    def groupby(self, grouper) -> Iterable[tuple[Any, "StateSpaceAnalysisSpec"]]:
        if self.cuts is None:
            raise ValueError("Cannot groupby without cuts")

        for group_key, group_df in self.cuts.groupby(grouper):
            new_target_frame_spans = []
            new_labels = []
            new_cut_idxs = []
            for label, label_df in group_df.groupby("label"):
                label_idx = self.labels.index(label)
                new_labels.append(label)
                new_label_spans = []
                for instance_idx, instance_df in label_df.groupby("instance_idx"):
                    new_label_spans.append(self.target_frame_spans[label_idx][instance_idx])
                    new_cut_idxs.append((label, instance_idx))
                new_target_frame_spans.append(new_label_spans)

            cut_indexer = pd.DataFrame(new_cut_idxs, columns=["label", "instance_idx"])
            cut_indexer["new_instance_idx"] = cut_indexer.groupby("label").cumcount()
            new_cuts = pd.merge(self.cuts.reset_index(), cut_indexer, on=["label", "instance_idx"])
            # relabel instance_idx
            new_cuts["instance_idx"] = new_cuts.new_instance_idx
            new_cuts = new_cuts.drop(columns=["new_instance_idx"]).set_index(["label", "instance_idx", "level"])
            
            yield group_key, StateSpaceAnalysisSpec(
                total_num_frames=self.total_num_frames,
                labels=new_labels,
                target_frame_spans=new_target_frame_spans,
                cuts=new_cuts,
            )
    
    def keep_top_k(self, k=100):
        """
        Return a copy of the current state space analysis spec with only the top `k`
        labels by instance count.
        """
        if k >= len(self.labels):
            return deepcopy(self)
        top_k_labels = self.label_counts.sort_values(ascending=False).head(k).index
        return self.drop_labels(drop_names=set(self.labels) - set(top_k_labels))
    
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


def make_simple_agg_fn(fn):
    @wraps(fn)
    def agg_fn(xs, *args, **kwargs):
        return fn(xs)
    return agg_fn

def make_agg_fn_mean_last_k(k):
    def agg_fn(xs, *args, **kwargs):
        nan_onset = np.isnan(xs[:, :, 0]).argmax(axis=1)
        # if there are no nans, set nan_onset to len
        nan_onset[~np.isnan(xs[:, :, 0]).any(axis=1)] = xs.shape[1]
        return np.stack([
            np.mean(xs[i, np.maximum(0, nan_onset[i] - k) : nan_onset[i]], axis=0, keepdims=True)
            for i in range(xs.shape[0])
        ])
    return agg_fn

class AggMeanWithinCut:
    def __init__(self, cut_level: str):
        self.cut_level = cut_level
    def __call__(self, trajectory_i: np.ndarray, state_space_spec: StateSpaceAnalysisSpec,
                 label_idx: int, pad: Union[str, float] = "last") -> np.ndarray:
        if state_space_spec.cuts is None:
            raise ValueError("No cuts available to aggregate within")
        
        label = state_space_spec.labels[label_idx]
        cuts_df = state_space_spec.cuts.loc[label]
        try:
            cuts_df = cuts_df.xs(self.cut_level, level="level")
        except KeyError:
            raise ValueError(f"Cut level {self.cut_level} not found in cuts")

        assert set(cuts_df.index.get_level_values("instance_idx")) == set(np.arange(len(trajectory_i)))

        max_num_cuts: int = cuts_df.groupby("instance_idx").size().max()  # type: ignore
        new_trajs = np.zeros((len(trajectory_i), max_num_cuts, trajectory_i.shape[2]), dtype=float)
        
        for instance_idx, instance_cuts in cuts_df.groupby("instance_idx"):
            instance_frame_start, _ = state_space_spec.target_frame_spans[label_idx][instance_idx]
            for cut_idx, (_, cut) in enumerate(instance_cuts.iterrows()):
                # get index of cut relative to instance onset frame
                cut_start: int = cut.onset_frame_idx - instance_frame_start
                cut_end: int = cut.offset_frame_idx - instance_frame_start

                new_trajs[instance_idx, cut_idx] = np.mean(trajectory_i[instance_idx, cut_start:cut_end], axis=0, keepdims=True)

            if pad == "last":
                pad_value = new_trajs[instance_idx, cut_idx]
            elif isinstance(pad, str):
                raise ValueError(f"Invalid pad value {pad}; use `last` or a float")
            else:
                pad_value = pad

            if cut_idx < max_num_cuts - 1:
                new_trajs[instance_idx, cut_idx + 1:] = pad_value

        return new_trajs
        

TRAJECTORY_AGG_FNS: dict[str, Callable] = {
    "mean": lambda xs: np.nanmean(xs, axis=1, keepdims=True),
    "max": lambda xs: np.nanmax(xs, axis=1, keepdims=True),
    "last_frame": lambda xs: xs[np.arange(xs.shape[0]), np.isnan(xs[:, :, 0]).argmax(axis=1) - 1][:, None, :],
}
TRAJECTORY_AGG_FNS = {k: make_simple_agg_fn(v) for k, v in TRAJECTORY_AGG_FNS.items()}

TRAJECTORY_META_AGG_FNS: dict[str, Callable] = {
    "mean_last_k": make_agg_fn_mean_last_k,
    "mean_within_cut": AggMeanWithinCut,
}


def aggregate_state_trajectory(trajectory: list[np.ndarray],
                               state_space_spec: StateSpaceAnalysisSpec,
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

    ret = [agg_fn(traj, state_space_spec=state_space_spec,
                  label_idx=idx)
           for idx, traj in enumerate(tqdm(trajectory, unit="label", desc="Aggregating", leave=False))]
    if not keepdims:
        ret = [traj.squeeze(1) for traj in ret]

    return ret


def flatten_trajectory(trajectory: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a flattened representation of all state space trajectories.

    Returns:
    - all_trajectories: (N, D) array of all state space trajectories
    - all_trajectories_src: (N, 3) array
        describes, for each row of `all_trajectories`, the originating label, instance idx, and frame idx
    """

    all_trajectories = np.concatenate([
        traj_i.reshape(-1, traj_i.shape[-1]) for traj_i in trajectory
    ])
    all_trajectories_src = np.concatenate([
        np.array([(label_idx, instance_idx, frame_idx) for label_idx, traj_i in enumerate(trajectory)
                 for instance_idx, frame_idx in np.ndindex(traj_i.shape[:2])])
    ])
    assert all_trajectories.shape[0] == all_trajectories_src.shape[0]
    assert all_trajectories_src.shape[1] == 3
    assert all_trajectories_src[:, 0].max() == len(trajectory) - 1

    # TODO this assumes NaN padding
    retain_idxs = ~np.isnan(all_trajectories).any(axis=1)
    all_trajectories = all_trajectories[retain_idxs]
    all_trajectories_src = all_trajectories_src[retain_idxs]

    return all_trajectories, all_trajectories_src