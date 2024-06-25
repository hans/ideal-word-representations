import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from src.analysis.state_space import StateSpaceAnalysisSpec, prepare_state_trajectory


L = logging.getLogger(__name__)

def get_mean_distance(samp1, samp2, metric=None):
    distances = cdist(samp1, samp2, metric=metric)
    distances = np.triu(distances, k=1)
    return distances[distances != 0].mean()


def estimate_within_distance(trajectory, lengths,
                             state_space_spec,
                             num_samples=50,
                             max_num_instances=50,
                             metric=None):
    within_distance = np.zeros((len(trajectory), trajectory[0].shape[1])) * np.nan
    within_distance_offset = np.zeros((len(trajectory), trajectory[0].shape[1])) * np.nan
    for i, (trajectory_i, lengths_i) in enumerate(zip(tqdm(trajectory), lengths)):
        within_distance_i = []
        num_instances, num_frames, num_dims = trajectory_i.shape
    
        num_instances_limited = min(num_instances, max_num_instances)
        idxs = np.random.choice(num_instances, size=num_instances_limited, replace=False)
        samples_i, sample_lengths_i = trajectory_i[idxs], lengths_i[idxs]
        for j in range(num_frames):
            mask = sample_lengths_i > j
            if mask.sum() <= 1:
                break
    
            within_distance[i, j] = get_mean_distance(samples_i[mask, j, :], samples_i[mask, j, :], metric=metric)
    
            lengths_i_masked = sample_lengths_i[mask]
            idx_for_offset = lengths_i_masked - j - 1
            within_distance_offset[i, j] = get_mean_distance(samples_i[mask, idx_for_offset, :],
                                                             samples_i[mask, idx_for_offset, :],
                                                             metric=metric)

    return within_distance, within_distance_offset


def estimate_between_distance(trajectory, lengths,
                              state_space_spec,
                              between_samples=None,
                              num_samples=50, max_num_instances=50,
                              metric=None):
    if num_samples > len(trajectory) - 1:
        L.warn(f"Reducing num_samples to {len(trajectory) - 1} given limited data")
        num_samples = len(trajectory) - 1

    if between_samples is None:
        between_samples = [np.random.choice(list(range(idx)) + list(range(idx + 1, len(trajectory))),
                                            num_samples, replace=False)
                           for idx in range(len(trajectory))]
    else:
        # num_samples: max number samples
        num_samples = max(len(between_samples_i) for between_samples_i in between_samples)

    sample_cache = {}
    def fetch_sample(idx):
        if idx in sample_cache:
            return sample_cache[idx]
        else:
            traj_j, lengths_j = trajectory[idx], lengths[idx]
            if traj_j.shape[0] > max_num_instances:
                idxs = np.random.choice(traj_j.shape[0], size=max_num_instances, replace=False)
                traj_j, lengths_j = traj_j[idxs], lengths_j[idxs]

            # prepare offset-aligned representation
            traj_j_reverse = np.array([
                np.pad(traj_j[l, :lengths_j[l], :][:, ::-1],
                        ((0, traj_j.shape[1] - lengths_j[l]), (0, 0)),
                        constant_values=np.nan)
                for l in range(traj_j.shape[0])])

            sample_cache[idx] = traj_j, traj_j_reverse, lengths_j
            return traj_j, traj_j_reverse, lengths_j

    # num_words * num_frames * num_samples
    # cell (i, j, k) is the mean distance between tokens of word i at frame j and word k at frame j
    between_distances = np.zeros((len(trajectory), trajectory[0].shape[1], num_samples)) * np.nan
    # num_words * num_frames * num_samples
    # cell (i, j, k) is the mean distance between tokens of word i at frame j and word k at frame j,
    # where j is [0, num_frames) distance from the end of the word
    between_distances_offset = np.zeros((len(trajectory), trajectory[0].shape[1], num_samples)) * np.nan
    for i, between_samples_i in enumerate(tqdm(between_samples)):
        traj_i, traj_i_reverse, lengths_i = fetch_sample(i)
        
        for j, between_sample in enumerate(between_samples_i):
            traj_j, traj_j_reverse, lengths_j = fetch_sample(between_sample)
            
            for k in range(trajectory[0].shape[1]):
                mask_i = lengths_i > k
                mask_j = lengths_j > k
                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    break
                
                between_distances[i, k, j] = get_mean_distance(traj_i[mask_i, k, :], traj_j[mask_j, k, :], metric=metric).mean()
    
                offset_distance_k = lengths_i[mask_i] - k - 1
                between_distances_offset[i, offset_distance_k, j] = get_mean_distance(
                    traj_i_reverse[mask_i, k, :], traj_j_reverse[mask_j, k, :],
                    metric=metric).mean()

    return between_distances, between_distances_offset


def estimate_within_between_distance_by_cut_index(
        state_space_spec, model_representations, cut_level,
        num_samples=50, max_num_instances=50, metric=None):
    state_space_spec_expanded = state_space_spec.expand_by_cut_index(cut_level)
    expanded_label2idx = {label: idx for idx, label in enumerate(state_space_spec_expanded.labels)}

    trajectory_expanded = prepare_state_trajectory(
        model_representations, state_space_spec_expanded, pad=np.nan)
    trajectory_expanded_lengths = [np.isnan(traj_i[:, :, 0]).argmax(axis=1) for traj_i in trajectory_expanded]

    between_samples = [np.random.choice(list(range(idx)) + list(range(idx + 1, len(state_space_spec.labels))),
                                    size=50, replace=False)
                   for idx in range(len(state_space_spec.labels))]

    within_distances = {label: [] for label in state_space_spec_expanded.labels}
    between_distances = {label: [] for label in state_space_spec_expanded.labels}

    cuts_df = state_space_spec.cuts.xs("phoneme", level="level").copy()
    cuts_df["num_frames"] = cuts_df.offset_frame_idx - cuts_df.onset_frame_idx
    cuts_df["idx_in_level"] = cuts_df.groupby(["label", "instance_idx"]).cumcount()
    frames_per_cut = cuts_df.groupby("idx_in_level").apply(lambda xs: np.percentile(xs.num_frames, 95)).astype(int)
    
    for i, (label, between_samples_i) in enumerate(zip(tqdm(state_space_spec.labels), between_samples)):
        between_labels_i = [state_space_spec.labels[j] for j in between_samples_i]

        for k in frames_per_cut.index:
            max_num_frames_k = frames_per_cut.loc[k]
            if (label, k) not in expanded_label2idx:
                continue

            expanded_label_idx = expanded_label2idx[label, k]
            trajectory_ik = trajectory_expanded[expanded_label_idx]
            lengths_ik = trajectory_expanded_lengths[expanded_label_idx]

            if trajectory_ik.shape[0] > max_num_instances:
                idxs = np.random.choice(trajectory_ik.shape[0], max_num_instances, replace=False)
                trajectory_ik = trajectory_ik[idxs]
                lengths_ik = lengths_ik[idxs]

            expanded_between_labels_idxs_ik = [expanded_label2idx[label, k] for label in between_labels_i
                                               if (label, k) in expanded_label2idx]

            for between_ikl in expanded_between_labels_idxs_ik:
                trajectory_ikl = trajectory_expanded[between_ikl]
                lengths_ikl = trajectory_expanded_lengths[between_ikl]

                if trajectory_ikl.shape[0] > max_num_instances:
                    idxs = np.random.choice(trajectory_ikl.shape[0], max_num_instances, replace=False)
                    trajectory_ikl = trajectory_ikl[idxs]
                    lengths_ikl = lengths_ikl[idxs]

                for m in range(min(trajectory_ik.shape[1], max_num_frames_k)):
                    mask_ik = lengths_ik > m
                    mask_ikl = lengths_ikl > m
                    if mask_ik.sum() == 0 or mask_ikl.sum() == 0:
                        break
        
                    assert not (np.isnan(trajectory_ikl[mask_ikl, m, :]).any())
                    assert not (np.isnan(trajectory_ik[mask_ik, m, :]).any())

                    between_distances[label, k].append(
                        get_mean_distance(trajectory_ik[mask_ik, m, :],
                                          trajectory_ikl[mask_ikl, m, :], metric=metric)
                    )

            # Estimate within-distance
            for m in range(min(trajectory_ik.shape[1], max_num_frames_k)):
                mask_ik = lengths_ik > m
                if mask_ik.sum() <= 1:
                    break

                within_distances[label, k].append(
                    get_mean_distance(trajectory_ik[mask_ik, m, :],
                                      trajectory_ik[mask_ik, m, :], metric=metric)
                )


    within_distances = pd.DataFrame(
        [np.nanmean(distances) for distances in within_distances.values()],
        columns=["distance"],
        index=pd.MultiIndex.from_tuples(within_distances.keys(), names=["label", "cut_idx"]))
    between_distances = pd.DataFrame(
        [np.nanmean(distances) for distances in between_distances.values()],
        columns=["distance"],
        index=pd.MultiIndex.from_tuples(between_distances.keys(), names=["label", "cut_idx"]))
    
    return pd.concat([within_distances.assign(type="within"), between_distances.assign(type="between")])
    


def estimate_category_within_between_distance(trajectory, lengths,
                                              category_assignment,
                                              labels=None,
                                              num_samples=50, max_num_instances=50,
                                              metric=None):
    # """
    # Returns:
    # - idxs: indices of items used to compute within- and between-category distances
    # - matched_distances: pairwise distance matrix, aligned to trajectory onset
    # - matched_distances_offset: pairwise distance matrix, aligned to trajectory offset and counting backwards
    # - mismatched_distances
    # - mismatched_distances_offset
    # """

    assert len(trajectory) == len(category_assignment) == len(lengths)

    # Only estimate distances for items with category assignment
    item_mask = np.array([category is not None for category in category_assignment])

    category_assignment = [category for category in category_assignment if category is not None]
    trajectory = [trajectory[idx] for idx in item_mask.nonzero()[0]]
    lengths = [lengths[idx] for idx in item_mask.nonzero()[0]]
    labels = [labels[idx] for idx in item_mask.nonzero()[0]]

    matched_samples, mismatched_samples = [], []
    for i, category_i in enumerate(category_assignment):
        matched_items_i = [j for j, category_j in enumerate(category_assignment)
                           if category_j == category_i and j != i]
        mismatched_items_i = [j for j, category_j in enumerate(category_assignment)
                              if category_j != category_i]
        
        matched_samples.append(np.random.choice(matched_items_i, min(num_samples, len(matched_items_i)), replace=False))
        mismatched_samples.append(np.random.choice(mismatched_items_i, min(num_samples, len(mismatched_items_i)), replace=False))
    
    matched_distances, matched_distances_offset = estimate_between_distance(
        trajectory, lengths, state_space_spec=None, between_samples=matched_samples,
        num_samples=num_samples, max_num_instances=max_num_instances, metric=metric
    )

    mismatched_distances, mismatched_distances_offset = estimate_between_distance(
        trajectory, lengths, state_space_spec=None, between_samples=mismatched_samples,
        num_samples=num_samples, max_num_instances=max_num_instances, metric=metric
    )

    if isinstance(labels, (list, tuple)):
        labels = ["".join(label) for label in labels]
    matched_df = pd.DataFrame(np.nanmean(matched_distances, axis=-1), index=pd.Index(labels, name="label")) \
        .reset_index().melt(id_vars=["label"], var_name="frame", value_name="distance")
    mismatched_df = pd.DataFrame(np.nanmean(mismatched_distances, axis=-1), index=pd.Index(labels, name="label")) \
        .reset_index().melt(id_vars=["label"], var_name="frame", value_name="distance")
    merged_df = pd.concat([matched_df.assign(type="matched"), mismatched_df.assign(type="mismatched")])

    matched_offset_df = pd.DataFrame(np.nanmean(matched_distances_offset, axis=-1), index=pd.Index(labels, name="label")) \
        .reset_index().melt(id_vars=["label"], var_name="frame", value_name="distance")
    mismatched_offset_df = pd.DataFrame(np.nanmean(mismatched_distances_offset, axis=-1), index=pd.Index(labels, name="label")) \
        .reset_index().melt(id_vars=["label"], var_name="frame", value_name="distance")
    merged_offset_df = pd.concat([matched_offset_df.assign(type="matched"), mismatched_offset_df.assign(type="mismatched")])

    return merged_df, merged_offset_df