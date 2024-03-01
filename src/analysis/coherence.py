import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm


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
            mask = sample_lengths_i >= j
            if mask.sum() <= 1:
                break
    
            within_distance[i, j] = get_mean_distance(samples_i[mask, j, :], samples_i[mask, j, :], metric=metric)
    
            lengths_i_masked = sample_lengths_i[mask]
            within_distance_offset[i, j] = get_mean_distance(samples_i[mask, lengths_i_masked - j, :],
                                                             samples_i[mask, lengths_i_masked - j, :],
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
    
    between_distances = np.zeros((len(trajectory), trajectory[0].shape[1], num_samples)) * np.nan
    between_distances_offset = np.zeros((len(trajectory), trajectory[0].shape[1], num_samples)) * np.nan
    for i, between_samples_i in enumerate(tqdm(between_samples)):
        traj_i, lengths_i = trajectory[i], lengths[i]
        if traj_i.shape[0] > max_num_instances:
            idxs = np.random.choice(traj_i.shape[0], size=max_num_instances, replace=False)
            traj_i = traj_i[idxs]
            lengths_i = lengths_i[idxs]
        
        for j, between_sample in enumerate(between_samples_i):
            traj_j, lengths_j = trajectory[between_sample], lengths[between_sample]
            if traj_j.shape[0] > max_num_instances:
                idxs = np.random.choice(traj_j.shape[0], size=max_num_instances, replace=False)
                traj_j, lengths_j = traj_j[idxs], lengths_j[idxs]
            
            for k in range(trajectory[0].shape[1]):
                mask_i = lengths_i >= k
                mask_j = lengths_j >= k
                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    break
                
                between_distances[i, k, j] = get_mean_distance(traj_i[mask_i, k, :], traj_j[mask_j, k, :], metric=metric).mean()
    
                lengths_i_masked = lengths_i[mask_i]
                lengths_j_masked = lengths_j[mask_j]
                between_distances_offset[i, k, j] = get_mean_distance(
                    traj_i[mask_i][np.arange(mask_i.sum()), lengths_i_masked - k, :],
                    traj_j[mask_j][np.arange(mask_j.sum()), lengths_j_masked - k, :],
                    metric=metric
                ).mean()

    return between_distances, between_distances_offset


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