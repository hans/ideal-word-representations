"""
State space analysis tools for integrator models.
"""

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
import torch

from src.datasets.speech_equivalence import SpeechEquivalenceDataset
from src.models.integrator import ContrastiveEmbeddingModel, compute_embeddings



def prepare_word_trajectory_spec(
        dataset: SpeechEquivalenceDataset,
        target_words: list[tuple[str, ...]],
) -> list[list[tuple[int, int]]]:
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

    return frame_bounds



def prepare_state_trajectory(
        embeddings: np.ndarray,
        trajectory_spec: list[list[tuple[int, int]]],
) -> list[np.ndarray]:
    """
    Prepare the state trajectory for the given dataset and model embeddings.
    """
    max_num_frames = max(max(end - start + 1 for start, end in trajectory_spec)
                         for trajectory_spec in trajectory_spec)
    ret = []

    for i, frame_spec in enumerate(trajectory_spec):
        num_instances = len(frame_spec)
        trajectory_frames = np.zeros((num_instances, max_num_frames, embeddings.shape[1]))
        for j, (start, end) in enumerate(frame_spec):
            trajectory_frames[j, :end - start + 1] = embeddings[start:end + 1]

            # Fill on right
            if end - start + 1 < max_num_frames:
                trajectory_frames[j, end - start + 1:] = embeddings[end]

        ret.append(trajectory_frames)

    return ret