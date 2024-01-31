"""
State space analysis tools for integrator models.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
import torch

from src.datasets.speech_equivalence import SpeechEquivalenceDataset
from src.models.integrator import ContrastiveEmbeddingModel, compute_embeddings



@dataclass
class StateSpaceAnalysisSpec:

    # Number of frames in the dataset associated with this state space spec.
    # Used for validation.
    total_num_frames: int

    labels: list[str]

    # Analyze K categories of N state space trajectories.
    target_frame_spans: list[list[tuple[int, int]]]

    def __post_init__(self):
        assert len(self.target_frame_spans) == len(self.labels)

    def is_compatible_with(self, dataset: SpeechEquivalenceDataset) -> bool:
        return self.total_num_frames == dataset.hidden_state_dataset.num_frames


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
) -> list[np.ndarray]:
    """
    Prepare the state trajectory for the given dataset and model embeddings.
    """
    max_num_frames = max(max(end - start + 1 for start, end in trajectory_spec)
                         for trajectory_spec in spec.target_frame_spans)
    ret = []

    for i, frame_spec in enumerate(spec.target_frame_spans):
        num_instances = len(frame_spec)
        trajectory_frames = np.zeros((num_instances, max_num_frames, embeddings.shape[1]))
        for j, (start, end) in enumerate(frame_spec):
            trajectory_frames[j, :end - start + 1] = embeddings[start:end + 1]

            # Fill on right
            if end - start + 1 < max_num_frames:
                trajectory_frames[j, end - start + 1:] = embeddings[end]

        ret.append(trajectory_frames)

    return ret