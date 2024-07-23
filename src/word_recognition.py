"""
Defines methods for training and evaluating word recognition
classifiers on model embeddings.
"""

import logging

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.analysis import state_space as ss
from src.datasets.speech_equivalence import SpeechHiddenStateDataset


L = logging.getLogger(__name__)


def prepare_trajectories(embeddings, state_space_spec, config):
    trajectory = ss.prepare_state_trajectory(
        embeddings, state_space_spec, pad=np.nan)
    
    # aggregate the trajectory
    featurization = getattr(config.embeddings, "featurization", None)
    if False:  # featurization is not None:  # DEV
        trajectory = ss.aggregate_state_trajectory(
            trajectory, state_space_spec, tuple(featurization)
        )
    flat_traj, flat_traj_src = ss.flatten_trajectory(trajectory)
    max_num_frames = flat_traj_src[:, 2].max() + 1

    # Group by frame
    flat_trajs_by_frame = []
    for frame in range(max_num_frames):
        mask = flat_traj_src[:, 2] == frame
        flat_trajs_by_frame.append((flat_traj[mask], flat_traj_src[mask]))

    return flat_trajs_by_frame


def train(config: DictConfig):
    if config.device == "cuda":
        if not torch.cuda.is_available():
            L.error("CUDA is not available. Falling back to CPU.")
            config.device = "cpu"

    recognition_config = config.recognition_model

    hidden_states = SpeechHiddenStateDataset.from_hdf5(config.base_model.hidden_state_path)
    state_space_spec: ss.StateSpaceAnalysisSpec = torch.load(config.analysis.state_space_specs_path)["word"]
    assert state_space_spec.is_compatible_with(hidden_states)
    embeddings = np.load(config.model.embeddings_path)

    trajectories = prepare_trajectories(embeddings, state_space_spec, recognition_config)

    # DEV just work on the first
    flat_traj, flat_traj_src = trajectories[0]

    label2idx = {label: idx for idx, label in enumerate(np.unique(flat_traj_src[:, 0]))}
    num_words = len(label2idx)

    flat_traj = torch.tensor(flat_traj, dtype=torch.float)
    labels = torch.tensor(flat_traj_src[:, 0]).long()
    dataset = TensorDataset(flat_traj, labels)
    train_dataset, eval_dataset = random_split(
        dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    
    train_loader = DataLoader(train_dataset,
                              batch_size=recognition_config.evaluation.train_batch_size,
                              shuffle=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=recognition_config.evaluation.eval_batch_size,
                             shuffle=False)

    model = nn.Linear(flat_traj.shape[1], num_words)
    model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = instantiate(recognition_config.optimizer, model.parameters())

    for epoch in range(recognition_config.evaluation.num_train_epochs):
        for i, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch = [b.to(config.device) for b in batch]
            x, y = batch
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            if i % 100 == 0:
                print(i, loss.item())

            loss.backward()
            optimizer.step()