"""
Defines methods for training and evaluating word recognition
classifiers on model embeddings.
"""

from dataclasses import dataclass
import logging
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
import transformers
from tqdm.auto import tqdm

from src.analysis import state_space as ss
from src.datasets.speech_equivalence import SpeechHiddenStateDataset


L = logging.getLogger(__name__)


def prepare_trajectories(embeddings, state_space_spec, config):
    trajectory = ss.prepare_state_trajectory(
        embeddings, state_space_spec, pad=np.nan)
    
    # aggregate the trajectory
    featurization = getattr(config.embeddings, "featurization", None)
    if featurization is not None:
        trajectory = ss.aggregate_state_trajectory(
            trajectory, state_space_spec, tuple(featurization)
        )
    flat_traj, flat_traj_src = ss.flatten_trajectory(trajectory)
    max_num_frames = flat_traj_src[:, 2].max() + 1

    # Group by frame
    flat_trajs_by_frame = []
    for frame in range(max_num_frames):
        mask = flat_traj_src[:, 2] == frame
        flat_trajs_by_frame.append((flat_traj[mask], flat_traj_src[mask, 0], flat_traj_src[mask, 1]))

    return flat_trajs_by_frame


class MyDataset(Dataset):
    def __init__(self, idxs, embeddings, labels, label_instance_idxs):
        self.idxs = idxs
        self.embeddings = embeddings
        self.labels = labels
        self.label_instance_idxs = label_instance_idxs

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"idxs": self.idxs[idx],
                "inputs": self.embeddings[idx],
                "labels": self.labels[idx],
                "label_instance_idxs": self.label_instance_idxs[idx]}
    

@dataclass
class MyModelOutput(transformers.utils.ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    

class MyModel(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, inputs, labels=None, **kwargs):
        logits = self.fc(inputs)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return MyModelOutput(
            loss=loss,
            logits=logits
        )


def prepare_dataset(embeddings, labels, label_instance_idxs,
                    num_splits=5) -> list[tuple[Dataset, Dataset]]:
    assert embeddings.shape[0] == labels.shape[0] == label_instance_idxs.shape[0]

    # l2 norm
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # do stratified k-fold split
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    datasets = []
    for train_idx, eval_idx in skf.split(embeddings, labels):
        train_embeddings, train_labels, train_label_instance_idxs = \
            embeddings[train_idx], labels[train_idx], label_instance_idxs[train_idx]
        eval_embeddings, eval_labels, eval_label_instance_idxs = \
            embeddings[eval_idx], labels[eval_idx], label_instance_idxs[eval_idx]

        datasets.append((MyDataset(torch.tensor(train_idx).long(),
                                   torch.tensor(train_embeddings).float(),
                                   torch.tensor(train_labels).long(),
                                   torch.tensor(train_label_instance_idxs).long()),
                         MyDataset(torch.tensor(eval_idx).long(),
                                   torch.tensor(eval_embeddings).float(),
                                   torch.tensor(eval_labels).long(),
                                   torch.tensor(eval_label_instance_idxs).long())))

    return datasets


def compute_metrics(p: transformers.EvalPrediction):
    labels, idxs, instance_idxs = p.label_ids
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}


def train(config: DictConfig):
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            L.error("CUDA is not available. Falling back to CPU.")
            config["device"] = "cpu"

    hidden_states = SpeechHiddenStateDataset.from_hdf5(config.base_model.hidden_state_path)
    state_space_spec = ss.StateSpaceAnalysisSpec.from_hdf5(config.analysis.state_space_specs_path, "word")
    assert state_space_spec.is_compatible_with(hidden_states)
    embeddings = np.load(config.model.embeddings_path)

    # Subsample state space according to config
    L.info(f"Keeping top {config.recognition_model.evaluation.keep_top_k} labels (out of {len(state_space_spec.labels)})")
    state_space_spec = state_space_spec.keep_top_k(config.recognition_model.evaluation.keep_top_k)
    state_space_spec = state_space_spec.subsample_instances(config.recognition_model.evaluation.subsample_instances)

    trajectories = prepare_trajectories(embeddings, state_space_spec, config.recognition_model)
    datasets = {
        frame_idx: prepare_dataset(*traj, num_splits=config.recognition_model.evaluation.num_stratified_splits)
        for frame_idx, traj in enumerate(trajectories)
    }
    all_labels = state_space_spec.labels

    device = torch.device(config.device)
    def make_model():
        return MyModel(embeddings.shape[1], len(all_labels)).to(device)

    # Overrides -- hacky because we're pulling a config from the main model config
    config.training_args.per_device_train_batch_size = config.recognition_model.evaluation.train_batch_size
    config.training_args.num_train_epochs = config.recognition_model.evaluation.num_train_epochs
    config.training_args.label_names = ["labels", "idxs", "label_instance_idxs"]
    config.training_args.learning_rate = config.recognition_model.optimizer.lr

    training_args = transformers.TrainingArguments(
        use_cpu=config.device == "cpu",
        output_dir=HydraConfig.get().runtime.output_dir,
        logging_dir=Path(HydraConfig.get().runtime.output_dir) / "logs",
        per_device_eval_batch_size=config.recognition_model.evaluation.eval_batch_size,
        eval_accumulation_steps=5,
        # max_steps=max_training_steps,
        **OmegaConf.to_object(config.training_args))
    
    callbacks = []
    if "callbacks" in config.trainer:
        callbacks = [instantiate(c) for c in config.trainer.callbacks]
    trainer_config = dict(config.trainer)
    trainer_config.pop("callbacks", None)
    trainer_mode = trainer_config.pop("mode", "train")
    hparam_config = trainer_config.pop("hyperparameter_search", None)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    for frame_idx, datasets in tqdm(datasets.items(), unit="frame"):
        all_test_evaluations, all_test_outputs = [], []
        for split_idx, (train_dataset, test_dataset) in enumerate(datasets):
            model = make_model()
            model_dir = output_dir / f"frame_{frame_idx}-split_{split_idx}"
            training_args.output_dir = str(model_dir)
            training_args.logging_dir = str(model_dir / "logs")

            # create a validation dataset from 10% of the training dataset
            train_dataset, eval_dataset = random_split(
                train_dataset, [len(train_dataset) - len(train_dataset) // 10,
                                len(train_dataset) // 10])

            trainer = transformers.Trainer(
                args=training_args,
                model=model,
                callbacks=callbacks,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                **trainer_config)
    
            trainer.train()

            all_test_evaluations.append(trainer.evaluate(test_dataset))

            test_output = trainer.predict(test_dataset)
            assert test_output.predictions is not None
            assert test_output.label_ids is not None
            model_logits: np.ndarray = test_output.predictions
            predicted_label_idx = model_logits.argmax(axis=1)

            model_probabilities = torch.nn.functional.softmax(torch.tensor(model_logits), dim=1).numpy()
            model_entropy = -np.sum(model_probabilities * np.log(model_probabilities), axis=1)
            # probability of top label
            predicted_probability = model_probabilities[np.arange(model_probabilities.shape[0]), predicted_label_idx]
            # probability of GT label
            gt_label_probability = model_probabilities[np.arange(model_probabilities.shape[0]), test_output.label_ids[0]]

            all_test_outputs.append({
                "predicted_label_idx": predicted_label_idx,
                "predicted_probability": predicted_probability,

                "gt_label_probability": gt_label_probability,

                "entropy": model_entropy,

                "label_idx": test_output.label_ids[0],
                "label_instance_idx": test_output.label_ids[2],
                "example_idx": test_output.label_ids[1],
            })

        predictions_df = pd.concat([pd.DataFrame(e) for e in all_test_outputs], ignore_index=True) \
            .sort_values("example_idx")
        predictions_df["label"] = predictions_df.label_idx.map(dict(enumerate(all_labels)))
        predictions_df["predicted_label"] = predictions_df.predicted_label_idx.map(dict(enumerate(all_labels)))
        predictions_df["correct"] = predictions_df.label_idx == predictions_df.predicted_label_idx
        predictions_df.to_csv(output_dir / f"predictions-frame_{frame_idx}.csv", index=False)
