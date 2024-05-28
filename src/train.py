from dataclasses import replace
import logging 
from pathlib import Path

import datasets
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from ray import tune
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import transformers

from src.datasets.speech_equivalence import SpeechEquivalenceDataset, SpeechHiddenStateDataset
from src.models import integrator

L = logging.getLogger(__name__)


def make_model_init(config: integrator.ContrastiveEmbeddingModelConfig, device="cpu"):
    def model_init(trial):
        if trial is not None:
            config_trial = replace(config,
                hidden_dim=trial["hidden_dim"],
                tau=trial["tau"])
        else:
            config_trial = config
        return integrator.ContrastiveEmbeddingModel(config_trial).to(device)  # type: ignore
    return model_init


def prepare_neg_dataset(equiv_dataset: SpeechEquivalenceDataset,
                        hidden_states_path: str, **kwargs
                        ) -> tuple[int, datasets.IterableDataset, datasets.IterableDataset, int]:
    # Pick a max length that accommodates the majority of the samples,
    # excluding outlier lengths
    evident_lengths = equiv_dataset.lengths
    evident_lengths = evident_lengths[evident_lengths != -1]
    target_length = int(torch.quantile(evident_lengths.double(), 0.95).item())

    num_examples, train_dataset, eval_dataset = integrator.prepare_dataset(
        equiv_dataset, hidden_states_path, target_length, **kwargs)

    return num_examples, train_dataset, eval_dataset, target_length


def hyperparameter_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "tau": tune.loguniform(1e-3, 1),
        "hidden_dim": tune.choice([32, 64, 128, 256]),
    }


HYPERPARAMETER_OBJECTIVE_DIRECTION = "maximize"
def hyperparameter_objective(metrics: dict[str, float]) -> float:
    from pprint import pprint
    pprint(metrics)
    return metrics["eval_embedding_isoscore"]


def train(config: DictConfig):
    if config.device == "cuda":
        if not torch.cuda.is_available():
            L.error("CUDA is not available. Falling back to CPU.")
            config.device = "cpu"
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir)
    assert not isinstance(dataset, datasets.DatasetDict), "should be a Dataset, not be a DatasetDict"

    hidden_states_path = config.base_model.hidden_state_path
    hidden_state_dataset = SpeechHiddenStateDataset.from_hdf5(config.base_model.hidden_state_path)

    with open(config.equivalence.path, "rb") as f:
        equiv_dataset: SpeechEquivalenceDataset = torch.load(f)

    # Prepare negative-sampling dataset
    if config.trainer.mode in ["train", "hyperparameter_search"]:
        total_num_examples, train_dataset, eval_dataset, max_length = prepare_neg_dataset(
            equiv_dataset, hidden_states_path)
        
        train_dataset = train_dataset.with_format("torch")
        eval_dataset = eval_dataset.with_format("torch")
    elif config.trainer.mode == "no_train":
        total_num_examples = 0
        train_dataset, eval_dataset = None, None
        max_length = equiv_dataset.lengths.max().item()
    else:
        raise ValueError(f"Invalid trainer mode: {config.trainer.mode}")

    max_training_steps = config.training_args.num_train_epochs * total_num_examples

    model_config = integrator.ContrastiveEmbeddingModelConfig(
        equivalence_classer=config.equivalence.equivalence_classer,
        max_length=max_length,
        input_dim=hidden_state_dataset.hidden_size,
        **OmegaConf.to_object(config.model))
    model_init = make_model_init(model_config, device=config.device)
    
    # Don't directly use `instantiate` with `TrainingArguments` or `Trainer` because the
    # type validation stuff is craaaaazy.
    # ^ can fix this with _recursive_ = False I think
    # We also have to use `to_object` to make sure the params are JSON-serializable
    
    model_learning_rate = config.model.get("learning_rate")
    if model_learning_rate is not None:
        L.warning("Overriding Trainer learning rate with config value from model config: %g", model_learning_rate)
        config.training_args.learning_rate = model_learning_rate

    training_args = transformers.TrainingArguments(
        output_dir=HydraConfig.get().runtime.output_dir,
        logging_dir=Path(HydraConfig.get().runtime.output_dir) / "logs",
        max_steps=max_training_steps,
        weight_decay=config.model.get("weight_decay", 0.0),
        **OmegaConf.to_object(config.training_args))

    callbacks = []
    if "callbacks" in config.trainer:
        callbacks = [instantiate(c) for c in config.trainer.callbacks]
    trainer_config = dict(config.trainer)
    trainer_config.pop("callbacks", None)
    trainer_mode = trainer_config.pop("mode", "train")
    hparam_config = trainer_config.pop("hyperparameter_search", None)
    trainer = transformers.Trainer(
        args=training_args,
        model=None, model_init=model_init,
        callbacks=callbacks,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        compute_metrics=integrator.compute_metrics,
        **trainer_config)

    if trainer_mode == "train":
        trainer.train()
    elif trainer_mode == "hyperparameter_search":
        trainer.hyperparameter_search(
            direction=HYPERPARAMETER_OBJECTIVE_DIRECTION,
            backend="ray",
            n_trials=hparam_config.n_trials,
            hp_space=hyperparameter_space,
            compute_objective=hyperparameter_objective,
            scheduler=instantiate(hparam_config.scheduler,
                                  mode=HYPERPARAMETER_OBJECTIVE_DIRECTION[:3]),
        )
    elif trainer_mode == "no_train":
        checkpoint_dir = Path(training_args.output_dir) / "checkpoint-0"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        trainer.save_model(checkpoint_dir)

        # Save dummy trainer state
        trainer.state.best_model_checkpoint = str(checkpoint_dir)
        trainer.state.save_to_json(checkpoint_dir / "trainer_state.json")