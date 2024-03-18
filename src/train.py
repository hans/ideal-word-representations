import logging 
from pathlib import Path

import datasets
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import transformers

from src.datasets.speech_equivalence import SpeechEquivalenceDataset, SpeechHiddenStateDataset
from src.models import integrator

L = logging.getLogger(__name__)


def make_model_init(config, device="cpu"):
    def model_init(trial):
        return integrator.ContrastiveEmbeddingModel(config).to(device)  # type: ignore
    return model_init


def prepare_neg_dataset(equiv_dataset: SpeechEquivalenceDataset,
                        hidden_state_dataset: SpeechHiddenStateDataset, **kwargs
                        ) -> tuple[int, datasets.IterableDataset, datasets.IterableDataset, int]:
    # Pick a max length that accommodates the majority of the samples,
    # excluding outlier lengths
    evident_lengths = equiv_dataset.lengths
    evident_lengths = evident_lengths[evident_lengths != -1]
    target_length = int(torch.quantile(evident_lengths.double(), 0.95).item())

    num_examples, train_dataset, eval_dataset = integrator.prepare_dataset(
        equiv_dataset, hidden_state_dataset, target_length, **kwargs)

    return num_examples, train_dataset, eval_dataset, target_length


def train(config: DictConfig):
    if config.device == "cuda":
        if not torch.cuda.is_available():
            L.error("CUDA is not available. Falling back to CPU.")
            config.device = "cpu"
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir)
    assert not isinstance(dataset, datasets.DatasetDict), "should be a Dataset, not be a DatasetDict"

    with open(config.base_model.hidden_state_path, "rb") as f:
        hidden_state_dataset: SpeechHiddenStateDataset = torch.load(f)

    with open(config.equivalence.path, "rb") as f:
        equiv_dataset: SpeechEquivalenceDataset = torch.load(f)

    # Prepare negative-sampling dataset
    if config.trainer.do_train:
        total_num_examples, train_dataset, eval_dataset, max_length = prepare_neg_dataset(
            equiv_dataset, hidden_state_dataset)
        
        train_dataset = train_dataset.with_format("torch")
        eval_dataset = eval_dataset.with_format("torch")
    else:
        total_num_examples = 0
        train_dataset, eval_dataset = None, None
        max_length = equiv_dataset.lengths.max().item()
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
    do_train = trainer_config.pop("do_train", True)
    trainer = transformers.Trainer(
        args=training_args,
        model=None, model_init=model_init,
        callbacks=callbacks,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        compute_metrics=integrator.compute_metrics,
        **trainer_config)

    if do_train:
        trainer.train()
    else:
        checkpoint_dir = Path(training_args.output_dir) / "checkpoint-0"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        trainer.save_model(checkpoint_dir)

        # Save dummy trainer state
        trainer.state.best_model_checkpoint = str(checkpoint_dir)
        trainer.state.save_to_json(checkpoint_dir / "trainer_state.json")