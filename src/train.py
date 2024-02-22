import logging 
from pathlib import Path

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import transformers

from src.datasets.speech_equivalence import SpeechEquivalenceDataset, load_or_make_timit_equivalence_dataset
from src.models import integrator
from src.utils.timit import load_or_prepare_timit_corpus

L = logging.getLogger(__name__)


def make_model_init(config, device="cpu"):
    def model_init(trial):
        return integrator.ContrastiveEmbeddingModel(config).to(device)
    return model_init


# def compute_classifier_metrics(p: transformers.EvalPrediction) -> dict:
#     assert isinstance(p.predictions, tuple)
#     preds = p.predictions[0]
#     label_mask, labels, _ = p.label_ids

#     def evaluate_label(j):
#         preds_j = preds[:, :, j]
#         labels_j = labels[:, :, j]

#         preds_j = preds_j[label_mask == 1]
#         labels_j = labels_j[label_mask == 1]
#         if labels_j.std() == 0:
#             # Only one class. Quit
#             return None
#         return roc_auc_score(labels_j, preds_j)

#     roc_auc_scores = [evaluate_label(j) for j in range(preds.shape[-1])]
#     return {"roc_auc": np.mean([score for score in roc_auc_scores if score is not None])}


def prepare_neg_dataset(equiv_dataset: SpeechEquivalenceDataset) -> tuple[Dataset, int]:
    # Pick a max length that accommodates the majority of the samples,
    # excluding outlier lengths
    evident_lengths = equiv_dataset.lengths
    evident_lengths = evident_lengths[evident_lengths != -1]
    target_length = int(torch.quantile(evident_lengths.double(), 0.95).item())

    return integrator.prepare_dataset(equiv_dataset, target_length), target_length


def train(config: DictConfig):
    if config.device == "cuda":
        if not torch.cuda.is_available():
            L.error("CUDA is not available. Falling back to CPU.")
            config.device = "cpu"

    tokenizer = instantiate(config.tokenizer)
    feature_extractor = instantiate(config.feature_extractor,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=False)
    processor = transformers.Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    base_model = transformers.Wav2Vec2Model.from_pretrained(config.model.base_model_ref)

    # Prepare basic speech dataset
    dataset = instantiate(config.dataset, processor=processor,
                          _convert_="partial")
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    # Prepare equivalence-classing dataset
    equiv_name = f"{config.model.base_model_ref}_{config.model.base_model_layer}-{config.equivalence.equivalence_classer}-{config.equivalence.num_frames_per_phoneme}"
    equiv_name = equiv_name.replace("/", "-")
    equiv_dataset = instantiate(
        config.equivalence, _recursive_=False,
        name=equiv_name,
        dataset=dataset,
        model=base_model,
        layer=config.model.base_model_layer)

    # Prepare negative-sampling dataset
    dataset_path = Path(HydraConfig.get().runtime.output_dir) / "neg_dataset"
    neg_dataset, max_length = prepare_neg_dataset(equiv_dataset)
    neg_dataset_split = neg_dataset.train_test_split(test_size=0.1, shuffle=True)
    neg_dataset_split.save_to_disk(dataset_path)

    train_dataset = neg_dataset_split["train"]
    eval_dataset = neg_dataset_split["test"]

    model_config = integrator.ContrastiveEmbeddingModelConfig(
        equivalence_classer=config.equivalence.equivalence_classer,
        max_length=max_length,
        input_dim=equiv_dataset.hidden_state_dataset.hidden_size,
        **config.model)
    model_init = make_model_init(model_config, device=config.device)
    
    # Don't directly use `instantiate` with `TrainingArguments` or `Trainer` because the
    # type validation stuff is craaaaazy.
    # ^ can fix this with _recursive_ = False I think
    # We also have to use `to_object` to make sure the params are JSON-serializable
    
    training_args = transformers.TrainingArguments(
        output_dir=HydraConfig.get().runtime.output_dir,
        logging_dir=Path(HydraConfig.get().runtime.output_dir) / "logs",
        **OmegaConf.to_object(config.training_args))

    callbacks = []
    if "callbacks" in config.trainer:
        callbacks = [instantiate(c) for c in config.trainer.callbacks]
    trainer_config = dict(config.trainer)
    trainer_config.pop("callbacks", None)
    trainer = transformers.Trainer(
        args=training_args,
        model=None, model_init=model_init,
        callbacks=callbacks,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        **trainer_config)

    trainer.train()