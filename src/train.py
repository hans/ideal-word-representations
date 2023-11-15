from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from sklearn.metrics import roc_curve
import torch
import transformers

from src.models.transformer import compute_metrics, drop_wav2vec_layers
from src.models.frame_level import FrameLevelRNNClassifier


def make_model_init(model_name_or_path, config, device="cpu"):
    def model_init(trial):
        model = FrameLevelRNNClassifier.from_pretrained(
            model_name_or_path, config=config).to(device)

        model.freeze_feature_extractor()

        if hasattr(config, "drop_layers"):
            model.wav2vec2 = drop_wav2vec_layers(model.wav2vec2, config.drop_layers)

        if getattr(config, "reinit_feature_extractor_weights", False):
            model.wav2vec2.feature_extractor.apply(lambda x: model.wav2vec2._init_weights(x))
        if getattr(config, "reinit_encoder_weights", False):
            model.wav2vec2.encoder.apply(lambda x: model.wav2vec2._init_weights(x))

        # Freeze all model weights.
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
        
        return model
    return model_init


def estimate_decision_thresholds(trainer: transformers.Trainer):
    assert trainer.eval_dataset is not None

    trainer.model.eval()
    with torch.no_grad():
        preds = trainer.predict(trainer.eval_dataset)
        label_mask, labels = preds.label_ids
        preds = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions

    # Get optimal cut-off for each label
    optimal_thresholds = []
    fpr, tpr, thresholds = [], [], []
    # roc_aucs = []
    for j in range(preds.shape[-1]):
        preds_j = preds[:, :, j]
        labels_j = labels[:, :, j]

        mask = label_mask == 1
        preds_j = preds_j[mask]
        labels_j = labels_j[mask]

        fpr_j, tpr_j, thresholds_j = roc_curve(labels_j, preds_j, pos_label=1)
        fpr.append(fpr_j)
        tpr.append(tpr_j)
        thresholds.append(thresholds_j)

        optimal_thresholds.append(thresholds_j[np.argmax(tpr_j - fpr_j)])

    return torch.tensor(optimal_thresholds)


def rollout_dataset(trainer, dataset, optimal_thresholds: torch.FloatTensor) -> Dataset:
    """
    Roll out on a new dataset; return a new dataset containing both the original
    inputs and the internal states, predictions, and accuracies produced
    during the model evaluation.
    """
    trainer.model.eval()
    with torch.no_grad():
        dataset_preds = trainer.predict(dataset)

    def add_predictions(batch, idxs):
        eval_output, rnn_states = dataset_preds.predictions
        logits = eval_output[idxs]
        preds = (logits > optimal_thresholds.numpy()).astype(int)

        batch["rnn_hidden_states"] = rnn_states[idxs]
        batch["logits"] = logits
        batch["distance_from_decision_threshold"] = logits - optimal_thresholds.numpy()
        batch["predicted"] = preds

        return batch

    result = dataset.map(add_predictions, batched=True, batch_size=8, with_indices=True)

    def compute_accuracy(item, idx):
        label_mask, labels = dataset_preds.label_ids
        label_mask = label_mask[idx] == 1
        labels = labels[idx]

        item["real_frames"] = label_mask.sum()
        item["labels"] = labels[label_mask]
        item["compression_ratio"] = item["real_frames"] / len(item["input_values"])
        item["correct"] = (np.array(item["predicted"])[label_mask] == labels[label_mask])
        item["fp"] = (np.array(item["predicted"])[label_mask] == 1) & (labels[label_mask] == 0)
        item["fn"] = (np.array(item["predicted"])[label_mask] == 0) & (labels[label_mask] == 1)
        item["tp"] = (np.array(item["predicted"])[label_mask] == 1) & (labels[label_mask] == 1)
        item["tn"] = (np.array(item["predicted"])[label_mask] == 0) & (labels[label_mask] == 0)
        item["accuracy"] = item["correct"].mean()
        return item

    result = result.map(compute_accuracy, batched=False, with_indices=True)
    return result


def train(config: DictConfig):
    tokenizer = instantiate(config.tokenizer)
    feature_extractor = instantiate(config.feature_extractor,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=False)
    processor = transformers.Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    dataset = instantiate(config.dataset, processor=processor)
    dataset_split = dataset["train"].train_test_split(test_size=0.1, shuffle=True)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    hf_config = transformers.AutoConfig.from_pretrained(
        config.model.base_model, num_labels=processor.tokenizer.vocab_size)
    for key, value in config.model.items():
        setattr(hf_config, key, value)
    model_init = make_model_init(config.model.base_model, hf_config, device=config.device)

    collator = instantiate(config.collator,
                           processor=processor,
                           model=model_init(None),
                           padding=True,
                           num_labels=hf_config.num_labels)
    
    # Don't directly use `instantiate` with `TrainingArguments` or `Trainer` because the
    # type validation stuff is craaaaazy.
    # We also have to use `to_object` to make sure the params are JSON-serializable
    training_args = transformers.TrainingArguments(
        output_dir=HydraConfig.get().runtime.output_dir,
        **OmegaConf.to_object(config.training_args))

    callbacks = []
    if "callbacks" in config.trainer:
        callbacks = [instantiate(c) for c in config.trainer.callbacks]
    trainer_config = dict(config.trainer)
    trainer_config.pop("callbacks", None)
    trainer = transformers.Trainer(
        args=training_args,
        data_collator=collator,
        model=None, model_init=model_init,
        callbacks=callbacks,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
        **trainer_config)

    trainer.train()

    ####

    optimal_thresholds = estimate_decision_thresholds(trainer)
    torch.save(optimal_thresholds, Path(training_args.output_dir) / "optimal_thresholds.pt")

    ####

    test_result = rollout_dataset(trainer, dataset["test"], optimal_thresholds)
    # Remove columns which are redundant and take up lots of space
    test_result = test_result.remove_columns(["audio", "input_values", "phone_targets"])
    test_result.save_to_disk(Path(trainer.args.output_dir) / "test_result")