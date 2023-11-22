from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import transformers

from src.models.transformer import drop_wav2vec_layers
from src.models.frame_level import FrameLevelLexicalAccess, \
    LexicalAccessConfig, LexicalAccessDataCollator


def make_model_init(
        model_name_or_path,
        config,
        word_representations,
        device="cpu"):
    def model_init(trial):
        # encoder = transformers.Wav2Vec2Model.from_pretrained(
        #     model_name_or_path, config=config.encoder_config).to(device)
        model = FrameLevelLexicalAccess(
            config, word_representations,
            encoder_name_or_path=model_name_or_path).to(device)

        model.freeze_feature_extractor()

        if config.drop_encoder_layers is not None:
            model.encoder = drop_wav2vec_layers(model.encoder, config.drop_encoder_layers)

        if config.reinit_feature_extractor_weights:
            model.encoder.feature_extractor.apply(lambda x: model.encoder._init_weights(x))
        if config.reinit_encoder_weights:
            model.encoder.encoder.apply(lambda x: model.encoder._init_weights(x))

        # Freeze all model weights.
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        return model
    return model_init


def compute_classifier_metrics(p: transformers.EvalPrediction) -> dict:
    assert isinstance(p.predictions, tuple)
    preds = p.predictions[0]
    label_mask, labels, _ = p.label_ids

    def evaluate_label(j):
        preds_j = preds[:, :, j]
        labels_j = labels[:, :, j]

        preds_j = preds_j[label_mask == 1]
        labels_j = labels_j[label_mask == 1]
        if labels_j.std() == 0:
            # Only one class. Quit
            return None
        return roc_auc_score(labels_j, preds_j)

    roc_auc_scores = [evaluate_label(j) for j in range(preds.shape[-1])]
    return {"roc_auc": np.mean([score for score in roc_auc_scores if score is not None])}


def compute_regressor_metrics(p: transformers.EvalPrediction) -> dict:
    assert isinstance(p.predictions, tuple)
    preds = p.predictions[1]
    target_mask, _, targets = p.label_ids

    preds = preds[target_mask == 1]
    targets = targets[target_mask == 1]

    return {"mse": ((preds - targets) ** 2).mean().item()}


def compute_metrics_dual_head(p: transformers.EvalPrediction):
    return {
        **compute_classifier_metrics(p),
        **compute_regressor_metrics(p)
    }


def estimate_decision_thresholds(trainer: transformers.Trainer):
    assert trainer.eval_dataset is not None

    trainer.model.eval()
    with torch.no_grad():
        preds = trainer.predict(trainer.eval_dataset)
        label_mask, labels, _ = preds.label_ids
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

    def add_model_outputs(batch, idxs):
        eval_logits, eval_targets, rnn_states = dataset_preds.predictions
        logits = eval_logits[idxs]
        preds = (logits > optimal_thresholds.numpy()).astype(int)

        # Internals
        batch["rnn_hidden_states"] = rnn_states[idxs]

        # Classifier outputs
        batch["logits"] = logits
        batch["distance_from_decision_threshold"] = logits - optimal_thresholds.numpy()
        batch["predicted_labels"] = preds

        # Regressor outputs
        batch["regression_output"] = eval_targets[idxs]

        return batch

    result = dataset.map(add_model_outputs, batched=True, batch_size=8, with_indices=True)

    def compute_accuracy(item, idx):
        label_mask, labels, _ = dataset_preds.label_ids
        label_mask = label_mask[idx] == 1
        labels = labels[idx]

        item["real_frames"] = label_mask.sum()
        item["labels"] = labels[label_mask]
        item["compression_ratio"] = item["real_frames"] / len(item["input_values"])
        item["correct"] = (np.array(item["predicted_labels"])[label_mask] == labels[label_mask])
        item["fp"] = (np.array(item["predicted_labels"])[label_mask] == 1) & (labels[label_mask] == 0)
        item["fn"] = (np.array(item["predicted_labels"])[label_mask] == 0) & (labels[label_mask] == 1)
        item["tp"] = (np.array(item["predicted_labels"])[label_mask] == 1) & (labels[label_mask] == 1)
        item["tn"] = (np.array(item["predicted_labels"])[label_mask] == 0) & (labels[label_mask] == 0)
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

    # Prepare word semantic representations as model targets
    semantic_target_encoder = instantiate(config.semantic_targets)
    all_words, word_representations = semantic_target_encoder(dataset)

    encoder_config = transformers.AutoConfig.from_pretrained(
        config.model.base_model)
    model_config = LexicalAccessConfig.from_configs(
        encoder_config=encoder_config,
        num_labels=tokenizer.vocab_size,
        word_vocabulary=all_words,
        regressor_target_size=word_representations.shape[1],
        **config.model)
    model_init = make_model_init(
        config.model.base_model, model_config,
        torch.tensor(word_representations).to(config.device),
        device=config.device)

    collator = instantiate(
        config.collator,
        processor=processor,
        model=model_init(None),
        padding=True,
        num_labels=model_config.num_labels,
        regression_target_size=word_representations.shape[1])
    
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
        compute_metrics=compute_metrics_dual_head,
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