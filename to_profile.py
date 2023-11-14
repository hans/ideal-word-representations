from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import transformers
from transformers import AutoConfig
from transformers import Wav2Vec2Model
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from models.transformer import Wav2Vec2ForSpeechClassification, DataCollator, \
    drop_wav2vec_layers
from utils.timit import TimitCorpus


def get_phone_vocab(corpus):
    from models import Vocabulary
    phone_vocab = Vocabulary("phones")
    corpus.map(lambda x: [phone_vocab.add_token(phon) for phon in x["phonetic_detail"]["utterance"]] and None,
               batched=False, load_from_cache_file=False)
    return phone_vocab


def prepare_corpus():
    corpus = datasets.load_dataset("timit_asr", data_dir="/userdata/jgauthier/data/TIMIT")

    from utils.timit import group_phonetic_detail
    corpus = corpus.map(group_phonetic_detail, batched=False, load_from_cache_file=False, fn_kwargs=dict(drop_phones=drop_phones))

    phone_vocab = get_phone_vocab(corpus)
    
    def prepare_audio(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        return batch
    corpus = corpus.map(prepare_audio)

    from models.transformer import TilingWordFeatureExtractor2
    twfe = TilingWordFeatureExtractor2(phone_vocab.index2token)
    def add_features(example):
        example["phone_targets"] = twfe(example)
        return example
    corpus = corpus.map(add_features, load_from_cache_file=False)
    
    return corpus, phone_vocab


def load_corpus(corpus_path="timit_corpus"):
    if not Path(corpus_path).exists():
        corpus, phone_vocab = prepare_corpus()
        corpus.save_to_disk(corpus_path)
    else:
        corpus = datasets.load_from_disk(corpus_path)
        phone_vocab = get_phone_vocab(corpus)

    return corpus, phone_vocab


def make_model_init(model_name_or_path, config, device="cpu"):
    def model_init(trial):
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_name_or_path, config=config).to(device)

        model.freeze_feature_extractor()
        # model.wav2vec2 = drop_wav2vec_layers(model.wav2vec2, 10)

        # Freeze all model weights.
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
        
        return model
    return model_init


def compute_metrics(p: transformers.EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = preds.reshape((-1, preds.shape[-1]))
    labels = p.label_ids.reshape((-1, preds.shape[-1]))

    def evaluate_label(j):
        preds_j = preds[:, j]
        labels_j = labels[:, j]

        mask = labels_j != -100
        preds_j = preds_j[mask]
        labels_j = labels_j[mask]
        if labels_j.std() == 0:
            # Only one class. Quit
            return None
        return roc_auc_score(labels_j, preds_j)

    roc_auc_scores = [evaluate_label(j) for j in range(preds.shape[-1])]
    return {"roc_auc": np.mean([score for score in roc_auc_scores if score is not None])}


def main():
    model_name_or_path = "facebook/wav2vec2-base-960h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = transformers.Wav2Vec2Processor.from_pretrained(model_name_or_path)
    corpus, phone_vocab = load_corpus()

    # Load wav2vec2
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        problem_type="multi_label_classification",
        num_labels=len(phone_vocab))
    setattr(config, "pooling_mode", "mean")
    setattr(config, "classifier_bias", False)
    model_init = make_model_init(model_name_or_path, config, device=device)

    coll = DataCollator(processor=processor, model=model_init(None), padding=True,
                        num_labels=len(phone_vocab.index2token))

    training_args = TrainingArguments(
        output_dir="out/cli_test",
        group_by_length=True,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=2,
        gradient_accumulation_steps=2,
        save_steps=25,
        eval_steps=25,
        logging_steps=2,
        learning_rate=1e-4,
        save_total_limit=5,
        use_cpu=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        # disable_tqdm=True,
    )

    trainer = Trainer(
        model=None, model_init=model_init,
        data_collator=coll,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
        train_dataset=corpus["train"].select(range(100)),
        eval_dataset=corpus["test"],
        tokenizer=processor.feature_extractor,
    )

    # # Manual
    model = model_init(None)
    batch = next(iter(trainer.get_train_dataloader()))
    model(**batch)

main()