# This file is out-of-date with respect to the new corpus as of
# 20231114. Should be updated to use a static `Tokenizer` rather
# than the dynamic phone vocabulary.


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


def load_corpus(corpus_path="timit_phoneme_corpus"):
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

        if hasattr(config, "drop_layers"):
            model.wav2vec2 = drop_wav2vec_layers(model.wav2vec2, config.drop_layers)

        # Freeze all model weights.
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
        
        return model
    return model_init


def compute_metrics(p: transformers.EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_mask, labels = p.label_ids

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


def main():
    model_name_or_path = "facebook/wav2vec2-base-960h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    drop_layers = 6

    processor = transformers.Wav2Vec2Processor.from_pretrained(model_name_or_path)
    corpus, phone_vocab = load_corpus()

    # Load wav2vec2
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        problem_type="multi_label_classification",
        num_labels=len(phone_vocab))
    setattr(config, "pooling_mode", "mean")
    setattr(config, "classifier_bias", False)
    setattr(config, "output_vocab", phone_vocab.index2token)
    setattr(config, "drop_layers", drop_layers)
    model_init = make_model_init(model_name_or_path, config, device=device)

    coll = DataCollator(processor=processor, model=model_init(None), padding=True,
                        num_labels=len(phone_vocab.index2token))

    training_args = TrainingArguments(
        output_dir=f"out/pure_hugging/run5_drop{drop_layers}layers",
        # group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=50,
        gradient_accumulation_steps=2,
        save_steps=50,
        eval_steps=50,
        logging_steps=2,
        learning_rate=1e-2,
        save_total_limit=5,
        use_cpu=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        label_names=["label_mask", "labels"],
        # disable_tqdm=True,
    )

    trainer = Trainer(
        model=None, model_init=model_init,
        data_collator=coll,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
        train_dataset=corpus["train"],
        eval_dataset=corpus["test"],
        tokenizer=processor.feature_extractor,
    )

    # ## Manual
    # batch = next(iter(trainer.get_train_dataloader()))
    # model = model_init(None)
    # model_out = model(**batch)
    # import ipdb; ipdb.set_trace()

    trainer.train()

main()