"""
Defines SLM frame-level classifier and regression models.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.file_utils import ModelOutput
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC


@dataclass
class RecurrentClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    wav2vec2_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    wav2vec2_attentions: Optional[Tuple[torch.FloatTensor]] = None
    rnn_hidden_states: Optional[torch.FloatTensor] = None


class FrameLevelRNNClassifier(Wav2Vec2ForCTC):
    """
    Frame-level RNN classifier for multi-label classification.
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)

        # RNN
        self.rnn = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=getattr(config, "rnn_num_layers", 1),
            bidirectional=False,
            batch_first=True,
        )

        # Classification head
        self.classifier_dropout = nn.Dropout(config.final_dropout)
        self.classifier = nn.Linear(config.rnn_hidden_size, config.num_labels, bias=config.classifier_bias)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values,
            attention_mask=None,
            label_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # RNN
        rnn_out, _ = self.rnn(hidden_states)

        logits = self.classifier(self.dropout(rnn_out))

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
                
            if label_mask is not None:
                active_loss = label_mask == 1
                active_logits = logits[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RecurrentClassifierOutput(
            loss=loss,
            logits=logits,
            wav2vec2_hidden_states=outputs.hidden_states,
            wav2vec2_attentions=outputs.attentions,
            rnn_hidden_states=rnn_out,
        )