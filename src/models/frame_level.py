"""
Defines SLM frame-level classifier and regression models.
"""

import copy
from dataclasses import dataclass
import logging
from typing import Optional, Tuple, Union, List, Dict

import torch
import torch.nn as nn
from transformers.file_utils import ModelOutput
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor, \
    Wav2Vec2Config, PretrainedConfig, \
    BatchFeature, PreTrainedModel, SequenceFeatureExtractor

from src.models.rnn import ExposedLSTM
from src.models.transformer import PhoneticTargetFeatureExtractor


L = logging.getLogger(__name__)


@dataclass
class RecurrentClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    wav2vec2_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    wav2vec2_attentions: Optional[Tuple[torch.FloatTensor]] = None
    rnn_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class LexicalAccessOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    semantic: torch.FloatTensor = None

    wav2vec2_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    wav2vec2_attentions: Optional[Tuple[torch.FloatTensor]] = None
    rnn_hidden_states: Optional[torch.FloatTensor] = None


class DummyIdentityRNN(nn.Module):
    """
    Dummy module which acts as identity, but returns RNN-like output.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x, None


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
        num_layers = getattr(config, "rnn_num_layers", 1)
        rnn_module = nn.LSTM if num_layers > 0 else DummyIdentityRNN
        if num_layers == 0:
            # TODO log?
            setattr(config, "rnn_hidden_size", config.hidden_size)

        self.rnn = rnn_module(
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


class SemanticTargetFeatureExtractor(SequenceFeatureExtractor):
    model_input_names: List[str] = ["targets"]


@dataclass
class LexicalAccessDataCollator:
    """
    Data collator for the lexical access model, with dual-head predictions
    where padding is the same for the two heads.

    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    model: "FrameLevelLexicalAccess"
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    num_labels: int = 2
    regression_target_size: int = 32
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def _collate_frame_labels(
            self,
            features: List[Dict[str, Union[List[int], torch.Tensor]]],
            batch: BatchFeature) -> BatchFeature:
        # Calculate how many frames we have per batch item
        batch_num_samples = batch.attention_mask.sum(dim=1)
        batch_num_frames = self.model.encoder._get_feat_extract_output_lengths(batch_num_samples).tolist()

        # TODO this is imprecise due to padding. may be very minor changes in alignment
        # that may matter if we care about precise word boundary effects
        batch_max_frames = self.model.encoder._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        compression_ratio = batch_max_frames / batch["input_values"].shape[-1]

        label_features = [torch.zeros((item_num_frames, self.num_labels), dtype=torch.long)
                          for item_num_frames in batch_num_frames]
        for i, feature in enumerate(features):
            for onset, offset, label in feature["phone_targets"]:
                onset = int(onset * compression_ratio)
                offset = int(offset * compression_ratio)
                label_features[i][onset:offset, label] = 1
        label_features = [{"phones": feature} for feature in label_features]

        my_padder = PhoneticTargetFeatureExtractor(
            self.num_labels,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding_value=0,)

        return my_padder.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def _collate_frame_regression_targets(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]],
            batch: BatchFeature) -> BatchFeature:
        # Calculate how many frames we have per batch item
        batch_num_samples = batch.attention_mask.sum(dim=1)
        batch_num_frames = self.model.encoder._get_feat_extract_output_lengths(batch_num_samples).tolist()

        # TODO this is imprecise due to padding. may be very minor changes in alignment
        # that may matter if we care about precise word boundary effects
        batch_max_frames = self.model.encoder._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        compression_ratio = batch_max_frames / batch["input_values"].shape[-1]

        targets = [torch.zeros((item_num_frames, self.regression_target_size),
                               dtype=torch.float)
                   for item_num_frames in batch_num_frames]
        for i, feature in enumerate(features):
            for onset, offset, word in zip(feature["word_detail"]["start"], feature["word_detail"]["stop"],
                                           feature["word_detail"]["utterance"]):
                word_id = self.model.word_to_idx[word]
                onset = int(onset * compression_ratio)
                offset = int(offset * compression_ratio)

                targets[i][onset:offset, :] = torch.tensor(self.model.word_representations[word_id])

        feature_extractor = SemanticTargetFeatureExtractor(
            feature_size=self.regression_target_size,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding_value=0,)
        result = feature_extractor.pad(
            [{"targets": targets_i} for targets_i in targets],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return result
        

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        # For classification
        # label_features = [feature["labels"] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        label_features = self._collate_frame_labels(features, batch)        
        batch["target_mask"] = label_features["attention_mask"]
        batch["classifier_labels"] = label_features["phones"]

        regression_features = self._collate_frame_regression_targets(features, batch)
        batch["regressor_targets"] = regression_features["targets"]

        return batch


class LexicalAccessConfig(PretrainedConfig):

    model_type = "lexical_access"
    is_composition = True

    def __init__(
            self,

            encoder_config: Optional[dict] = None,
            drop_encoder_layers: Optional[int] = None,
            reinit_feature_extractor_weights: bool = False,
            reinit_encoder_weights: bool = False,

            dropout: float = 0.1,

            rnn_num_layers: int = 2,
            rnn_hidden_size: int = 128,
            expose_rnn: bool = False,

            classifier_num_labels: int = 32,
            classifier_bias: bool = True,

            word_vocabulary: Optional[List[str]] = None,
            regressor_target_size: int = 32,
            regressor_loss: str = "cosine",

            loss_alpha: float = 0.5,
            **kwargs):
        super().__init__(**kwargs)

        if encoder_config is None:
            encoder_config = {}
            L.info("No encoder config provided, using default encoder config.")
        self.encoder_config = Wav2Vec2Config(**encoder_config)
        self.drop_encoder_layers = drop_encoder_layers
        self.reinit_feature_extractor_weights = reinit_feature_extractor_weights
        self.reinit_encoder_weights = reinit_encoder_weights

        self.dropout = dropout

        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.expose_rnn = expose_rnn

        self.classifier_num_labels = classifier_num_labels
        self.classifier_bias = classifier_bias

        self.word_vocabulary = word_vocabulary
        self.regressor_target_size = regressor_target_size
        self.regressor_loss = regressor_loss

        if self.regressor_loss not in [None, "mse", "cosine"]:
            raise ValueError(f"Regressor loss {self.regressor_loss} not supported.")
        
        self.loss_alpha = loss_alpha

    @classmethod
    def from_configs(cls, encoder_config: PretrainedConfig,
                     **kwargs) -> PretrainedConfig:
        return cls(encoder_config=encoder_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["encoder_config"] = self.encoder_config.to_dict()
        return output


class FrameLevelLexicalAccess(PreTrainedModel):
    """
    Dual-head frame level model, outputting
    1) multi-label classifier outputs
    2) semantic representation
    """
    config_class = LexicalAccessConfig

    def __init__(self,
                 config: LexicalAccessConfig,
                 word_representations: Optional[torch.FloatTensor] = None,
                 encoder_name_or_path: Optional[str] = None,):
        super().__init__(config)
        self.config = config

        word_vocab_size = len(config.word_vocabulary) if config.word_vocabulary is not None else 0
        if word_representations is None:
            word_representations = torch.zeros(word_vocab_size, config.regressor_target_size)
        self.word_representations = nn.Parameter(word_representations, requires_grad=False)
        assert len(config.word_vocabulary) == word_representations.shape[0]
        self.word_to_idx = {word: idx for idx, word in enumerate(config.word_vocabulary)}

        if encoder_name_or_path is None:
            self.encoder = Wav2Vec2Model(config.encoder_config)
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(encoder_name_or_path,
                                                         config=config.encoder_config)

        # RNN
        if config.rnn_num_layers == 0:
            rnn_module = DummyIdentityRNN
        elif config.expose_rnn:
            rnn_module = ExposedLSTM
        else:
            rnn_module = nn.LSTM
        self.rnn = rnn_module(
            input_size=config.encoder_config.hidden_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.encoder_dropout = nn.Dropout(config.dropout)

        # Classification head
        self.classifier_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.rnn_hidden_size, config.num_labels, bias=config.classifier_bias)

        # Regression head
        self.regressor_dropout = nn.Dropout(config.dropout)
        self.regressor = nn.Linear(config.rnn_hidden_size, config.regressor_target_size)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.encoder.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            
            target_mask=None,
            classifier_labels=None,
            regressor_targets=None,
            loss_alpha=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.encoder_dropout(hidden_states)

        # RNN
        if self.config.expose_rnn:
            (rnn_outputs, rnn_cells, rnn_input_gates,
             rnn_forget_gates, rnn_cell_gates, rnn_output_gates) = self.rnn(hidden_states)
            
            rnn_out = rnn_outputs[-1]
        else:
            rnn_out, _ = self.rnn(hidden_states)

        # Outputs
        logits = self.classifier(self.classifier_dropout(rnn_out))
        semantic = self.regressor(self.regressor_dropout(rnn_out))

        # Classification loss
        loss = torch.tensor(0.).to(logits)

        if loss_alpha is None:
            loss_alpha = self.config.loss_alpha
        loss_alpha = torch.tensor(loss_alpha).to(logits)
        # Bizarre logic I know, I just wanted to add a loss scaler easily
        if loss_alpha > 1:
            loss_alpha_classifier, loss_alpha_regressor = 1 / loss_alpha, loss_alpha
        else:
            loss_alpha_classifier, loss_alpha_regressor = loss_alpha, 1 - loss_alpha

        loss_mask = target_mask == 1 if target_mask is not None else None
        if classifier_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            
            active_logits = logits[loss_mask] if loss_mask is not None else logits
            active_labels = classifier_labels[loss_mask] if loss_mask is not None else classifier_labels
            loss += loss_alpha_classifier * loss_fct(active_logits, active_labels.float())

        # Regression loss
        if regressor_targets is not None and self.config.regressor_loss is not None:
            active_semantic = semantic[loss_mask] if loss_mask is not None else semantic
            active_targets = regressor_targets[loss_mask] if loss_mask is not None else regressor_targets

            if self.config.regressor_loss == "mse":
                loss += loss_alpha_regressor * nn.MSELoss()(active_semantic, active_targets)
            elif self.config.regressor_loss == "cosine":
                loss += loss_alpha_regressor * nn.CosineEmbeddingLoss()(active_semantic, active_targets, target=torch.ones(active_semantic.shape[0])).to(active_semantic)
            else:
                raise ValueError(f"Regressor loss {self.config.regressor_loss} not supported.")

        if not return_dict:
            output = (logits, semantic) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LexicalAccessOutput(
            loss=loss,
            logits=logits,
            semantic=semantic,
            wav2vec2_hidden_states=outputs.hidden_states,
            wav2vec2_attentions=outputs.attentions,

            # TODO expose RNN states
            rnn_hidden_states=rnn_out,
        )