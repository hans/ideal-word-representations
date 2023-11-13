from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def drop_wav2vec_layers(model: Wav2Vec2Model, n=1) -> Wav2Vec2Model:
    """
    Drop the last n layers of the wav2vec encoder model. Operates in place.
    """
    model.encoder.layers = model.encoder.layers[:-n]
    return model


class TilingWordFeatureExtractor:
    """
    Extracts word-level features from TIMIT input and returns a compressed description
    of time series spans.
    """

    def __init__(self, all_phones):
        self.all_phones = sorted(all_phones)

        self.all_diphones = sorted(set(
            (phone1, phone2)
            for phone1 in self.all_phones
            for phone2 in self.all_phones
        ))

        self.phone2idx = {phone: i for i, phone in enumerate(self.all_phones)}
        self.diphone2idx = {diphone: i for i, diphone in enumerate(self.all_diphones)}

    @property
    def num_features(self):
        # return len(self.diphone2idx)
        return len(self.phone2idx)

    def _extract_features(self, timit_word):
        """
        Extract diphone features.
        """
        # return [self.diphone2idx[phone1["phone"], phone2["phone"]]
        #         for phone1, phone2 in zip(timit_word["phones"], timit_word["phones"][1:])]
        return [self.phone2idx[phone["phone"]] for phone in timit_word["phones"]]

    def __call__(self, timit_item) -> list[Tuple[int, int, int]]:
        ret = []

        for word in timit_item["words"]:
            for feature in self._extract_features(word):
                ret.append((word["onset"], word["offset"], feature))
        
        return ret


class TilingWordFeatureExtractor2:
    """
    Extracts word-level features from TIMIT input and returns a compressed description
    of time series spans.
    """

    def __init__(self, all_phones):
        self.all_phones = sorted(all_phones)

        self.all_diphones = sorted(set(
            (phone1, phone2)
            for phone1 in self.all_phones
            for phone2 in self.all_phones
        ))

        self.phone2idx = {phone: i for i, phone in enumerate(self.all_phones)}
        self.diphone2idx = {diphone: i for i, diphone in enumerate(self.all_diphones)}

    @property
    def num_features(self):
        # return len(self.diphone2idx)
        return len(self.phone2idx)

    def _extract_features(self, timit_word):
        """
        Extract diphone features.
        """
        # return [self.diphone2idx[phone1["phone"], phone2["phone"]]
        #         for phone1, phone2 in zip(timit_word["phones"], timit_word["phones"][1:])]
        return [self.phone2idx[phone["phone"]] for phone in timit_word["phones"]]

    def __call__(self, timit_item) -> list[Tuple[int, int, int]]:
        ret = []

        for word_start, word_stop, phones in zip(timit_item["word_detail"]["start"], timit_item["word_detail"]["stop"], timit_item["word_phonetic_detail"]):
            labels = [self.phone2idx[phone["phone"]] for phone in phones]
            for label in labels:
                ret.append((word_start, word_stop, label))
        
        return ret


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels, bias=config.classifier_bias)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

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
        
        # TODO optional pooling
        # hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                
                if label_mask is not None:
                    print("Label mask is present")
                    import ipdb; ipdb.set_trace()
                    active_loss = label_mask == 1
                    active_logits = logits[active_loss]
                    active_labels = labels[active_loss]
                    loss = loss_fct(active_logits, active_labels.float())
                else:
                    loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PhoneticTargetFeatureExtractor(SequenceFeatureExtractor):
    model_input_names: List[str] = ["phones"]


@dataclass
class DataCollator:
    """
    Data collator that will dynamically pad the inputs received.
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
    model: Wav2Vec2Model
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    num_labels: int = 2
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

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

        # Calculate how many frames we have per batch item
        batch_num_samples = batch.attention_mask.sum(dim=1)
        batch_num_frames = self.model._get_feat_extract_output_lengths(batch_num_samples)

        # TODO this is imprecise due to padding. may be very minor changes in alignment
        # that may matter if we care about precise word boundary effects
        batch_max_frames = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        compression_ratio = batch_max_frames / batch["input_values"].shape[-1]

        # For frame labeling
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
        label_pad_ret = my_padder.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["label_mask"] = label_pad_ret["attention_mask"]
        batch["labels"] = label_pad_ret["phones"]

        # batch["labels"] = torch.tensor(label_features, dtype=torch.long)

        # with self.processor.as_target_processor():
        #     labels_batch = self.processor.pad(
        #         label_features,
        #         padding=self.padding,
        #         max_length=self.max_length_labels,
        #         pad_to_multiple_of=self.pad_to_multiple_of_labels,
        #         return_tensors="pt",
        #     )

        # # replace padding with -100 to ignore loss correctly
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # batch["labels"] = labels

        return batch