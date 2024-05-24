from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import Optional, Iterator

from datasets import Dataset, IterableDataset
from IsoScore.IsoScore import IsoScore
import numpy as np
from scipy.spatial.distance import cdist, pdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from transformers import PreTrainedModel, PretrainedConfig, EvalPrediction
from transformers.file_utils import ModelOutput
from tqdm.auto import tqdm, trange

from src.datasets.speech_equivalence import SpeechHiddenStateDataset, SpeechEquivalenceDataset


L = logging.getLogger(__name__)


class RNNModel(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, type="lstm"):
        super(RNNModel, self).__init__()
        if num_layers == 0:
            assert hidden_dim == input_dim, f"Hidden dim {hidden_dim} must match input dim {input_dim} if num_layers is 0"
            self.rnn = nn.Identity()
        else:
            fn = nn.LSTM if type == "lstm" else nn.RNN
            self.rnn = fn(num_layers=num_layers, input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc: nn.Module = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        lengths = lengths.to("cpu")
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        if isinstance(self.rnn, nn.Identity):
            packed_output = packed_input
        else:
            packed_output, _ = self.rnn(packed_input)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output
    

class ContrastiveEmbeddingObjective(nn.Module):
    def __init__(self, tau=0.1, batch_soft_negatives=False,
                 regularization=None):
        super(ContrastiveEmbeddingObjective, self).__init__()
        self.tau = tau
        self.batch_soft_negatives = batch_soft_negatives

        regularization_type = regularization[0] if isinstance(regularization, (tuple, list)) else regularization
        if regularization_type not in [None, "covariance", "spectral_norm"]:
            raise ValueError(f"Unknown regularization {regularization}")
        self.regularization_type = regularization_type
        self.regularization = regularization

    def forward(self, embeddings, pos_embeddings, neg_embeddings,
                reduction="mean",
                embeddings_class=None):
        pos_dist = F.cosine_similarity(embeddings, pos_embeddings, dim=1)
        neg_dist = F.cosine_similarity(embeddings, neg_embeddings, dim=1)

        pos_loss = -torch.log(torch.exp(pos_dist / self.tau))
        neg_loss = -torch.log(torch.exp(-neg_dist / self.tau))

        if reduction == "mean":
            pos_loss = pos_loss.mean()
            neg_loss = neg_loss.mean()

        if self.batch_soft_negatives:
            if embeddings_class is None:
                raise ValueError("Must provide embeddings_class if using batch_soft_negatives")
            if reduction != "mean":
                raise ValueError("Must use mean reduction with batch_soft_negatives")

            # Compute pairwise cosine similarity matrix
            anchors = embeddings
            soft_negatives = embeddings  # TODO could also include hard positives/negatives of other examples
            pairwise_cosine_sim = F.cosine_similarity(anchors.unsqueeze(1), soft_negatives.unsqueeze(0), dim=2)

            # Evaluate upper triangle
            mask = torch.triu(embeddings_class.unsqueeze(1) != embeddings_class.unsqueeze(0), diagonal=1)
            pairwise_cosine_sim = pairwise_cosine_sim[mask]

            soft_neg_loss = -torch.log(torch.exp(-pairwise_cosine_sim / self.tau)).mean()
            # Guard for NaN (will happen if there are no soft negatives)
            if torch.isnan(soft_neg_loss):
                soft_neg_loss = torch.tensor(0.0, device=soft_neg_loss.device)

            neg_loss += soft_neg_loss
        
        train_loss = pos_loss + neg_loss

        ### Regularization

        if self.training:
            regularization_loss = torch.tensor(0.0, device=embeddings.device)
            if self.regularization_type == "covariance":
                scaler = float(self.regularization[1]) if isinstance(self.regularization, (list, tuple)) else 1.0
                if embeddings.shape[0] >= 2 and embeddings.shape[1] >= 2:
                    # Regularize by penalizing divergence of embedding covariance matrix from identity
                    # i.e., encourage axes to be decorrelated
                    embedding_cov = torch.cov(embeddings.T)

                    # Only penalize covariance off diagonal
                    embedding_cov = embedding_cov[~torch.eye(embedding_cov.shape[0], dtype=bool)]
                    target = torch.zeros_like(embedding_cov)

                    # TODO tune the scale of this relative to training loss
                    regularization_loss = scaler * F.mse_loss(embedding_cov, target)

                    if torch.isnan(regularization_loss):
                        import ipdb; ipdb.set_trace()
                
            elif self.regularization_type == "spectral_norm":
                # Do nothing -- this happens outside the loss function, which only has access to embeddings
                pass

            train_loss += regularization_loss

        return train_loss


@dataclass
class ContrastiveEmbeddingModelOutput(ModelOutput):
    loss: torch.Tensor

    embeddings: Optional[torch.Tensor] = None
    embeddings_hard_positive: Optional[torch.Tensor] = None
    embeddings_hard_negative: Optional[torch.Tensor] = None


@dataclass
class ContrastiveEmbeddingModelConfig(PretrainedConfig):
    base_model_ref: str = "facebook/wav2vec2-base"
    base_model_layer: int = 6

    # equivalence-classing config
    equivalence_classer: str = "phoneme_within_word_prefix"

    # NN config
    max_length: int = 20
    input_dim: int = 4
    num_layers: int = 1
    hidden_dim: int = 256
    output_dim: int = 4
    tau: float = 0.1

    in_batch_soft_negatives: bool = True
    """
    If True, use all other examples in the batch as soft negatives unless they have
    the same class. If False, only use the hard negative example.
    """

    regularization: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_compatible_with(self, dataset: SpeechHiddenStateDataset):
        return self.base_model_ref == dataset.model_name and \
               self.input_dim == dataset.hidden_size


class ContrastiveEmbeddingModel(PreTrainedModel):
    config_class = ContrastiveEmbeddingModelConfig
    main_input_name = "example"

    def __init__(self, config):
        super().__init__(config)
        self.rnn = RNNModel(config.num_layers,
                            config.input_dim,
                            config.hidden_dim,
                            config.output_dim)
        
        if self.config.regularization == "spectral_norm":
            # Register spectral norm parameterization on RNN output layer.
            self.rnn.fc = spectral_norm(self.rnn.fc)
        
    def is_compatible_with(self, dataset: SpeechHiddenStateDataset):
        return self.config.is_compatible_with(dataset)

    def forward(self, example, example_length, pos, pos_length, neg, neg_length,
                loss_reduction="mean",
                return_loss=True, return_embeddings=True,
                in_batch_soft_negatives=None,
                **kwargs):
        embeddings, pos_embeddings, neg_embeddings = self.compute_batch_embeddings(
            example, example_length, pos, pos_length, neg, neg_length)
        
        in_batch_soft_negatives = in_batch_soft_negatives if in_batch_soft_negatives is not None \
            else self.config.in_batch_soft_negatives

        loss_fn = ContrastiveEmbeddingObjective(
            tau=self.config.tau,
            batch_soft_negatives=in_batch_soft_negatives,
            regularization=self.config.regularization)
        loss = loss_fn(embeddings, pos_embeddings, neg_embeddings,
                       reduction=loss_reduction,
                       embeddings_class=kwargs.get("example_idx"))

        if not return_embeddings:
            return ContrastiveEmbeddingModelOutput(loss=loss)
        else:
            return ContrastiveEmbeddingModelOutput(
                loss=loss,
                embeddings=embeddings,
                embeddings_hard_positive=pos_embeddings,
                embeddings_hard_negative=neg_embeddings
            )

    def compute_embeddings(self, example, example_length, return_all_states=False):
        embeddings = self.rnn(example, example_length)

        if return_all_states:
            # Mask states beyond the length of each example.
            max_batch_length = example_length.max().item()
            mask = torch.arange(max_batch_length).expand(example.shape[0], -1).to(example.device) >= example_length.unsqueeze(1)
            embeddings[mask] = 0
            return embeddings
        else:
            # Gather final embedding of each sequence
            embeddings = torch.gather(embeddings, 1, (example_length - 1).reshape(-1, 1, 1).expand(-1, 1, embeddings.shape[-1])).squeeze(1)
            return embeddings
        
    def compute_batch_embeddings(self, example, example_length, pos, pos_length, neg, neg_length):
        return self.compute_embeddings(example, example_length), \
                self.compute_embeddings(pos, pos_length), \
                self.compute_embeddings(neg, neg_length)


def get_sequence(F, start_index, end_index, max_length):
    if end_index - start_index + 1 > max_length:
        start_index = end_index - max_length + 1
    sequence = F[start_index:end_index + 1]

    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence)
    
    # Pad on right if necessary
    if len(sequence) < max_length:
        pad_size = max_length - len(sequence)
        padding = torch.zeros(pad_size, F.shape[1]).to(F)
        sequence = torch.cat((sequence, padding), dim=0)
    
    return sequence


def iter_dataset(equiv_dataset: SpeechEquivalenceDataset,
                 hidden_state_dataset: SpeechHiddenStateDataset,
                 max_length: int,
                 num_examples: Optional[int] = None,
                 layer: Optional[int] = None,
                 select_idxs: Optional[list[int]] = None,
                 infinite=True) -> Iterator[dict]:
    if layer is None and hidden_state_dataset.num_layers > 1:
        raise ValueError("Must specify layer if there are multiple layers")
    F = hidden_state_dataset.get_layer(layer if layer is not None else 0)
    
    lengths = torch.minimum(equiv_dataset.lengths, torch.tensor(max_length))
    # TODO understand why we have zero-length examples
    lengths[lengths == 0] = 1

    if select_idxs is not None:
        assert (equiv_dataset.Q[select_idxs] != -1).all()
        non_null_frames = torch.tensor(select_idxs)
    else:
        non_null_frames = (equiv_dataset.Q != -1).nonzero(as_tuple=True)[0]
        if num_examples is not None:
            non_null_frames = np.random.choice(non_null_frames.numpy(), num_examples, replace=False)

    # Pre-compute mapping from equivalence class to frame indices
    equiv_class_to_idxs = {}
    equiv_class_to_complement_idxs = {}
    for idx in range(equiv_dataset.num_classes):
        equiv_class_to_idxs[idx] = (equiv_dataset.Q == idx).nonzero(as_tuple=True)[0]
        equiv_class_to_complement_idxs[idx] = ((equiv_dataset.Q != -1) & (equiv_dataset.Q != idx)).nonzero(as_tuple=True)[0]

    # infinite generation
    while True:
        for i in non_null_frames:
            if lengths[i] == -1:
                continue

            pos_indices = equiv_class_to_idxs[equiv_dataset.Q[i].item()]
            neg_indices = equiv_class_to_complement_idxs[equiv_dataset.Q[i].item()]

            if len(pos_indices) > 1 and len(neg_indices) > 0:
                pos_indices = pos_indices[pos_indices != i]
                pos_idx = random.choice(pos_indices)
                neg_idx = random.choice(neg_indices)

                # Extract sequences
                example_seq = get_sequence(F, equiv_dataset.S[i], i, max_length)
                pos_seq = get_sequence(F, equiv_dataset.S[pos_idx], pos_idx, max_length)
                neg_seq = get_sequence(F, equiv_dataset.S[neg_idx], neg_idx, max_length)

                # Sanity chcks
                assert lengths[i] > 0
                assert lengths[pos_idx] > 0
                assert lengths[neg_idx] > 0
                assert equiv_dataset.Q[i] != -1
                assert equiv_dataset.Q[pos_idx] != -1
                assert equiv_dataset.Q[neg_idx] != -1

                yield {
                    "example": example_seq,
                    "example_idx": i,
                    "example_class": equiv_dataset.Q[i],
                    "example_length": lengths[i],

                    "pos": pos_seq,
                    "pos_idx": pos_idx,
                    "pos_class": equiv_dataset.Q[pos_idx],
                    "pos_length": lengths[pos_idx],

                    "neg": neg_seq,
                    "neg_idx": neg_idx,
                    "neg_class": equiv_dataset.Q[neg_idx],
                    "neg_length": lengths[neg_idx],
                }

        if not infinite:
            break


def prepare_dataset(equiv_dataset: SpeechEquivalenceDataset,
                    hidden_state_dataset: SpeechHiddenStateDataset,
                    max_length: int,
                    layer: Optional[int] = None,
                    eval_fraction=0.1,
                    max_eval_size=10000,
                    **kwargs) -> tuple[int, IterableDataset, Dataset]:
    """
    Prepare a negative-sampling dataset for contrastive embedding learning.

    Returns train and test split datasets.
    """

    all_idxs = (equiv_dataset.Q != -1).nonzero(as_tuple=True)[0].numpy()
    all_idxs = np.random.permutation(all_idxs)
    
    eval_num_samples = int(eval_fraction * len(all_idxs))
    if max_eval_size is not None and eval_num_samples > max_eval_size:
        L.info(f"Reducing eval set size from {eval_num_samples} to {max_eval_size}")
        eval_num_samples = max_eval_size

    test_idxs, train_idxs = all_idxs[:eval_num_samples], all_idxs[eval_num_samples:]

    dataset_kwargs = {
        "equiv_dataset": equiv_dataset,
        "hidden_state_dataset": hidden_state_dataset,
        "max_length": max_length,
        "layer": layer,
        **kwargs
    }

    train_dataset = IterableDataset.from_generator(
        iter_dataset, gen_kwargs={**dataset_kwargs, "select_idxs": train_idxs,
                                  "infinite": True})
    test_dataset = Dataset.from_generator(
        iter_dataset, gen_kwargs={**dataset_kwargs, "select_idxs": test_idxs,
                                  "infinite": False})

    return len(all_idxs), train_dataset, test_dataset


def compute_embeddings(model: ContrastiveEmbeddingModel,
                       equiv_dataset: SpeechEquivalenceDataset,
                       hidden_state_dataset: SpeechHiddenStateDataset,
                       batch_size=16,
                       device=None) -> torch.Tensor:
    """
    Compute integrator embeddings for a given model on a speech
    equivalence classing dataset.
    """
    assert model.is_compatible_with(hidden_state_dataset)
    if device is not None:
        model = model.to(device)
    device = model.device

    model_representations = []

    batch_size = 16
    # TODO this is a hack -- better to have the dataset explicitly represent what
    # layers it retains after subsetting
    if hidden_state_dataset.num_layers > 1:
        F = hidden_state_dataset.get_layer(model.config.base_model_layer)
    else:
        F = hidden_state_dataset.get_layer(0)
    
    F = F.to(device)
    lengths = equiv_dataset.lengths.to(device)

    for batch_start in trange(0, hidden_state_dataset.num_frames, batch_size):
        batch_idxs = torch.arange(batch_start, min(batch_start + batch_size, hidden_state_dataset.num_frames))
        batch = torch.stack([get_sequence(F, equiv_dataset.S[idx], idx, model.config.max_length)
                             for idx in batch_idxs])
        
        lengths_batch = torch.minimum(lengths[batch_idxs], torch.tensor(model.config.max_length))
        # HACK
        lengths_batch[lengths_batch <= 0] = 1

        with torch.no_grad():
            model_representations.append(model.compute_embeddings(batch, lengths_batch))

    return torch.cat(model_representations, dim=0)


def load_or_compute_embeddings(model, equiv_dataset, model_dir, equiv_dataset_path,
                               embedding_cache_dir="out/embedding_cache", **kwargs):
    embedding_cache_path = Path(embedding_cache_dir) / f"{model_dir.replace('/', '-')}-{equiv_dataset_path.replace('/', '-')}.npy"
    
    if Path(embedding_cache_path).exists():
        model_representations = np.load(embedding_cache_path)
    else:
        model_representations = compute_embeddings(model, equiv_dataset,
                                                   **kwargs)
        model_representations = model_representations.numpy()
        np.save(embedding_cache_path, model_representations)
    return model_representations


def compute_embedding_loss(embeddings, pos_embeddings, neg_embeddings, tau=0.1):
    pos_dist = cdist(embeddings, pos_embeddings, metric="cosine")
    neg_dist = cdist(embeddings, neg_embeddings, metric="cosine")

    pos_loss = -np.log(np.exp(pos_dist / tau))
    neg_loss = -np.log(np.exp(-neg_dist / tau))

    return pos_loss.mean() + neg_loss.mean()


def compute_embedding_alignment(embeddings, pos_embeddings, metric="euclidean"):
    """
    Compute average Euclidean distance between embeddings and their positive anchors.
    """
    if metric == "cosine":
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        pos_embeddings /= np.linalg.norm(pos_embeddings, ord=2, axis=1, keepdims=True)
        return 1 - (embeddings * pos_embeddings).sum(axis=1).mean()
    elif metric == "euclidean":
        return np.linalg.norm(embeddings - pos_embeddings, ord=2, axis=1).mean()
    else:
        raise ValueError(f"Unknown metric {metric}")


def compute_embedding_uniformity(embeddings: np.ndarray, metric="euclidean"):
    """
    Compute uniformity a la Wang & Isola (2020)
    """
    distances = pdist(embeddings, metric=metric)
    return distances.mean()


def compute_embedding_axis_correlation(embeddings: np.ndarray):
    corrs = np.corrcoef(embeddings.T)
    return np.abs(corrs[np.triu_indices_from(corrs, k=1)]).mean()


def compute_metrics(p: EvalPrediction):
    assert len(p.predictions) == 3
    embeddings, hard_positive_embeddings, hard_negative_embeddings = p.predictions

    assert isinstance(p.label_ids, np.ndarray)
    example_classes = p.label_ids
    assert embeddings.shape[0] == example_classes.shape[0]

    return {
        "eval_loss": compute_embedding_loss(embeddings, hard_positive_embeddings, hard_negative_embeddings),
        "eval_embedding_norm": np.linalg.norm(embeddings, ord=2, axis=1).mean(),
        "eval_embedding_alignment": compute_embedding_alignment(embeddings, hard_positive_embeddings, metric="euclidean"),
        "eval_embedding_alignment_cosine": compute_embedding_alignment(embeddings, hard_positive_embeddings, metric="cosine"),
        "eval_embedding_uniformity": compute_embedding_uniformity(embeddings),
        "eval_embedding_corr": compute_embedding_axis_correlation(embeddings),
        "eval_embedding_isoscore": IsoScore(embeddings),
    }