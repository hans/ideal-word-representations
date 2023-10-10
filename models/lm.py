"""
Tools for extracting representations from a language model.
"""

from minicons.scorer import IncrementalLMScorer
import numpy as np
import torch
from torch import nn


def encode_text(model, tokenizer, sentences: list[str], layers: list[int],
                pooling_method="mean", pool_within_word=True):
    """
    If `pool_within_word`, then words which are tokenized into multiple
    tokens will have representations pooled within-word, across-token.
    """

    encoded = tokenizer(sentences, padding="longest", return_tensors="pt")

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"].unsqueeze(-1)

    with torch.no_grad():
        output = model(**encoded)

    print(output.hidden_states[0].shape, attention_mask.shape)
    hidden_states = [output.hidden_states[i] * attention_mask for i in layers]

    assert pooling_method == "mean", "unsupported"
    
    if pool_within_word:
        # Reshape to array of sentence tensors which are num_layers * num_words * d
        sentence_lengths = [sentence.count(" ") + 1 for sentence in sentences]
        sentence_token_lengths = [len([i for i in ids if i != 0]) for ids in input_ids.tolist()]
        hidden_states_new = []
        for i, (sentence_length, sentence_length_tokens) in enumerate(zip(sentence_lengths, sentence_token_lengths)):
            sentence_hiddens = []
            sentence_word_ids = np.array(encoded.word_ids(i))
            for layer in hidden_states:
                sentence_token_hiddens = layer[i, :sentence_length_tokens, :]
                sentence_word_hiddens = []
                for j in range(sentence_length):
                    word_token_idxs = (sentence_word_ids == j).nonzero()[0]
                    word_representations = sentence_token_hiddens[word_token_idxs, :]
                    sentence_word_hiddens.append(word_representations.mean(dim=0))

                sentence_hiddens.append(torch.stack(sentence_word_hiddens))

            hidden_states_new.append(torch.stack(sentence_hiddens))
    else:
        raise NotImplementedError()
    
    return hidden_states_new


def next_word_distribution(scorer: IncrementalLMScorer, sentences: list[str],
                           aggregate_within_word=True):
    # HACK if tokenization happens differently internally, we won't know
    encoded = scorer.tokenizer(sentences, padding="longest", return_tensors="pt")
    next_word_distribution = scorer.next_word_distribution(sentences)

    # ret = []
    # for sentence, next_word_dist_i in