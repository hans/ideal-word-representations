from typing import Tuple

from datasets import Dataset
import numpy as np
import torch



class TypeLevelSemanticModel:
    """
    Model for encoding word types from the text corpus.
    These word encodings will serve as semantic targets for the
    lexical access model.
    """
    
    def __call__(self, dataset: Dataset) -> Tuple[list[str], np.ndarray]:
        raise NotImplementedError()


class RandomSemanticModel(TypeLevelSemanticModel):
    """
    Designate a word type's "meaning" as a random point on the unit sphere.
    """

    def __init__(self, embedding_size: int, normalize=True):
        self.embedding_size = embedding_size
        self.normalize = normalize
    
    def __call__(self, dataset: Dataset) -> Tuple[list[str], np.ndarray]:
        all_words = set()
        def update_all_words(item):
            all_words.update(set(item["word_detail"]["utterance"]))
            return None
        dataset.map(update_all_words)
        all_words = sorted(all_words)

        word_representations = np.random.randn(len(all_words), self.embedding_size)
        if self.normalize:
            word_representations /= np.linalg.norm(word_representations, ord=2, axis=1, keepdims=True)

        return all_words, word_representations