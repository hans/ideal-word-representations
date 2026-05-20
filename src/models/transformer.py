"""
Utilities for working with transformer base models.
"""

from hydra.utils import instantiate
import transformers


SUPPORTED_MODEL_REF_PREFIXES = (
    "facebook/wav2vec2",
    "LeBenchmark/wav2vec2",
)


def prepare_processor(config):
    if not any(p in config.base_model.model_ref for p in SUPPORTED_MODEL_REF_PREFIXES):
        raise NotImplementedError(
            f"prepare_processor does not support model_ref={config.base_model.model_ref}")

    tokenizer = instantiate(config.tokenizer)
    feature_extractor = instantiate(config.feature_extractor,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=False)
    processor = transformers.Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor