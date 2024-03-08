from argparse import ArgumentParser
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import torch

from src.models import get_best_checkpoint
from src.models.integrator import compute_embeddings, ContrastiveEmbeddingModel


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    checkpoint_dir = get_best_checkpoint(config.model.output_dir)
    model: ContrastiveEmbeddingModel = ContrastiveEmbeddingModel.from_pretrained(checkpoint_dir)

    hidden_state_path = config.base_model.hidden_state_path
    with open(hidden_state_path, "rb") as f:
        hidden_states = torch.load(f)

    equiv_dataset_path = config.equivalence.path
    with open(equiv_dataset_path, "rb") as f:
        equiv_dataset = torch.load(f)

    embeddings = compute_embeddings(model, hidden_states, equiv_dataset)
    embeddings.numpy()
    
    out_path = Path(HydraConfig.get().runtime.output_dir) / "embeddings.npy"
    np.save(out_path, embeddings)


if __name__ == "__main__":
    main()