from pathlib import Path

import datasets
import torch

from beartype import beartype
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from omegaconf import DictConfig

from src.datasets.speech_equivalence import SpeechHiddenStateDataset


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir)
    
    hidden_state_path = config.base_model.hidden_state_path
    with open(hidden_state_path, "rb") as f:
        hidden_state_dataset: SpeechHiddenStateDataset = torch.load(f)

    equiv_dataset = call(config.equivalence, _partial_=True)(
                         dataset=dataset,
                         hidden_states=hidden_state_dataset)

    with open(Path(HydraConfig.get().runtime.output_dir) / "equivalence.pkl", "wb") as f:
        torch.save(equiv_dataset, f)


if __name__ == "__main__":
    main()