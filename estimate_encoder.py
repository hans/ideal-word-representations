import hydra
from omegaconf import DictConfig

from src.estimate_encoder import main as estimate_encoder


@hydra.main(config_path="conf_encoder", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    print(config)
    estimate_encoder(config)


if __name__ == "__main__":
    main()