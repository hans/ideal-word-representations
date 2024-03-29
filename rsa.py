import hydra
from omegaconf import DictConfig

from src.analysis_rsa import main as rsa


@hydra.main(config_path="conf_encoder", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    print(config)
    rsa(config)


if __name__ == "__main__":
    main()