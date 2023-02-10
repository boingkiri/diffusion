from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def start(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__=="__main__":
    start()