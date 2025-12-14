import hydra
from omegaconf import DictConfig
from src.run_pipeline import run_pipeline

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_pipeline(cfg)
    
if __name__ == "__main__":
    main()