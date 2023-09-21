from jax import random

import wandb
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from framework.unifying_framework import UnifyingFramework

import os
import argparse

@hydra.main(config_path="configs", config_name="config")
def start(config: DictConfig):
    rng = random.PRNGKey(config.rand_seed)
    model_type = config.type
    
    print("-------------------Config Setting---------------------")
    print(OmegaConf.to_yaml(config))
    print("------------------------------------------------------")
    diffusion_framework = UnifyingFramework(model_type, config, rng)


    if config.do_training:
        name = config['exp_name']
        tags = config["tags"]
        project_name = f"my-{config.type}-WIP"
        args ={
            "project": project_name,
            "name": name,
            "tags": tags,
            "config": {**config}
        }
        wandb.init(**args)
        print("Training selected")
        diffusion_framework.train()

    if config.do_sampling:
        print("Sampling selected")
        diffusion_framework.sampling_and_save(config.num_sampling)
        fid_score = diffusion_framework.fid_utils.calculate_fid(config.exp.sampling_dir)
        print(fid_score)
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Diffuion")
    # parser.add_argument("--config", action="store", type=str, default="config")
    # args = parser.parse_args()

    # config_path = "configs"
    
    # with initialize(version_base=None, config_path=config_path) as cfg:
    #     # if args.config != "config":
    #     #     args.config = os.path.join("examples", args.config)
    #     #     cfg = compose(config_name=args.config)
    #     #     cfg = cfg.examples
    #     # else:
    #     #     cfg = compose(config_name=args.config)

    #     cfg = compose(config_name=args.config)
    #     start(cfg)
    start()