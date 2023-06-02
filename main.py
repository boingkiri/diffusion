from jax import random

import wandb
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from framework.unifying_framework import UnifyingFramework

import os
import argparse

def start(config: DictConfig):
    rng = random.PRNGKey(config.rand_seed)
    model_type = config.type
    
    print("-------------------Config Setting---------------------")
    print(OmegaConf.to_yaml(config))
    print("------------------------------------------------------")
    diffusion_framework = UnifyingFramework(model_type, config, rng)
    
    # if jax.devices

    if config.do_training:
        if config.type == "ldm":
            wandb.init(project="my-ldm-WIP", config={**config})
        elif config.type == "ddpm":
            wandb.init(project="my-ddpm-WIP", config={**config})
        elif config.type == "ddim":
            wandb.init(project="my-ddim-WIP", config={**config})
        elif config.type == "edm":
            wandb.init(project="my-edm-WIP", config={**config})
        elif config.type == "cm":
            wandb.init(project="my-cm-WIP", config={**config})
        elif config.type == "cm_diffusion":
            wandb.init(project="my-cm-diffusion-WIP", config={**config})

        print("Training selected")
        diffusion_framework.train()

    if config.do_sampling:
        print("Sampling selected")
        diffusion_framework.sampling_and_save(config.num_sampling)
        fid_score = diffusion_framework.fid_utils.calculate_fid(config.exp.sampling_dir)
        print(fid_score)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffuion")
    parser.add_argument("--config", action="store", type=str, default="config")
    args = parser.parse_args()

    config_path = "configs"
    
    with initialize(version_base=None, config_path=config_path) as cfg:
        # if args.config != "config":
        #     args.config = os.path.join("examples", args.config)
        #     cfg = compose(config_name=args.config)
        #     cfg = cfg.examples
        # else:
        #     cfg = compose(config_name=args.config)

        cfg = compose(config_name=args.config)
        start(cfg)