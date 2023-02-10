import argparse

from utils.config_utils import ConfigContainer
from framework.diffusion_framework import DiffusionFramework

from jax import random
import wandb

import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="configs", config_name="config")
def start(configs: DictConfig):
    config = ConfigContainer()
    rng = random.PRNGKey(config.get_random_seed())

    model_type = config['type']
    
    diffusion_framework = DiffusionFramework(model_type, config, rng)
    
    if args.sampling_dir is not None:
        config.set_sampling_dir(args.sampling_dir)

    if args.do_train:
        if args.model == "ldm":
            wandb.init(project="my-ldm-WIP", config=config.get_config_dict())
        elif args.model == "ddpm":
            wandb.init(project="my-ddpm-WIP", config=config.get_config_dict())

        print("Training selected")
        diffusion_framework.train()

    if args.do_sampling:
        print("Sampling selected")
        diffusion_framework.sampling_and_save(args.num_sampling)
        fid_score = diffusion_framework.fid_utils.calculate_fid(config.get_sampling_dir())
        print(fid_score)

if __name__ == "__main__":
    start()