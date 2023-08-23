from jax import random

import wandb
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from framework.unifying_framework import UnifyingFramework

import os
import argparse

from datetime import datetime 

def start(config: DictConfig):
    rng = random.PRNGKey(config.rand_seed)
    model_type = config.type
    
    # Print config setting
    print("-------------------Config Setting---------------------")
    print(OmegaConf.to_yaml(config))
    print("------------------------------------------------------")

    # Set unifying framework
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    config['now'] = now_str
    diffusion_framework = UnifyingFramework(model_type, config, rng)

    if config.do_training:
        wandb.init(
            project=config['logger']['logger_project_name'],
            name=now_str + "_" + config['logger']['logger_experiment_name'],
            config={**config},
            config_exclude_keys=[
                "logger",
                "exp",
                "do_training",
                "do_sampling",
                "num_sampling",
            ],
            tags=config['logger']['logger_tags'],
        )

        print("Training selected")
        diffusion_framework.train()

    if config.do_sampling:
        print("Sampling selected")
        diffusion_framework.sampling_and_save(config.num_sampling)
        fid_score = diffusion_framework.fid_utils.calculate_fid(config.exp.sampling_dir)
        print(fid_score)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffuion")
    
    # Set config path
    parser.add_argument("--config", action="store", type=str, default="config")

    # Set do_training and do_sampling flag with default value (True)
    parser.add_argument("--do_training", action="store_true")
    parser.add_argument("--no-do_training", dest="do_training", action="store_false")
    parser.set_defaults(do_training=True)
    parser.add_argument("--do_sampling", action="store_true")
    parser.add_argument("--no-do_sampling", dest="do_training", action="store_false")
    parser.set_defaults(do_sampling=True)

    args = parser.parse_args()

    config_path = "configs"
    
    with initialize(version_base=None, config_path=config_path) as cfg:
        cfg = compose(config_name=args.config)

        # Set do_training and do_sampling flag
        cfg.do_training = args.do_training
        cfg.do_sampling = args.do_sampling

        start(cfg)