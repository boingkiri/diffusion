# from diffusion.framework.unifying_framework import UnifyingFramework
from framework.unifying_framework import UnifyingFramework

import jax
from jax import random


import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="configs", config_name="config")
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

        print("Training selected")
        diffusion_framework.train()

    if config.do_sampling:
        print("Sampling selected")
        diffusion_framework.sampling_and_save(config.num_sampling)
        fid_score = diffusion_framework.fid_utils.calculate_fid(config.exp.sampling_dir)
        print(fid_score)
    
if __name__ == "__main__":
    start()