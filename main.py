import argparse

from utils.common_utils import get_config_from_yaml, init_setting
from framework.diffusion_framework import DiffusionFramework


from jax import random

def start(args):
    args.model = args.model.lower()
    if config == None:
        if args.model == "ldm":
            args.config = "configs/ldm.yml"
        elif args.model == "ddpm":
            args.config = "configs/ddpm.yml"

    config = get_config_from_yaml(args.config)
    rng = random.PRNGKey(config["rand_seed"])
    
    diffusion_framework = DiffusionFramework(args.model, config, rng)
    
    if args.sampling_dir is not None:
        config['exp']['sampling_dir'] = args.sampling_dir

    if args.do_train:
        print("Training selected")
        diffusion_framework.train()

    if args.do_sampling:
        print("Sampling selected")
        # sampling_and_save(config, args.num_sampling, ddpm, state, rng)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_sampling", action="store_true")
    parser.add_argument("--sampling_dir", type=str)
    parser.add_argument('-n', "--num_sampling", type=int)
    parser.add_argument("-m", "--model", type=str)

    args = parser.parse_args()
    start(args)