import argparse

from utils.common_utils import get_config_from_yaml, init_setting
from training import train
from sampling import sampling_and_save


from jax import random

def start(args):
    config = get_config_from_yaml(args.config)
    rng = random.PRNGKey(config["rand_seed"])
    state, ddpm, start_step, ema_obj, rng = init_setting(config, rng)
    
    if args.do_train:
        print("Training selected")
        train(config, state, ddpm, start_step, ema_obj, rng)
    else:
        print("Sampling selected")
        sampling_and_save(config, args.num_sampling, ddpm, ema_obj, rng)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config")
    parser.add_argument("--do_train", type=bool, required=True)
    parser.add_argument("--num_sampling", type=int)

    args = parser.parse_args()
    start(args)