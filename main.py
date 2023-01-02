import argparse

from utils.common_utils import get_config_from_yaml, init_setting
from training import train
from sampling import sampling_and_save


from jax import random

def start(args):
    config = get_config_from_yaml(args.config)
    rng = random.PRNGKey(config["rand_seed"])
    
    if args.sampling_dir is not None:
        config['exp']['sampling_dir'] = args.sampling_dir

    if args.do_train:
        print("Training selected")
        state, ddpm, start_step, ema_obj, rng = init_setting(config, rng)
        train(config, state, ddpm, start_step, ema_obj, rng)

    if args.do_sampling:
        print("Sampling selected")
        state, ddpm, start_step, ema_obj, rng = init_setting(config, rng)
        sampling_and_save(config, args.num_sampling, ddpm, state, rng)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_sampling", action="store_true")
    parser.add_argument("--sampling_dir", type=str)
    parser.add_argument('-n', "--num_sampling", type=int)

    args = parser.parse_args()
    start(args)