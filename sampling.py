from tqdm import tqdm
import numpy as np
import jax

from PIL import Image
import os

from utils import fs_utils, common_utils


def sampling(ddpm, params, n_samples, image_size, rng):
    # Sampling
    n_timesteps = ddpm.n_timestep
    sampling_key, rng = jax.random.split(rng)
    x_t = jax.random.normal(sampling_key, (n_samples, *image_size))

    sampling_bar = tqdm(reversed(range(n_timesteps)))
    for t in sampling_bar:
        x_t = ddpm.p_sample(params, x_t, t)
        sampling_bar.set_description(f"Sampling: {t}")
    return x_t, rng

def save_result_as_fig(save_path_dir, images, starting_pos=0):
    current_sampling = 0

    images = common_utils.unnormalize_minus_one_to_one(images)
    images = np.clip(images, 0, 1)
    images = images * 255
    images = np.array(images).astype(np.uint8)
    for image in images:
        im = Image.fromarray(image)
        sample_path = os.path.join(save_path_dir, f"{starting_pos + current_sampling}.png")
        im.save(sample_path)
        current_sampling += 1
    return current_sampling


def sampling_and_save(config, total_samples, ddpm, state, rng, sampling_dir=None):
    current_sampling = 0
    batch_size = config['sampling']['batch_size']
    image_size = common_utils.get_image_size_from_dataset(config['dataset'])
    if sampling_dir is None:
        sampling_dir = fs_utils.get_sampling_dir(config)
    while current_sampling < total_samples:
        images, rng = sampling(ddpm, state.params_ema, batch_size, image_size, rng)
        current_sampling += save_result_as_fig(sampling_dir, images, current_sampling)
    