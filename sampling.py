import jax
from tqdm import tqdm

from utils import fs_utils, common_utils

from PIL import Image
import os

def sampling(ddpm, ema_obj, n_samples, image_size, rng):
    # Sampling
    n_timesteps = ddpm.n_timestep
    sampling_key, rng = jax.random.split(rng)
    x_t = jax.random.normal(sampling_key, (n_samples, *image_size))

    sampling_bar = tqdm(reversed(range(n_timesteps)))
    for t in sampling_bar:
        x_t = ddpm.p_sample(ema_obj.get_ema_params(), x_t, t)
        sampling_bar.set_description(f"Sampling: {t}")
    return x_t, rng

def sampling_and_save(config, total_samples, ddpm, ema_obj, rng):
    current_sampling = 0
    batch_size = config['sampling']['batch_size']
    image_size = common_utils.get_image_size_from_dataset(config['dataset'])
    sampling_dir = fs_utils.get_sampling_dir()
    while current_sampling < total_samples:
        images, rng = sampling(ddpm, ema_obj, batch_size, image_size, rng)
        images = common_utils.unnormalize_minus_one_to_one(images)
        for image in images:
            im = Image.fromarray(image)
            sample_path = os.path.join(sampling_dir, f"{current_sampling}.png")
            im.save(sample_path)
            current_sampling += 1
    