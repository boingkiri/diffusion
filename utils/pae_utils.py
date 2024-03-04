if __name__=="__main__":
    import sys
    sys.path.append("../")

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from hydra import compose

from framework.diffusion.edm_framework import EDMFramework
from framework.diffusion.consistency_framework import CMFramework
from utils.common_utils import load_dataset_from_tfds
from utils.fs_utils import FSUtils


from tqdm import tqdm 

import wandb
import io

# Calculate Pixel alignment error (PAE)

class PAEUtils():
    def __init__(self, consistency_config, wandb_obj) -> None:
        denoiser_config_path = "config_denoiser"
        self.denoiser_config = compose(config_name=denoiser_config_path) # Assume that the hydra is already initialized
        self.rng = jax.random.PRNGKey(42)
        denoiser_rng, self.rng = jax.random.split(self.rng)
        denoiser_fs_obj = FSUtils(self.denoiser_config)
        self.denoiser_framework = EDMFramework(self.denoiser_config, denoiser_rng, denoiser_fs_obj, wandb_obj)

        self.n_timestep = 18
        self.num_denoiser_samples = consistency_config.sampling_batch # 256
        self.num_consistency_samples_per_denoiser_sample = 32

        sigma_min = 0.02
        sigma_max = 80
        rho = 7
        sweep_timestep = jnp.arange(self.n_timestep)
        self.t_steps = (sigma_max ** (1 / rho) + sweep_timestep / (self.n_timestep - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    
    def reset_dataset(self):
        # Assume that the CIFAR10 would only be used  
        self.datasets = load_dataset_from_tfds(self.denoiser_config, "cifar10", self.num_denoiser_samples, 1, x_flip=False, shuffle=False)

    def viz_sample(self, sample):
        sample = jnp.asarray(sample)
        # sample = sample.reshape((-1, 32, 32, 3))
        # sample = sample[:2]
        
        sample = (sample + 1) / 2
        sample = jnp.clip(sample, 0, 1)
        sample = jnp.asarray(sample * 255, dtype=jnp.uint8)

        image_samples = []
        img_containter = np.zeros((32 * 2, 32, 3), dtype=np.uint8)
        for i in range(sample.shape[0]):
            img_set = sample[i]
            denoiser_img = img_set[0]
            consistency_img = img_set[1]
            img_containter[:32, :] = np.asarray(denoiser_img)
            img_containter[32:, :] = np.asarray(consistency_img)
            image_samples.append(img_containter)
            img_containter = np.zeros((32 * 2, 32, 3), dtype=np.uint8)

        image_samples = [Image.fromarray(np.asarray(image_samples[i])) for i in range(len(image_samples))]
        # num_samples = sample.shape[0]
        # image_samples = jnp.zeros((32 * 2, 32 * (num_samples // 2), 3), dtype=jnp.uint8)
        # for i in range(num_samples):
        #     x = i % 2
        #     y = i // 2
        #     image_samples = image_samples.at[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32].set(sample[i])
        # image_samples = np.array(image_samples)
        return image_samples
    
    def calculate_pae(self, consistency_framework: CMFramework, step: int):
        self.reset_dataset()
        data = next(self.datasets)
        data = data[0][:, 0, ...] # 256 samples (8 * 32)

        error_x_label = []
        error_y_label = []
        sample_images = []

        print(f"Start to calculate for step {step}.")

        rng = self.rng

        for timestep in range(0, self.n_timestep):
            print(f"Start {timestep} / {self.n_timestep}")
            
            rng, sampling_rng = jax.random.split(rng)
            noise = jax.random.normal(sampling_rng, shape=data.shape) * self.t_steps[timestep]
            # noise = jax.random.normal(sampling_rng, shape=data.shape) * timestep

            # Get denoiser output
            denoiser_output = self.denoiser_framework.sampling_denoiser(self.num_denoiser_samples, original_data=data, sweep_timesteps=timestep, noise=noise)

            # Get consistency output
            consistency_output = consistency_framework.sampling_cm_intermediate(self.num_denoiser_samples, original_data=data, sweep_timesteps=timestep, noise=noise)
            # consistency_output = jnp.expand_dims(consistency_output, axis=1)
            consistency_output = jnp.reshape(consistency_output, data.shape)

            # Sample multiple datapoints
            sampling_list = []
            for i in tqdm(range(self.num_consistency_samples_per_denoiser_sample)): # 32
                rng, sampling_rng = jax.random.split(rng)

                new_noise = jax.random.normal(sampling_rng, shape=consistency_output.shape) * self.t_steps[timestep]
                # new_noise = jax.random.normal(sampling_rng, shape=consistency_output.shape) * timestep
                second_consistency_output = consistency_framework.sampling_cm_intermediate(
                    self.num_denoiser_samples, original_data=consistency_output, sweep_timesteps=timestep, noise=new_noise)
                sampling_list.append(second_consistency_output)
                # second_consistency_output = jnp.reshape(second_consistency_output, data.shape)

            # sampling_list = jnp.concatenate(sampling_list, axis=0)
            sampling_list = jnp.stack(sampling_list, axis=0)
            sampling_list = jnp.mean(sampling_list, axis=0)
            denoiser_output = jnp.reshape(denoiser_output, (self.num_denoiser_samples, 32, 32, 3))
            pixel_alignment_error = jnp.mean(jnp.abs(sampling_list - denoiser_output), axis=(-1, -2, -3))

            sample_images.append([denoiser_output[0], sampling_list[0]])
            # second_consistency_output_empirical_mean = jnp.mean(sampling_list, axis=0)
            # error = jnp.mean(jnp.abs(second_consistency_output_empirical_mean - denoiser_output), axis=(-1, -2, -3))
            print("Total mean of error: ", jnp.mean(pixel_alignment_error))

            error_x_label.append(timestep)
            error_y_label.append(jnp.mean(pixel_alignment_error))

        total_pixel_alignment_error = jnp.mean(jnp.array(error_y_label))
        total_pixel_alignment_error_var = jnp.var(jnp.array(error_y_label))
        print(f"Total pixel alignment error: {total_pixel_alignment_error}")
        print(f"Total pixel alignment error variance: {total_pixel_alignment_error_var}")

        data = [[x, y] for (x, y) in zip(error_x_label, error_y_label)]
        wandb_table = wandb.Table(data=data, columns=['timestep', 'PAE'])
        wandb.log(
            {"Pixel Alignment error": wandb.plot.bar(
                wandb_table, "timestep", "PAE", title="Pixel Alignment error")
            },
            step=step
        )

        wandb.log({
            "train/Total Pixel Alignment error": total_pixel_alignment_error,
            "train/Total Pixel Alignment error variance": total_pixel_alignment_error_var
        }, step=step)

        np_image = self.viz_sample(sample_images)
        wandb_image = [wandb.Image(image, caption="Sample images") for image in np_image]
        wandb.log({"train/Sample images": wandb_image}, step=step)
        return total_pixel_alignment_error, total_pixel_alignment_error_var
            


# WARNING: This unit test is working only when the file is placed in the root directory 
# (in diffusion directory, not in utils directory)
if __name__=="__main__":
    from hydra import initialize

    from utils.log_utils import WandBLog

    # consistency_config_path = "config_consistency"
    consistency_config_path = "config"

    args ={
            "project": "test",
            "name": "pae_unit_test",
        }
    wandb.init(**args)

    # with initialize(config_path="../configs") as cfg:
    with initialize(config_path="configs") as cfg:
        consistency_config = compose(config_name=consistency_config_path)
        consistency_config["do_training"] = False
        consistency_fs_obj = FSUtils(consistency_config)
        wandb_obj = WandBLog()
        pae_utils = PAEUtils(consistency_config, wandb_obj)

        framework_rng = jax.random.PRNGKey(42)
        consistency_framework = CMFramework(consistency_config, framework_rng, consistency_fs_obj, wandb_obj)
        mean, var = pae_utils.calculate_pae(consistency_framework, 0)
        print(mean, var)