if __name__=="__main__":
    import sys
    sys.path.append("../")

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf
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
    def __init__(self, consistency_config, wandb_obj=None) -> None:
        self.rng = jax.random.PRNGKey(42)

        self.n_timestep = 18
        self.num_denoiser_samples = consistency_config.sampling_batch # 256
        self.num_consistency_samples_per_denoiser_sample = 32

        sigma_min = 0.02
        sigma_max = 80
        rho = 7
        sweep_timestep = jnp.arange(self.n_timestep)
        self.t_steps = (sigma_max ** (1 / rho) + sweep_timestep / (self.n_timestep - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        self.sampled_data = self.sample_target_data()
        # self.calculate_and_save_ideal_denoiser()

    def gather_all_data_in_dataset(self):
        def normalize_to_minus_one_to_one(image):
            return image * 2 - 1

        def normalize_channel_scale(image, label):
            image = tf.cast(image, tf.float32)
            image = (image / 255.0)
            image = normalize_to_minus_one_to_one(image)
            return image, label
        
        def augmentation(image, label):
            image, label = normalize_channel_scale(image, label)
            return image, label
        ds = tfds.load("cifar10", as_supervised=True)
        train_ds, _ = ds['train'], ds['test']
        train_ds = train_ds.map(augmentation)
        train_ds = train_ds.batch(16)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        train_ds = map(lambda data: jax.tree_map(lambda x: x._numpy(), data), train_ds)
        return train_ds

    def sample_target_data(self):
        self.reset_dataset()
        data = next(self.datasets)
        data = data[0][:, 0, ...]
        return data

    def calculate_ideal_denoiser_for_each_sample(self, sample, timestep, idx, rng):
        sigma_fn = lambda timestep, idx: (sigma_min ** (1 / rho) + idx / (timestep - 1) * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho
        sigma_min = 0.002
        sigma_max = 80
        rho = 7
        gathered_dataset = self.gather_all_data_in_dataset()
        sample = np.concatenate([sample] * 16, axis=0)

        sigma = sigma_fn(timestep, idx)
        denominator = 2 * sigma ** 2

        rng, sample_rng = jax.random.split(rng)
        perturbed_sample = sample + jax.random.normal(rng, shape=sample.shape) * sigma

        total_denominator = 0
        total_numerator = 0
        for y in gathered_dataset:
            diff = perturbed_sample - y
            diff_square = diff ** 2
            diff_square_sum = jnp.sum(diff_square, axis=(1, 2, 3))
            gaussian_exp_component = -diff_square_sum / denominator
            
            total_denominator = jax.scipy.special.logsumexp(gaussian_exp_component)



    def calculate_and_save_ideal_denoiser(self):
        sigma_min = 0.002
        sigma_max = 80
        rho = 7
        timesteps = [10, 20, 40, 80, 160, 320, 640, 1280]

        for timestep in timesteps:
            total_timestep = timestep + 1
            timestep_range = range(0, total_timestep, timestep // 10)
            for idx in timestep_range:

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
    if False:
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
    if True:
        consistency_config_path = "config"

        with initialize(config_path="../configs") as cfg:
        # with initialize(config_path="configs") as cfg:
            consistency_config = compose(config_name=consistency_config_path)
            consistency_config["do_training"] = False
            # consistency_fs_obj = FSUtils(consistency_config)
            pae_utils = PAEUtils(consistency_config)
            breakpoint()
