import jax
import jax.numpy as jnp

import flax
from flax.training import train_state

# from model.unetpp import UNetpp
from model.unetpp import EDMPrecond
from model.unet import UNet
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema import EMA
from utils.augment_utils import AugmentPipe
from framework.default_diffusion import DefaultModel

from typing import TypedDict
from tqdm import tqdm

from omegaconf import DictConfig
from functools import partial

class EDMFramework(DefaultModel):
    def __init__(self, config: DictConfig, rand_key, fs_obj: FSUtils, wandblog: WandBLog):
        diffusion_framework = config.framework.diffusion
        self.n_timestep = diffusion_framework['n_timestep']
        self.type = diffusion_framework['type']
        self.learn_sigma = diffusion_framework['learn_sigma']
        self.noise_schedule = diffusion_framework['noise_schedule']
        self.rand_key = rand_key
        self.fs_obj = fs_obj
        self.wandblog = wandblog
        self.pmap = config.pmap
        
        if self.pmap:
            self.pmap_axis = "batch"

        # Create UNet and its state
        model_config = {**config.model.diffusion}
        model_type = model_config.pop("type")
        self.model = EDMPrecond(model_config, image_channels=model_config['image_channels'], model_type=model_type)
        # self.model = UNet(**model_config)
        state_rng, self.rand_key = jax.random.split(self.rand_key, 2)
        # self.model_state = jax_utils.create_train_state(config, 'diffusion', self.model, state_rng, None)
        self.model_state = self.init_model_state(config)
        self.model_state = fs_obj.load_model_state("diffusion", self.model_state)
        
        # Parameters
        self.sigma_min = max(diffusion_framework['sigma_min'], self.model.sigma_min)
        self.sigma_max = min(diffusion_framework['sigma_max'], self.model.sigma_max)
        self.rho = diffusion_framework['rho']
        self.S_churn = 0 if diffusion_framework['deterministic_sampling'] else diffusion_framework['S_churn']
        self.S_min = 0 if diffusion_framework['deterministic_sampling'] else diffusion_framework['S_min']
        self.S_max = float('inf') if diffusion_framework['deterministic_sampling'] else diffusion_framework['S_max']
        self.S_noise = 1 if diffusion_framework['deterministic_sampling'] else diffusion_framework['S_noise']
        # PMAP
        if self.pmap:
            self.model_state = flax.jax_utils.replicate(self.model_state)

        # Create ema obj
        ema_config = config.ema
        self.ema_obj = EMA(**ema_config, pmap=self.pmap)

        # Augmentation pipeline
        augment_rng, self.rand_key = jax.random.split(self.rand_key)
        self.augment_rate = diffusion_framework.get("augment_rate", None)
        if self.augment_rate is not None:
            self.augmentation_pipeline = AugmentPipe(
                augment_rng, p=self.augment_rate, xflip=1e8, 
                yflip=1, scale=1, rotate_frac=1, 
                aniso=1, translate_frac=1)

        @jax.jit
        def loss_fn(params, rng_key, y, augment_label):
            p_mean = -1.2
            p_std = 1.2
            sigma_data = 0.5

            rng_key, sigma_key, dropout_key = jax.random.split(rng_key, 3)
            rnd_normal = jax.random.normal(sigma_key, (y.shape[0], 1, 1, 1))
            sigma = jnp.exp(rnd_normal * p_std + p_mean)
            weight = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
            # TODO: Implement augmented pipe 
            n = jax.random.normal(rng_key, y.shape) * sigma
            
            # Network will predict D_yn (denoised dataset rather than epsilon) directly.
            D_yn = self.model.apply(
                {'params': params}, x=(y + n), sigma=sigma, train=True, rngs={'dropout': dropout_key})
            loss = weight * ((D_yn - y) ** 2)
            loss = jnp.mean(loss)

            loss_dict = {}
            loss_dict['total_loss'] = loss
            return loss, loss_dict
        
        def update(state:train_state.TrainState, rng, x0, augment_label):
            (_, loss_dict), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params, rng, x0, augment_label)
            if self.pmap:
                grad = jax.lax.pmean(grad, axis_name=self.pmap_axis)
            new_state = state.apply_gradients(grads=grad)
            for loss_key in loss_dict:
                # loss_dict[loss_key] = jnp.mean(loss_dict[loss_key])
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)
            return new_state, loss_dict
        
        def p_sample_jit(params, x_cur, rng_key, gamma, t_cur, t_next):
            rng_key, dropout_key, dropout_key_2 = jax.random.split(rng_key, 3)

            # Increase noise temporarily.
            t_hat = t_cur + gamma * t_cur
            noise = jax.random.normal(rng_key, x_cur.shape) * self.S_noise
            x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * noise

            # Euler step
            denoised = self.model.apply(
                {'params': params}, x=x_hat, sigma=t_hat, augment_labels= None, train=False, rngs={'dropout': dropout_key})
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            def second_order_corrections(x_next, t_next, x_hat, t_hat, d_cur, rng_key):
                denoised = self.model.apply(
                    {'params': params}, x=x_next, sigma=t_next, augment_labels= None, train=False, rngs={'dropout': rng_key})
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                return x_next
            x_result = jax.lax.cond(t_next != 0.0, 
                                    second_order_corrections, 
                                    lambda x_next, t_next, x_hat, t_hat, d_cur, dropout_key_2: x_next, 
                                    x_next, t_next, x_hat, t_hat, d_cur, dropout_key_2)
            return x_result

        self.loss_fn = loss_fn
        self.grad_fn = jax.pmap(jax.value_and_grad(self.loss_fn, has_aux=True), axis_name=self.pmap_axis)
        self.update_fn = jax.pmap(update, axis_name=self.pmap_axis)
        self.p_sample_jit = jax.pmap(p_sample_jit)

    
    def p_sample(self, param, xt, t):
        # Sample from p_theta(x_{t-1}|x_t)
        self.rand_key, normal_key, dropout_key = jax.random.split(self.rand_key, 3)
        return self.p_sample_jit(param, xt, t, normal_key, dropout_key)
    
    def init_model_state(self, config: DictConfig):
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        input_format = jnp.ones([1, *config.dataset.data_size])

        augment_dim = config.model.diffusion.get("augment_dim", None)
        augment_labels = jnp.ones([1, augment_dim]) if augment_dim is not None else None
        params = self.model.init(rng_dict, x=input_format, sigma=jnp.ones([1,]), augment_labels=augment_labels, train=False)['params']

        return jax_utils.create_train_state(config, 'diffusion', self.model.apply, params)

    def get_model_state(self):
        if self.pmap:
            return [flax.jax_utils.unreplicate(self.model_state)]
        return [self.model_state]
    

    def fit(self, x0, cond=None, step=0):
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)

        # Augment pipeline
        x0, labels = self.augmentation_pipeline(x0) if self.augment_rate is not None else x0, None
        new_state, loss_dict = self.update_fn(self.model_state, dropout_key, x0, labels)

        for loss_key in loss_dict:
            loss_dict[loss_key] = jnp.mean(loss_dict[loss_key])

        self.model_state = new_state

        # Update EMA parameters
        self.model_state, _ = self.ema_obj.ema_update(self.model_state, step)

        return_dict = {}
        return_dict.update(loss_dict)
        self.wandblog.update_log(return_dict)
        return return_dict
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None):
        if self.pmap:
            latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        else:
            latent_sampling_tuple = (num_image, *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        step_indices = jnp.arange(self.n_timestep)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = jnp.append(t_steps, jnp.zeros_like(t_steps[0]))
        pbar = tqdm(zip(t_steps[:-1], t_steps[1:]))

        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * t_steps[0]
        for t_cur, t_next in pbar:
            rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
            gamma_val = jnp.minimum(jnp.sqrt(2) - 1, self.S_churn / self.n_timestep)
            gamma = jnp.where(self.S_min <= t_cur and t_cur <= self.S_max,
                            gamma_val, 0)
            if self.pmap:
                rng_key = jax.random.split(rng_key, jax.local_device_count())
                t_cur = jnp.asarray([t_cur] * jax.local_device_count())
                t_next = jnp.asarray([t_next] * jax.local_device_count())
                gamma = jnp.asarray([gamma] * jax.local_device_count())
            latent_sample = self.p_sample_jit(self.model_state.params_ema, latent_sample, rng_key, gamma, t_cur, t_next)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample
