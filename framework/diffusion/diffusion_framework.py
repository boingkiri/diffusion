import jax
import jax.numpy as jnp

import flax
from flax.training import train_state

from model.unet import UNet
from model.DiT import DiT
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema import EMA
from framework.default_diffusion import DefaultModel
# from framework.diffusion import losses

from typing import TypedDict
from tqdm import tqdm

from omegaconf import DictConfig
from functools import partial

class DiffusionFramework(DefaultModel):
    def __init__(self, config: DictConfig, rand_key, fs_obj: FSUtils, wandblog: WandBLog):
        super().__init__(config, rand_key)
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
        if model_type == "unet":
            self.model = UNet(**model_config)
        elif model_type == "dit":
            self.model = DiT(**model_config)
        state_rng, self.rand_key = jax.random.split(self.rand_key, 2)
        self.model_state = jax_utils.create_train_state(config, 'diffusion', self.model, state_rng, None)
        self.model_state = fs_obj.load_model_state("diffusion", self.model_state)
        # PMAP
        if self.pmap:
            self.model_state = flax.jax_utils.replicate(self.model_state)

        # Create ema obj
        ema_config = config.ema
        self.ema_obj = EMA(**ema_config, pmap=self.pmap)

        beta = diffusion_framework['beta']
        loss = diffusion_framework['loss']

        # DDPM perturbing configuration

        if self.noise_schedule == "linear":
            self.beta = jnp.linspace(beta[0], beta[1], self.n_timestep)
        elif self.noise_schedule == "cosine":
            s = 0.008
            def alpha_bar(t):
                return jnp.cos((t / self.n_timestep + s) / (1 + s) * jnp.pi / 2) ** 2
            beta = []
            for i in range(self.n_timestep):
                beta.append(min(1 - alpha_bar(i + 1) / alpha_bar(i), 0.999))
            self.beta = jnp.asarray(beta)
        self.alpha = 1. - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha, axis=0)
        self.sqrt_alpha = jnp.sqrt(self.alpha)
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1 - self.alpha_bar)
        self.alpha_bar_prev = jnp.cumprod(jnp.append(1., self.alpha[:-1]), axis=0)
        self.alpha_bar_next = jnp.cumprod(jnp.append(self.alpha[1:], 0.0), axis=0)
        self.posterior_variance = self.beta * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_log_variance_clipped = jnp.log(jnp.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.logvar_upper_bound = jnp.log(self.beta)
        self.logvar_lower_bound = self.posterior_log_variance_clipped
        self.posterior_mean_coef1 = self.beta * jnp.sqrt(self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_mean_coef2 = self.sqrt_alpha * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)
        
        # DDPM loss configuration
        self.loss = loss

        @jax.jit
        def loss_fn(params, data, rng_key):
            rng_key, perturbing_key, time_key = jax.random.split(rng_key, 3)
            real_noise = jax.random.normal(perturbing_key, data.shape)
            time = jax.random.randint(time_key, (data.shape[0], ), 0, self.n_timestep)
            perturbed_data, real_noise = self.q_sample(data, time, eps=real_noise)
            # Apply pmap
            pred_noise = self.model.apply(
                {'params': params}, x=perturbed_data, t=time, train=True, rngs={'dropout': rng_key})
            loss_dict = {}

            if self.learn_sigma:
                pred_noise, pred_logvar = jnp.split(pred_noise, 2, axis=-1)
            
            if self.loss == "l2":
                loss = self._l2_loss(real_noise, pred_noise)
            elif self.loss == "l1":
                loss = self._l1_loss(real_noise, pred_noise)
            elif self.loss == "vlb":
                loss = self._vlb_loss(
                    perturbed_data, real_noise, 
                    pred_noise, pred_logvar, time, stop_gradient=False)
            elif self.loss == "hybrid": # For improved DDPM
                simple_loss = self._l2_loss(real_noise, pred_noise)
                vlb_loss = self._vlb_loss(
                    perturbed_data, real_noise, 
                    pred_noise, pred_logvar, time, stop_gradient=True) / 1000.0
                loss = simple_loss + vlb_loss
                loss_dict['vlb_loss'] = vlb_loss
                loss_dict['pred_logvar'] = jnp.mean(pred_logvar)
            loss_dict['total_loss'] = loss
            return loss, loss_dict
        
        def update(state:train_state.TrainState, x0, rng):
            (_, loss_dict), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x0, rng)
            if self.pmap:
                grad = jax.lax.pmean(grad, axis_name=self.pmap_axis)
            new_state = state.apply_gradients(grads=grad)
            for loss_key in loss_dict:
                # loss_dict[loss_key] = jnp.mean(loss_dict[loss_key])
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)
            return new_state, loss_dict
        
        if self.type == "ddpm":
            self.skip_timestep = 1
            def p_sample_jit(params, perturbed_data, time, rng_key):
                time = jnp.array(time)
                time = jnp.repeat(time, perturbed_data.shape[0])

                rng_key, normal_key = jax.random.split(rng_key)
                pred_noise = self.model.apply(
                    {'params': params}, x=perturbed_data, t=time, train=False, rngs={'dropout': rng_key})

                if self.learn_sigma:
                    pred_noise, pred_logvar = jnp.split(pred_noise, 2, axis=-1)
                
                # Mean
                beta = jnp.take(self.beta, time)
                sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, time)
                eps_coef = beta / sqrt_one_minus_alpha_bar
                eps_coef = eps_coef[:, None, None, None]
                sqrt_alpha = jnp.take(self.sqrt_alpha, time)
                sqrt_alpha = sqrt_alpha[:, None, None, None]
                mean = (perturbed_data - eps_coef * pred_noise) / sqrt_alpha

                # Var
                var = beta[:, None, None, None] if not self.learn_sigma else jnp.exp(self.get_learned_logvar(pred_logvar, time))
                eps = jax.random.normal(normal_key, perturbed_data.shape)
                return_val = jnp.where(time[0] == 0, mean, mean + (var ** 0.5) * eps)
                return return_val
        elif self.type == "ddim":
            try:
                self.skip_timestep = diffusion_framework['skip_timestep']
            except:
                self.skip_timestep = 1
            def p_sample_jit(params, perturbed_data, time, next_time, rng_key):
                time = jnp.array(time)
                time = jnp.repeat(time, perturbed_data.shape[0])

                next_time = jnp.where(next_time >= 0, next_time, 0)
                next_time = jnp.array(next_time)
                next_time = jnp.repeat(next_time, perturbed_data.shape[0])

                pred_noise = self.model.apply(
                    {'params': params}, x=perturbed_data, t=time, train=False, rngs={'dropout': rng_key})
                
                # Take current time alpha  
                pred_x0 = self.predict_x0_from_eps(x_t=perturbed_data, t=time, eps=pred_noise)
                
                # Take next time alpha
                next_sqrt_alpha_bar = jnp.take(self.sqrt_alpha_bar, next_time)[:, None, None, None]
                next_sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, next_time)[:, None, None, None]
                direction_point_to_xt = next_sqrt_one_minus_alpha_bar * pred_noise
                return_val = next_sqrt_alpha_bar * pred_x0 + direction_point_to_xt

                return return_val
        
        else:
            NotImplementedError("Diffusion framework only accept 'DDPM' or 'DDIM' for now.")

        self.loss_fn = jax.jit(loss_fn)
        self.grad_fn = jax.pmap(jax.value_and_grad(self.loss_fn, has_aux=True), axis_name=self.pmap_axis)
        self.update_fn = jax.pmap(update, axis_name=self.pmap_axis)
        self.p_sample_jit = jax.pmap(p_sample_jit)

    def _l2_loss(self, real, pred):
        return jnp.mean((real - pred) ** 2)

    def _l1_loss(self, real, pred):
        return jnp.mean((real - pred) ** 2)
    
    def _vlb_loss(self, real_perturbed_data, real_epsilon, pred_epsilon, pred_logvar, time, stop_gradient=False):
        # Get real mean from real_perturbed_data
        coef1 = jnp.take(self.posterior_mean_coef1, time)[:, None, None, None]
        coef2 = jnp.take(self.posterior_mean_coef2, time)[:, None, None, None]
        real_mean = coef1 * self.predict_x0_from_eps(real_perturbed_data, time, real_epsilon) \
                    + coef2 * real_perturbed_data
        real_logvar = jnp.take(self.posterior_log_variance_clipped, time)[:, None, None, None]


        pred_mean = coef1 * self.predict_x0_from_eps(real_perturbed_data, time, pred_epsilon) \
                    + coef2 * real_perturbed_data
        pred_mean = jax.lax.stop_gradient(pred_mean) if stop_gradient else pred_mean
        pred_logvar = self.get_learned_logvar(pred_logvar, time)
        return self._kl_loss(real_mean, real_logvar, pred_mean, pred_logvar)

    def _kl_loss(self, mean1, logvar1, mean2, logvar2):
        kl= 0.5 * (
            -1.0 + 
            logvar2 - logvar1 + 
            jnp.exp(logvar1 - logvar2) + 
            ((mean1 - mean2) ** 2) * jnp.exp(-logvar2))
        return jnp.mean(kl) / jnp.log(2.0)
        # return jax.lax.pmean()
    
    def get_learned_logvar(self, pred_sigma, time):
        min_log_var = jnp.take(self.logvar_lower_bound, time)[:, None, None, None]
        max_log_var = jnp.take(self.logvar_upper_bound, time)[:, None, None, None]
        frac = (pred_sigma + 1) / 2 # The model var_values is [-1, 1] for [min_log_var, max_log_var]
        pred_log_var = min_log_var * frac + max_log_var * (1. - frac)
        return pred_log_var

    def predict_x0_from_eps(self, x_t, t, eps):
        # sqrt_alpha_bar = jnp.take(self.alpha_bar, t)[:, None, None, None]
        sqrt_alpha_bar = jnp.take(self.sqrt_alpha_bar, t)[:, None, None, None]
        sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, t)[:, None, None, None]
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
        return pred_x0

    def q_xt_x0(self, x0, t):
        mean_coeff = self.sqrt_alpha_bar[t][:, None, None, None]
        mean = mean_coeff * x0

        std = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        std = std
        return mean, std
    
    def q_sample(self, x0, t, eps=None):
        # Sample from q(x_t|x_0)
        if eps is None:
            self.rand_key, normal_key = jax.random.split(self.rand_key, 2)
            eps = jax.random.normal(normal_key, x0.shape)
        mean, std = self.q_xt_x0(x0, t)
        return mean + std * eps, eps
    
    def p_sample(self, param, xt, t):
        # Sample from p_theta(x_{t-1}|x_t)
        self.rand_key, normal_key, dropout_key = jax.random.split(self.rand_key, 3)
        return self.p_sample_jit(param, xt, t, normal_key, dropout_key)
    
    def get_model_state(self) -> TypedDict:
        self.set_ema_params_to_state()
        if self.pmap:
            return [flax.jax_utils.unreplicate(self.model_state)]
        return [self.model_state]
    
    def set_ema_params_to_state(self):
        self.model_state = self.model_state.replace(params_ema=self.ema_obj.get_ema_params())

    def fit(self, x0, cond=None, step=0):
        # batch_size = x0.shape[0]
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)
        new_state, loss_dict = self.update_fn(self.model_state, x0, dropout_key)

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
        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple)

        if self.type == "ddpm":
            pbar = tqdm(reversed(range(0, self.n_timestep, self.skip_timestep)))
            for t in pbar:
                rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
                if self.pmap:
                    rng_key = jax.random.split(rng_key, jax.local_device_count())
                    t = jnp.asarray([t] * jax.local_device_count())
                latent_sample = self.p_sample_jit(self.model_state.params_ema, latent_sample, t, rng_key)
        elif self.type == "ddim":
            seq = jnp.linspace(0, jnp.sqrt(self.n_timestep - 1), self.n_timestep // self.skip_timestep) ** 2
            seq = [int(s) for s in list(seq)]
            pbar = reversed(seq)
            next_pbar = reversed([-1] + seq[:-1])
            pbar = tqdm(pbar)
            for t, next_t in zip(pbar, next_pbar):
                rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
                if self.pmap:
                    rng_key = jax.random.split(rng_key, jax.local_device_count())
                    t = jnp.asarray([t] * jax.local_device_count())
                    next_t = jnp.asarray([next_t] * jax.local_device_count())
                latent_sample = self.p_sample_jit(self.model_state.params_ema, latent_sample, t, next_t, rng_key)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return jnp.reshape(latent_sample, (-1, *img_size))
