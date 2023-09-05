import jax
import jax.numpy as jnp

import flax
from flax.training import train_state

from model.modelContainer import ModelContainer
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema.ema_ddpm import DDPMEMA
from framework.default_diffusion import DefaultModel

from tqdm import tqdm

from omegaconf import DictConfig
from functools import partial

class DDPMFramework(DefaultModel):
    def __init__(self, config: DictConfig, rand_key, fs_obj: FSUtils, wandblog: WandBLog):
        super().__init__()
        diffusion_framework = config['framework']['diffusion']
        self.n_timestep = diffusion_framework['n_timestep']
        self.type = diffusion_framework['type']
        self.learn_sigma = diffusion_framework['learn_sigma']
        self.noise_schedule = diffusion_framework['noise_schedule']

        # pmap
        # self.pmap_axis = "batch"
        
        # Create ema obj
        ema_config = config.ema
        self.ema_obj = DDPMEMA(**ema_config)

        beta = diffusion_framework['beta']

        # DDPM perturbing configuration
        if self.noise_schedule == "linear":
            self.beta = jnp.linspace(beta[0], beta[1], self.n_timestep)
        elif self.noise_schedule == "cosine":
            s = 0.008
            max_beta = 0.999
            ts = jnp.linspace(0, 1, self.n_timestep + 1)
            alphas_bar = jnp.cos((ts + s) / (1 + s) * jnp.pi /2) ** 2
            alphas_bar = alphas_bar/alphas_bar[0]
            self.beta = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            self.beta = jnp.clip(self.beta, 0, max_beta)

        self.alpha = 1. - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha, axis=0)
        self.alpha_bar_prev = jnp.cumprod(jnp.append(1., self.alpha[:-1]), axis=0)
        self.alpha_bar_next = jnp.cumprod(jnp.append(self.alpha[1:], 0.0), axis=0)

        self.sqrt_alpha = jnp.sqrt(self.alpha)
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1 - self.alpha_bar)

        self.posterior_variance = self.beta * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_log_variance_clipped = jnp.log(jnp.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = self.beta * jnp.sqrt(self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_mean_coef2 = self.sqrt_alpha * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)

        self.logvar_upper_bound = jnp.log(self.beta)
        self.logvar_lower_bound = self.posterior_log_variance_clipped

        # DDPM loss configuration
        self.loss = diffusion_framework['loss']

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

        def update(carry_state, x0):
            (rng, state) = carry_state
            rng, new_rng = jax.random.split(rng)
            (_, loss_dict), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x0, rng)

            grad = jax.lax.pmean(grad)
            new_state = state.apply_gradients(grads=grad)
            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key])
            new_state = self.ema_obj.ema_update(new_state)
            new_carry_state = (new_rng, new_state)
            return new_carry_state, loss_dict
        
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
                pred_x0 = self.predict_x0_from_eps(x_t=perturbed_data, t=time, eps=pred_noise)
                pred_x0 = jnp.clip(pred_x0, -1, 1)
                coef1 = jnp.take(self.posterior_mean_coef1, time)[:, None, None, None]
                coef2 = jnp.take(self.posterior_mean_coef2, time)[:, None, None, None]
                mean = coef1 * pred_x0 + coef2 * perturbed_data

                beta = jnp.take(self.beta, time)
                sigma = beta[:, None, None, None] ** 0.5 if not self.learn_sigma \
                        else jnp.exp(0.5 * self.get_learned_logvar(pred_logvar, time))
                eps = jax.random.normal(normal_key, perturbed_data.shape)
                return_val = jnp.where(time[0] == 0, mean, mean + sigma * eps)
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

        self.update_fn = jax.pmap(partial(jax.lax.scan, update), donate_argnums=1)
        self.p_sample_jit = jax.pmap(p_sample_jit)

    def _l2_loss(self, real, pred):
        return jnp.mean((real - pred) ** 2)

    def _l1_loss(self, real, pred):
        return jnp.mean(jnp.abs(real - pred))
    
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
    
    def get_learned_logvar(self, pred_sigma, time):
        min_log_var = jnp.take(self.logvar_lower_bound, time)[:, None, None, None]
        max_log_var = jnp.take(self.logvar_upper_bound, time)[:, None, None, None]
        frac = (pred_sigma + 1) / 2 # The model var_values is [-1, 1] for [min_log_var, max_log_var]
        pred_log_var = max_log_var * frac + min_log_var * (1. - frac)
        return pred_log_var

    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_alpha_bar = jnp.take(self.sqrt_alpha_bar, t)[:, None, None, None]
        sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, t)[:, None, None, None]
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
        return pred_x0

    def q_sample(self, x0, t, eps=None):
        # Sample from q(x_t|x_0)
        if eps is None:
            self.rand_key, normal_key = jax.random.split(self.rand_key, 2)
            eps = jax.random.normal(normal_key, x0.shape)
        # Statistics of q(x_t|x_0) (mean, std)
        mean = self.sqrt_alpha_bar[t][:, None, None, None] * x0
        std = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return mean + std * eps, eps
    
    def get_model_state(self) -> dict:
        # self.set_ema_params_to_state()
        # return [flax.jax_utils.unreplicate(self.model_state)]
        return {"ddpm": flax.jax_utils.unreplicate(self.model_state)}

    def init_model_state(self, config: DictConfig):
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        if config.type == "ldm" and config.framework['train_idx'] == 2:
            f_value = len(config.model.autoencoder.ch_mults)
            z_dim = config.model.autoencoder.embed_dim
            input_format_shape = input_format.shape
            input_format = jnp.ones(
                [input_format_shape[0], 
                input_format_shape[1] // f_value, 
                input_format_shape[2] // f_value, 
                z_dim])
        input_format = jnp.ones([1, *config.dataset.data_size])
        params = self.model.init(rng_dict, x=input_format, t=jnp.ones([1,]), train=False)['params']

        return jax_utils.create_train_state(config, 'diffusion', self.model.apply, params)
    

    def fit(self, x0, cond=None, step=0):
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)
        new_carry, loss_dict_stack = self.update_fn((dropout_key, self.model_state), x0)
        (_, new_state) = new_carry

        loss_dict = flax.jax_utils.unreplicate(loss_dict_stack)
        for loss_key in loss_dict:
            loss_dict[loss_key] = jnp.mean(loss_dict[loss_key])

        self.model_state = new_state

        return_dict = {}
        return_dict.update(loss_dict)
        self.wandblog.update_log(return_dict)
        return return_dict
    
    def sampling(self, num_image, rng_key: jax.random.PRNGKeyArray, img_size=(32, 32, 3), original_data=None):
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)

        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)
        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple)

        if self.type == "ddpm":
            pbar = tqdm(reversed(range(0, self.n_timestep, self.skip_timestep)))
            for t in pbar:
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
                rng_key = jax.random.split(rng_key, jax.local_device_count())
                t = jnp.asarray([t] * jax.local_device_count())
                next_t = jnp.asarray([next_t] * jax.local_device_count())
                latent_sample = self.p_sample_jit(self.model_state.params_ema, latent_sample, t, next_t, rng_key)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample
