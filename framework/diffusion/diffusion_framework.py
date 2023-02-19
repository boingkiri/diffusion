import jax
import jax.numpy as jnp

from flax.training import train_state

from model.unet import UNet
from model.DiT import DiT
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema import EMA
from framework.default_diffusion import DefaultModel

from typing import TypedDict
from tqdm import tqdm

from omegaconf import DictConfig

class DiffusionFramework(DefaultModel):
    def __init__(self, config: DictConfig, rand_key, fs_obj: FSUtils, wandblog: WandBLog):
        super().__init__(config, rand_key)
        diffusion_framework = config.framework.diffusion
        self.n_timestep = diffusion_framework['n_timestep']
        self.type = diffusion_framework['type']
        self.rand_key = rand_key
        self.fs_obj = fs_obj
        self.wandblog = wandblog

        # Create UNet and its state
        model_config = {**config.model.diffusion}
        model_type = model_config.pop("type")
        if model_type == "unet":
            self.model = UNet(**model_config)
        elif model_type == "dit":
            self.model = DiT(**model_config)
        state_rng, self.rand_key = jax.random.split(self.rand_key, 2)
        self.model_state = jax_utils.create_train_state(config, 'diffusion', self.model, state_rng)
        self.model_state = fs_obj.load_model_state("diffusion", self.model_state)

        # Create ema obj
        # ema_config = config.get_ema_config()
        ema_config = config.ema
        self.ema_obj = EMA(self.model_state.params_ema, **ema_config)

        beta = diffusion_framework['beta']
        loss = diffusion_framework['loss']

        # DDPM perturbing configuration
        self.beta = jnp.linspace(beta[0], beta[1], self.n_timestep)
        self.alpha = 1. - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha, axis=0)
        self.sqrt_alpha = jnp.sqrt(self.alpha)
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1 - self.alpha_bar)
        
        # DDPM loss configuration
        self.loss = loss

        def loss_fn(params, perturbed_data, time, real_noise, dropout_key):
            pred_noise = self.model.apply(
                {'params': params}, x=perturbed_data, t=time, train=True, rngs={'dropout': dropout_key})
            if self.loss == "l2":
                loss = jnp.mean((pred_noise - real_noise) ** 2)
            elif self.loss == "l1":
                loss = jnp.mean(jnp.absolute(pred_noise - real_noise))
            return loss
        
        def update_grad(state:train_state.TrainState, grad):
            return state.apply_gradients(grads=grad)
        
        if self.type == "ddpm":
            self.skip_timestep = 1
            def p_sample_jit(params, perturbed_data, time, dropout_key, normal_key):
                time = jnp.array(time)
                time = jnp.repeat(time, perturbed_data.shape[0])

                pred_noise = self.model.apply(
                    {'params': params}, x=perturbed_data, t=time, train=False, rngs={'dropout': dropout_key})
                beta = jnp.take(self.beta, time)
                sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, time)
                eps_coef = beta / sqrt_one_minus_alpha_bar
                eps_coef = eps_coef[:, None, None, None]

                # alpha = jnp.take(self.alpha, time)
                sqrt_alpha = jnp.take(self.sqrt_alpha, time)
                sqrt_alpha = sqrt_alpha[:, None, None, None]

                mean = (perturbed_data - eps_coef * pred_noise) / sqrt_alpha

                var = beta[:, None, None, None]
                eps = jax.random.normal(normal_key, perturbed_data.shape)

                return_val = jnp.where(time[0] == 0, mean, mean + (var ** 0.5) * eps)

                return return_val
        elif self.type == "ddim":
            try:
                self.skip_timestep = diffusion_framework['skip_timestep']
            except:
                self.skip_timestep = 1
            
            def p_sample_jit(params, perturbed_data, time, next_time, dropout_key, normal_key):
                time = jnp.array(time)
                time = jnp.repeat(time, perturbed_data.shape[0])

                next_time = jnp.where(next_time >= 0, next_time, 0)
                next_time = jnp.array(next_time)
                next_time = jnp.repeat(next_time, perturbed_data.shape[0])

                pred_noise = self.model.apply(
                    {'params': params}, x=perturbed_data, t=time, train=False, rngs={'dropout': dropout_key})
                
                # Take current time alpha  
                sqrt_alpha_bar = jnp.take(self.sqrt_alpha_bar, time)[:, None, None, None]
                sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, time)[:, None, None, None]
                pred_x0 = (perturbed_data - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
                
                # Take next time alpha
                next_sqrt_alpha_bar = jnp.take(self.sqrt_alpha_bar, next_time)[:, None, None, None]
                next_sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, next_time)[:, None, None, None]
                direction_point_to_xt = next_sqrt_one_minus_alpha_bar * pred_noise
                return_val = next_sqrt_alpha_bar * pred_x0 + direction_point_to_xt

                return return_val
        
        else:
            NotImplementedError("Diffusion framework only accept 'DDPM' or 'DDIM' for now.")
        
        self.loss_fn = jax.jit(loss_fn)
        self.grad_fn = jax.jit(jax.value_and_grad(self.loss_fn))
        self.update_grad = jax.jit(update_grad)
        self.p_sample_jit = jax.jit(p_sample_jit)
        # self.p_sample_jit = p_sample_jit

    def q_xt_x0(self, x0, t):
        mean_coeff = jnp.take(self.sqrt_alpha_bar, t)
        mean = mean_coeff[:, None, None, None] * x0

        std = jnp.take(self.sqrt_one_minus_alpha_bar, t)
        std = std[:, None, None, None]
        return mean, std
    
    def q_sample(self, x0, t, eps=None):
        # Sample from q(x_t|x_0)
        if eps is None:
            # TODO: need to check
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
        return [self.model_state]
    
    def set_ema_params_to_state(self):
        self.model_state = self.model_state.replace(params_ema=self.ema_obj.get_ema_params())

    def fit(self, x0, cond=None, step=0):
        batch_size = x0.shape[0]
        key, int_key, normal_key, dropout_key = jax.random.split(self.rand_key, 4)
        self.rand_key = key

        noise = jax.random.normal(normal_key, x0.shape)
        t = jax.random.randint(int_key, (batch_size, ), 0, self.n_timestep)
        xt, noise = self.q_sample(x0, t, eps=noise)
        
        loss, grad = self.grad_fn(self.model_state.params, xt, t, noise, dropout_key)
        new_state = self.update_grad(self.model_state, grad)

        self.model_state = new_state

        # Update EMA parameters
        self.ema_obj.ema_update(self.model_state.params, step)

        return_dict = {
            "diffusion_loss": loss
        }
        self.wandblog.update_log(return_dict)
        return return_dict
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None):
        # self.set_ema_params_to_state()
        latent_sampling_tuple = (num_image, *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)
        if original_data is None:
            latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple)
        else:
            t = jnp.array([self.n_timestep - 1] * num_image)
            perturbed_data = self.q_sample(original_data, t)[0]
            latent_sample = perturbed_data

        # pbar = tqdm(reversed(range(0, self.n_timestep, self.skip_timestep)))
        if self.type == "ddpm":
            pbar = tqdm(reversed(range(0, self.n_timestep, self.skip_timestep)))
            for t in pbar:
                normal_key, dropout_key, self.rand_key = jax.random.split(self.rand_key, 3)
                latent_sample = self.p_sample_jit(self.ema_obj.get_ema_params(), latent_sample, t, normal_key, dropout_key)
        elif self.type == "ddim":
            seq = jnp.linspace(0, jnp.sqrt(self.n_timestep - 1), self.n_timestep // self.skip_timestep) ** 2
            seq = [int(s) for s in list(seq)]
            pbar = reversed(seq)
            # pbar = tqdm(reversed(list(range(0, self.n_timestep, self.skip_timestep)) + [self.n_timestep - 1]))
            next_pbar = reversed([-1] + seq[:-1])
            pbar = tqdm(pbar)
            for t, next_t in zip(pbar, next_pbar):
                normal_key, dropout_key, self.rand_key = jax.random.split(self.rand_key, 3)
                latent_sample = self.p_sample_jit(self.ema_obj.get_ema_params(), latent_sample, t, next_t, normal_key, dropout_key)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample
    
