import jax
import jax.numpy as jnp

import flax
from flax.training import checkpoints

from model.unetpp import CMPrecond, ScoreDistillPrecond
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema.ema_cm import CMEMA
from utils.augment_utils import AugmentPipe
from framework.default_diffusion import DefaultModel
# import lpips
import lpips_jax

from tqdm import tqdm

from omegaconf import DictConfig
from functools import partial

import os
from typing import Any

from clu import parameter_overview

class CMFramework(DefaultModel):
    def __init__(self, config: DictConfig, rand_key, fs_obj: FSUtils, wandblog: WandBLog):
        diffusion_framework: DictConfig = config.framework.diffusion
        self.n_timestep = diffusion_framework['n_timestep']
        self.type = diffusion_framework['type']
        self.learn_sigma = diffusion_framework['learn_sigma']
        self.rand_key = rand_key
        self.fs_obj = fs_obj
        self.wandblog = wandblog        
        self.pmap_axis = "batch"


        # Create UNet and its state
        model_config = {**config.model.diffusion}
        model_type = model_config.pop("type")

        head_config = {**config.model.head}
        head_type = head_config.pop("type")
        
        self.model = CMPrecond(model_config, 
                               image_channels=model_config['image_channels'], 
                               model_type=model_type, 
                               sigma_min=diffusion_framework['sigma_min'],
                               sigma_max=diffusion_framework['sigma_max'])
        self.head = ScoreDistillPrecond(head_config, 
                               image_channels=model_config['image_channels'], 
                               sigma_min=diffusion_framework['sigma_min'],
                               sigma_max=diffusion_framework['sigma_max'],
                               model_type=head_type)
        
        self.model_state, self.head_state = self.init_model_state(config)
        # print(parameter_overview.get_parameter_overview(self.head_state.params))
        # breakpoint()
        # self.model_state = fs_obj.load_model_state("diffusion", self.model_state, checkpoint_dir='pretrained_models/cd_750k')
        checkpoint_dir =  "experiments/cm_distillation_ported_from_torch_ve/checkpoints"
        self.model_state = fs_obj.load_model_state("diffusion", self.model_state, checkpoint_dir=checkpoint_dir)
        # self.head_state = fs_obj.load_model_state("diffusion", self.head_state)
        # self.model_state = flax.jax_utils.replicate(self.model_state)
        self.head_state = flax.jax_utils.replicate(self.head_state)
        

        # Parameters
        self.sigma_min = diffusion_framework['sigma_min']
        self.sigma_max = diffusion_framework['sigma_max']
        self.rho = diffusion_framework['rho']
        self.S_churn = 0 
        self.S_min = 0 
        self.S_max = float('inf') 
        self.S_noise = 1
        self.mu_0 = diffusion_framework.params_ema_for_training[0]
        self.s_0 = diffusion_framework.params_ema_for_training[1]
        self.s_1 = diffusion_framework.params_ema_for_training[2]
        self.target_model_ema_decay = diffusion_framework.target_model_ema_decay

        # Set step indices for distillation
        step_indices = jnp.arange(self.n_timestep)
        self.t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho


        # Create ema obj
        ema_config = config.ema
        self.ema_obj = CMEMA(**ema_config)


        # Define loss functions
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16', replicate=False)
        
        @jax.jit
        def distill_loss_fn(head_params, y, rng_key):
            rng_key, step_key, solver_key, dropout_key = jax.random.split(rng_key, 4)

            # sigma = jax.random.uniform(step_key, (y.shape[0],), 
            #                            minval=self.sigma_min ** (1 / self.rho), maxval=self.sigma_max ** (1 / self.rho)) ** self.rho
            # sigma = sigma[:, None, None, None]
            
            idx = jax.random.randint(step_key, (y.shape[0], ), minval=1, maxval=self.n_timestep)
            sigma = self.t_steps[idx][:, None, None, None]
            prev_sigma = self.t_steps[idx-1][:, None, None, None]
            
            noise = jax.random.normal(rng_key, y.shape)
            perturbed_x = y + sigma * noise

            # Get consistency function values
            D_x, aux = self.model.apply(
                {'params': self.model_state.params}, x=perturbed_x, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            F_x, t_emb, last_x_emb = aux
            
            denoised = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=True, augment_labels=None, rngs={'dropout': dropout_key})
            
            # model_fn = lambda x, t: self.model.apply(
            #     {'params': self.model_state.params}, x=x, sigma=t, 
            #     train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            # _, distill_loss, _ = jax.jvp(model_fn, (perturbed_x, sigma), (dx_dt, jnp.ones_like(sigma)), has_aux=True)
            # distill_loss = jnp.mean(distill_loss ** 2)
            
            # prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * dx_dt
            # predicted_score = - (perturbed_x - denoised) ** 2 / (sigma ** 2)
            dx_dt = (perturbed_x - denoised) / sigma
            prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * dx_dt # Euler step
            
            prev_D_x, aux = self.model.apply(
                {'params': self.model_state.params}, x=prev_perturbed_x, sigma=prev_sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            # Monitor distillation loss
            l2_dist = jnp.mean((D_x - prev_D_x) ** 2)
            
            output_shape = (y.shape[0], 224, 224, y.shape[-1])
            D_x = jax.image.resize(D_x, output_shape, "bilinear")
            prev_D_x = jax.image.resize(prev_D_x, output_shape, "bilinear")
            D_x = (D_x + 1) / 2.0
            prev_D_x = (prev_D_x + 1) / 2.0
            lpips_dist = jnp.mean(self.perceptual_loss(D_x, prev_D_x))
            
            # Optimize DSM loss
            # dsm_loss = jnp.mean((dx_dt - noise) ** 2)
            dsm_loss = jnp.mean((denoised - y) ** 2)

            loss_dict = {}
            loss_dict['l2_dist'] = l2_dist
            loss_dict['lpips_dist'] = lpips_dist
            loss_dict['dsm_loss'] = dsm_loss
            return dsm_loss, loss_dict
        
        
        # Define update function
        def update(carry_state, x0):
            (rng, head_state) = carry_state
            rng, new_rng = jax.random.split(rng)
            
            # Update head (for multiple times)
            (_, loss_dict), head_grad = jax.value_and_grad(distill_loss_fn, has_aux=True)(head_state.params, x0, rng)
            head_grad = jax.lax.pmean(head_grad, axis_name=self.pmap_axis)
            new_head_state = head_state.apply_gradients(grads=head_grad)
            
            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)

            # Update EMA for sampling
            new_head_state = self.ema_obj.ema_update(new_head_state)

            new_carry_state = (new_rng, new_head_state)
            return new_carry_state, loss_dict


        # Define p_sample_jit functions for sampling
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        def p_sample_jit(head_params, x_cur, rng_key, gamma, t_cur, t_prev):
            rng_key, dropout_key, dropout_key_2 = jax.random.split(rng_key, 3)

            # Increase noise temporarily.
            t_hat = t_cur + gamma * t_cur
            noise = jax.random.normal(rng_key, x_cur.shape) * self.S_noise
            x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * noise

            # Euler step
            D_x, aux = self.model.apply(
                {'params': self.model_state.params_ema}, x=x_hat, sigma=t_hat, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            F_x, t_emb, last_x_emb = aux
            
            # dx_dt = self.head.apply(
            #     {'params': head_params}, x=x_hat, sigma=t_hat, F_x=F_x, t_emb=t_emb, last_x_emb=last_x_emb,
            #     train=False, augment_labels=None, rngs={'dropout': dropout_key}
            # )
            # euler_x_prev = x_hat + (t_prev - t_hat) * dx_dt

            denoised = self.head.apply(
                {'params': head_params}, x=x_hat, sigma=t_hat, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=False, augment_labels=None, rngs={'dropout': dropout_key}
            )

            # predicted_score = - (x_hat - denoised) ** 2 / (t_hat ** 2)
            d_cur = (x_hat - denoised) / t_hat
            euler_x_prev = x_hat + (t_prev - t_hat) * d_cur

            # Apply 2nd order correction.
            def second_order_corrections(euler_x_prev, t_prev, x_hat, t_hat, d_cur, rng_key):
                D_x, aux = self.model.apply(
                    {'params': self.model_state.params_ema}, x=euler_x_prev, sigma=t_prev,
                    train=False, augment_labels=None, rngs={'dropout': rng_key})
                
                F_x, t_emb, last_x_emb = aux
                
                # dx_dt_prime = self.head.apply(
                #     {'params': head_params}, x=euler_x_prev, sigma=t_prev, F_x=F_x, t_emb=t_emb, last_x_emb=last_x_emb,
                #     train=False, augment_labels=None, rngs={'dropout': dropout_key}
                # )
                denoised = self.head.apply(
                    {'params': head_params}, x=euler_x_prev, sigma=t_prev, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                    train=False, augment_labels=None, rngs={'dropout': dropout_key}
                )
                d_prime = (euler_x_prev - denoised) / t_prev
                heun_x_prev = x_hat + 0.5 * (t_prev - t_hat) * (d_cur + d_prime)
                return heun_x_prev
            
            # heun_x_prev = jax.lax.cond(t_prev != 0.0,
            #                         second_order_corrections,
            #                         lambda euler_x_prev, t_prev, x_hat, t_hat, dx_dt, rng_key: euler_x_prev,
            #                         euler_x_prev, t_prev, x_hat, t_hat, dx_dt, dropout_key_2)
            heun_x_prev = jax.lax.cond(t_prev != 0.0,
                                    second_order_corrections,
                                    lambda euler_x_prev, t_prev, x_hat, t_hat, dx_dt, rng_key: euler_x_prev,
                                    euler_x_prev, t_prev, x_hat, t_hat, d_cur, dropout_key_2)
            return heun_x_prev

        self.p_sample_jit = jax.pmap(p_sample_jit)
        self.update_fn = jax.pmap(partial(jax.lax.scan, update), axis_name=self.pmap_axis)
    
    
    def init_model_state(self, config: DictConfig):
        class TrainState(jax_utils.TrainState):
            target_model: Any = None
        
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        input_format = jnp.ones([1, *config.dataset.data_size])
        
        model_params = self.model.init(rng_dict, x=input_format, sigma=jnp.ones([1,]), train=False, augment_labels=None)['params']
        
        _, aux = self.model.apply(
                {'params': model_params}, x=input_format, sigma=jnp.ones([1,]), 
                train=False, augment_labels=None, rngs={'dropout': self.rand_key})
        
        F_x, t_emb, last_x_emb = aux
        
        head_params = self.head.init(rng_dict, x=input_format, sigma=jnp.ones([1,]), F_x=F_x, last_x_emb=last_x_emb, t_emb=t_emb,
                                     train=False, augment_labels=None)['params']

        tx = jax_utils.create_optimizer(config, "diffusion")
        new_model_state = TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            params_ema=model_params,
            target_model=model_params, # NEW!
            tx=tx
        )
        new_head_state = TrainState.create(
            apply_fn=self.head.apply,
            params=head_params,
            params_ema=head_params,
            tx=tx
        )
        return new_model_state, new_head_state


    def get_model_state(self):
        return [flax.jax_utils.unreplicate(self.head_state)]
    
    
    def fit(self, x0, cond=None, step=0):
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)
        new_carry, loss_dict_stack = self.update_fn((dropout_key, self.head_state), x0)
        (_, new_head_state) = new_carry

        loss_dict = flax.jax_utils.unreplicate(loss_dict_stack)
        for loss_key in loss_dict:
            loss_dict[loss_key] = jnp.mean(loss_dict[loss_key])
        
        self.head_state = new_head_state

        return_dict = {}
        return_dict.update(loss_dict)
        self.wandblog.update_log(return_dict)
        return return_dict
    
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None):
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        step_indices = jnp.arange(self.n_timestep)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = jnp.append(t_steps, jnp.zeros_like(t_steps[0]))
        pbar = tqdm(zip(t_steps[:-1], t_steps[1:]))

        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * t_steps[0]
        for t_cur, t_next in pbar:
            rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
            # gamma_val = jnp.minimum(jnp.sqrt(2) - 1, self.S_churn / self.n_timestep)
            # gamma = jnp.where(self.S_min <= t_cur and t_cur <= self.S_max,
            #                 gamma_val, 0)
            gamma = 0

            rng_key = jax.random.split(rng_key, jax.local_device_count())
            t_cur = jnp.asarray([t_cur] * jax.local_device_count())
            t_next = jnp.asarray([t_next] * jax.local_device_count())
            gamma = jnp.asarray([gamma] * jax.local_device_count())
            latent_sample = self.p_sample_jit(self.head_state.params_ema, latent_sample, rng_key, gamma, t_cur, t_next)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample