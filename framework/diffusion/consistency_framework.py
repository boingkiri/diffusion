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
        
        self.model = CMPrecond(model_config, 
                               image_channels=model_config['image_channels'], 
                               model_type=model_type, 
                               sigma_min=diffusion_framework['sigma_min'],
                               sigma_max=diffusion_framework['sigma_max'])
        self.head = ScoreDistillPrecond(model_config, 
                               image_channels=model_config['image_channels'], 
                               sigma_min=diffusion_framework['sigma_min'],
                               sigma_max=diffusion_framework['sigma_max'])
        
        self.model_state, self.head_state = self.init_model_state(config)
        self.model_state = fs_obj.load_model_state("diffusion", self.model_state)
        self.head_state = fs_obj.load_model_state("diffusion", self.head_state)
        self.model_state = flax.jax_utils.replicate(self.model_state)
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
        self.one_step_sampling = diffusion_framework.one_step_sampling
        self.target_model_ema_decay = diffusion_framework.target_model_ema_decay
        self.dsm_ratio = diffusion_framework.dsm_ratio
        self.distill_step = diffusion_framework.distill_step

        # Set step indices for distillation
        step_indices = jnp.arange(self.n_timestep)
        self.t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho


        # Create ema obj
        ema_config = config.ema
        self.ema_obj = CMEMA(**ema_config)


        # Define loss functions
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16', replicate=False)
        
        @jax.jit
        def joint_loss_fn(params, target_params, head_params, y, rng_key):
            # head_params not updated during this phase
            head_params = jax.lax.stop_gradient(head_params)
            
            rng_key, step_key, solver_key, dropout_key = jax.random.split(rng_key, 4)

            # Sample n ~ U[0, N-2]
            idx = jax.random.randint(step_key, (y.shape[0], ), minval=0, maxval=self.n_timestep-1)
            sigma = self.t_steps[idx][:, None, None, None]
            next_sigma = self.t_steps[idx+1][:, None, None, None]

            noise = jax.random.normal(rng_key, y.shape)
            perturbed_x = y + next_sigma * noise

            # Get consistency function values
            online_consistency, aux = self.model.apply(
                {'params': params}, x = perturbed_x, sigma=next_sigma, 
                train=True, augment_labels=None, rngs={'dropout': dropout_key})
            
            F_x, t_emb, last_x_emb = aux
            
            # Apply Heun's 2nd order method to obtain the previous sample point
            delta_t = next_sigma - sigma
            
            distill_output = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=next_sigma, F_x=F_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            target_euler = perturbed_x + delta_t * distill_output
            
            _, aux = self.model.apply(
                {'params': params}, x = target_euler, sigma=sigma, 
                train=True, augment_labels=None, rngs={'dropout': dropout_key})
            
            prev_F_x, prev_t_emb, prev_last_x_emb = aux
            
            distill_output_prime = self.head.apply(
                {'params': head_params}, x=target_euler, sigma=sigma, F_x=prev_F_x, t_emb=prev_t_emb, last_x_emb=prev_last_x_emb,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})

            target_heun = perturbed_x + 0.5 * delta_t * (distill_output + distill_output_prime)
            
            target_consistency, _ = self.model.apply(
                {'params': target_params}, x = target_heun, sigma=sigma, 
                train=True, augment_labels=None, rngs={'dropout': dropout_key})

            assert diffusion_framework.loss == "lpips"
            
            output_shape = (y.shape[0], 224, 224, y.shape[-1])
            online_consistency = jax.image.resize(online_consistency, output_shape, "bilinear")
            target_consistency = jax.image.resize(target_consistency, output_shape, "bilinear")
            online_consistency = (online_consistency + 1) / 2.0
            target_consistency = (target_consistency + 1) / 2.0
            
            target_consistency = jax.lax.stop_gradient(target_consistency)
            
            perceptual_loss = jnp.mean(self.perceptual_loss(online_consistency, target_consistency))
            dsm_loss = jnp.mean((distill_output + noise) ** 2)
            joint_loss = perceptual_loss + self.dsm_ratio * dsm_loss

            loss_dict = {}
            loss_dict['perceptual_loss'] = perceptual_loss
            loss_dict['dsm_loss'] = dsm_loss
            loss_dict['joint_loss'] = joint_loss
            return joint_loss, loss_dict
        
        
        @jax.jit
        def distill_loss_fn(head_params, params, y, rng_key):
            # params not updated during this phase
            params = jax.lax.stop_gradient(params)
            
            rng_key, step_key, solver_key, dropout_key = jax.random.split(rng_key, 4)

            # Sample n ~ U[0, N-1]
            idx = jax.random.randint(step_key, (y.shape[0], ), minval=0, maxval=self.n_timestep)
            sigma = self.t_steps[idx][:, None, None, None]

            noise = jax.random.normal(rng_key, y.shape)
            perturbed_x = y + sigma * noise

            # Get consistency function values
            D_x, aux = self.model.apply(
                {'params': params}, x=perturbed_x, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            F_x, t_emb, last_x_emb = aux
            
            distill_output = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, F_x=F_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=True, augment_labels=None, rngs={'dropout': dropout_key})
            
            model_fn = lambda x, t: self.model.apply(
                {'params': params}, x=x, sigma=t, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            _, distill_loss, _ = jax.jvp(model_fn, (perturbed_x, sigma), (distill_output, -jnp.ones_like(sigma)), has_aux=True)
            distill_loss = jnp.mean(distill_loss ** 2)

            loss_dict = {}
            loss_dict['distill_loss'] = distill_loss
            return distill_loss, loss_dict
        
        
        # Define update function
        def update(carry_state, x0):
            (rng, state, head_state) = carry_state
            rng, new_rng = jax.random.split(rng)
            
            # Update CM
            (_, joint_loss_dict), cm_grad = jax.value_and_grad(joint_loss_fn, has_aux=True)(state.params, state.target_model, head_state.params, x0, rng)
            cm_grad = jax.lax.pmean(cm_grad, axis_name=self.pmap_axis)
            new_state = state.apply_gradients(grads=cm_grad)
            
            # Update head (for multiple times)
            (_, distill_loss_dict), head_grad = jax.value_and_grad(distill_loss_fn, has_aux=True)(head_state.params, new_state.params, x0, rng)
            head_grad = jax.lax.pmean(head_grad, axis_name=self.pmap_axis)
            new_head_state = head_state.apply_gradients(grads=head_grad)
            for _ in range(self.distill_step-1):
                (_, distill_loss_dict), head_grad = jax.value_and_grad(distill_loss_fn, has_aux=True)(new_head_state.params, new_state.params, x0, rng)
                head_grad = jax.lax.pmean(head_grad, axis_name=self.pmap_axis)
                new_head_state = new_head_state.apply_gradients(grads=head_grad)

            # Merge two loss dictionaries 
            for loss_key in joint_loss_dict:
                joint_loss_dict[loss_key] = jax.lax.pmean(joint_loss_dict[loss_key], axis_name=self.pmap_axis)
            
            for loss_key in distill_loss_dict:
                distill_loss_dict[loss_key] = jax.lax.pmean(distill_loss_dict[loss_key], axis_name=self.pmap_axis)

            loss_dict = {**joint_loss_dict, **distill_loss_dict}

            # Update EMA for target model
            ema_updated_params = jax.tree_map(
                lambda x, y: self.target_model_ema_decay * x + (1 - self.target_model_ema_decay) * y,
                new_state.target_model, new_state.params)
            new_state = new_state.replace(target_model=ema_updated_params)

            # Update EMA for sampling
            new_state = self.ema_obj.ema_update(new_state)
            new_head_state = self.ema_obj.ema_update(new_head_state)

            new_carry_state = (new_rng, new_state, new_head_state)
            return new_carry_state, loss_dict


        # Define p_sample_jit functions for sampling
        # One step sampling using CM
        if self.one_step_sampling:
            def p_sample_jit(params, x_cur, rng_key, step):
                dropout_key = rng_key
                
                consistency_output, _ = self.model.apply(
                    {'params': params}, x=x_cur, sigma=step, 
                    train=False, augment_labels=None, rngs={'dropout': dropout_key})
                
                return consistency_output
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        else:
            def p_sample_jit(params, head_params, x_cur, rng_key, gamma, t_cur, t_prev):
                rng_key, dropout_key, dropout_key_2 = jax.random.split(rng_key, 3)

                # Increase noise temporarily.
                t_hat = t_cur + gamma * t_cur
                noise = jax.random.normal(rng_key, x_cur.shape) * self.S_noise
                x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * noise

                # Euler step
                _, aux = self.model.apply(
                    {'params': params}, x=x_hat, sigma=t_hat, 
                    train=False, augment_labels=None, rngs={'dropout': dropout_key})
                
                F_x, t_emb, last_x_emb = aux
                
                t_score = self.head.apply(
                    {'params': head_params}, x=x_hat, sigma=t_hat, F_x=F_x, t_emb=t_emb, last_x_emb=last_x_emb,
                    train=False, augment_labels=None, rngs={'dropout': dropout_key}
                )
                
                euler_x_prev = x_hat + (t_hat - t_prev) * t_score

                # Apply 2nd order correction.
                def second_order_corrections(euler_x_prev, t_prev, x_hat, t_hat, t_score, rng_key):
                    _, aux = self.model.apply(
                        {'params': params}, x=euler_x_prev, sigma=t_prev,
                        train=False, augment_labels=None, rngs={'dropout': rng_key})
                    
                    F_x, t_emb, last_x_emb = aux
                    
                    t_score_prime = self.head.apply(
                        {'params': head_params}, x=euler_x_prev, sigma=t_prev, F_x=F_x, t_emb=t_emb, last_x_emb=last_x_emb,
                        train=False, augment_labels=None, rngs={'dropout': dropout_key}
                    )
                    
                    heun_x_prev = x_hat + 0.5 * (t_hat - t_prev) * (t_score + t_score_prime)
                    return heun_x_prev
                
                heun_x_prev = jax.lax.cond(t_prev != 0.0,
                                        second_order_corrections,
                                        lambda euler_x_prev, t_prev, x_hat, t_hat, t_score, rng_key: euler_x_prev,
                                        euler_x_prev, t_prev, x_hat, t_hat, t_score, dropout_key_2)
                return heun_x_prev

        self.p_sample_jit = jax.pmap(p_sample_jit)
        self.update_fn = jax.pmap(partial(jax.lax.scan, update), axis_name=self.pmap_axis)
    
    
    def init_model_state(self, config: DictConfig):
        class TrainState(jax_utils.TrainState):
            target_model: Any = None
        
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        input_format = jnp.ones([1, *config.dataset.data_size])
        t_emb_format = jnp.ones([512,])
        last_x_emb_format = jnp.ones([1, 32, 32, 256])
        

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
        return [flax.jax_utils.unreplicate(self.model_state)]
    
    
    def fit(self, x0, cond=None, step=0):
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)
        new_carry, loss_dict_stack = self.update_fn((dropout_key, self.model_state, self.head_state), x0)
        (_, new_state, new_head_state) = new_carry

        loss_dict = flax.jax_utils.unreplicate(loss_dict_stack)
        for loss_key in loss_dict:
            loss_dict[loss_key] = jnp.mean(loss_dict[loss_key])

        self.model_state = new_state
        self.head_state = new_head_state

        return_dict = {}
        return_dict.update(loss_dict)
        self.wandblog.update_log(return_dict)
        return return_dict
    
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None):
        # One step sampling using CM
        if self.one_step_sampling:
            latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
            sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

            # One-step generation
            latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * self.sigma_max
            
            rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
            rng_key = jax.random.split(rng_key, jax.local_device_count())
            sigma_max =  jnp.asarray([self.sigma_max] * jax.local_device_count())
            latent_sample = self.p_sample_jit(self.model_state.params_ema, latent_sample, rng_key, sigma_max)

            if original_data is not None:
                rec_loss = jnp.mean((latent_sample - original_data) ** 2)
                self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
            return latent_sample
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        else:
            latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
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

                rng_key = jax.random.split(rng_key, jax.local_device_count())
                t_cur = jnp.asarray([t_cur] * jax.local_device_count())
                t_next = jnp.asarray([t_next] * jax.local_device_count())
                gamma = jnp.asarray([gamma] * jax.local_device_count())
                latent_sample = self.p_sample_jit(self.model_state.params_ema, self.head_state.params_ema, latent_sample, rng_key, gamma, t_cur, t_next)

            if original_data is not None:
                rec_loss = jnp.mean((latent_sample - original_data) ** 2)
                self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
            return latent_sample