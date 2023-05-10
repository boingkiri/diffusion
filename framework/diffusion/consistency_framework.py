import jax
import jax.numpy as jnp

import flax
from flax.training import checkpoints

from model.unetpp import CMPrecond, EDMPrecond
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema.ema_cm import CMEMA
from utils.ema.ema_edm import EDMEMA
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
        # self.model = EDMPrecond(model_config, image_channels=model_config['image_channels'], model_type=model_type)
        self.model = CMPrecond(model_config, 
                               image_channels=model_config['image_channels'], 
                               model_type=model_type, 
                               sigma_min=diffusion_framework['sigma_min'],
                               sigma_max=diffusion_framework['sigma_max'])
        self.model_state = self.init_model_state(config)
        self.model_state = fs_obj.load_model_state("diffusion", self.model_state)

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

        # Distillation or Training
        self.is_distillation = diffusion_framework.is_distillation
        if self.is_distillation:
            teacher_model_path = diffusion_framework.distillation_path
            prefix = fs_obj.get_state_prefix("diffusion")
            if prefix not in teacher_model_path: # It means teacher_model_path is indicating exp name 
                checkpoint_dir = config.exp.checkpoint_dir.split("/")[-1]
                teacher_model_path = os.path.join(teacher_model_path, checkpoint_dir)
            ## TODO: need to implement various pretrained model
            ## For now, I just write code to execute our EDM pretrained model
            self.teacher_model = EDMPrecond(model_config, image_channels=model_config['image_channels'], model_type=model_type)
            self.teacher_model_state = checkpoints.restore_checkpoint(teacher_model_path, None, prefix=prefix)

            # Set step indices for distillation
            step_indices = jnp.arange(self.n_timestep)
            # self.t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
            self.t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
            
            # Initialize model 
            if self.model_state.step == 0:
                frozened_params = flax.core.frozen_dict.freeze(self.teacher_model_state["params_ema"])
                self.model_state = self.model_state.replace(params=frozened_params)
                self.model_state = self.model_state.replace(params_ema=frozened_params)
                self.model_state = self.model_state.replace(target_model=frozened_params)
                self.target_model_ema_decay = diffusion_framework.target_model_ema_decay
        else:
            self.n_timestep_fn = lambda k: jnp.ceil(jnp.sqrt((k / diffusion_framework.train.total_step) * ((self.s_1 + 1) ** 2 - self.s_0 ** 2) + self.s_0 ** 2) - 1) + 1
            # input parameter changed from "k" to "n_timestep"
            self.ema_power_fn = lambda n_timestep: jnp.exp(self.s_0 * jnp.log(self.mu_0) / n_timestep)
            self.t_steps_fn = lambda idx, n_timestep: (self.sigma_min ** (1 / self.rho) + idx / (n_timestep - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
            '''
            # CT training is also initialized by pre-trained EDM model for fair comparison with continuous-time CT
            self.teacher_model = EDMPrecond(model_config, image_channels=model_config['image_channels'], model_type=model_type)
            self.teacher_model_state = checkpoints.restore_checkpoint(teacher_model_path, None, prefix=prefix)
            
            # Initialize model 
            if self.model_state.step == 0:
                frozened_params = flax.core.frozen_dict.freeze(self.teacher_model_state["params_ema"])
                self.model_state = self.model_state.replace(params=frozened_params)
                self.model_state = self.model_state.replace(params_ema=frozened_params)
                self.model_state = self.model_state.replace(target_model=frozened_params)
                self.target_model_ema_decay = diffusion_framework.target_model_ema_decay
            '''
        
        # Replicate model state to use multiple computation units 
        self.model_state = flax.jax_utils.replicate(self.model_state)

        # Create ema obj
        ema_config = config.ema
        self.ema_obj = CMEMA(**ema_config)
        # self.ema_obj = EDMEMA(**ema_config)

        # Augmentation pipeline
        augment_rng, self.rand_key = jax.random.split(self.rand_key)
        self.augment_rate = diffusion_framework.get("augment_rate", None)
        if self.augment_rate is not None:
            self.augmentation_pipeline = AugmentPipe(
                rng_key=augment_rng, p=self.augment_rate, xflip=1e8, 
                yflip=1, scale=1, rotate_frac=1, 
                aniso=1, translate_frac=1)

        # Distillation and Training loss function are different.
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16', replicate=False)
        if self.is_distillation:
            @jax.jit
            def loss_fn(params, target_model, y, rng_key):
                rng_key, step_key, solver_key, dropout_key = jax.random.split(rng_key, 4)

                # Sample n ~ U[0, N-2]
                idx = jax.random.randint(step_key, (y.shape[0], ), minval=0, maxval=self.n_timestep-1)
                sigma = self.t_steps[idx][:, None, None, None]
                next_sigma = self.t_steps[idx+1][:, None, None, None]

                gamma = self.get_gamma(idx)[:, None, None, None]
                noise = jax.random.normal(rng_key, y.shape)
                perturbed_x = y + next_sigma * noise

                # Calculate heun 2nd method
                target_xn = heun_2nd_method(self.teacher_model_state['params_ema'], perturbed_x, solver_key, gamma, idx+1)

                augment_dim = config.model.diffusion.get("augment_dim", None)
                augment_labels = jnp.zeros((*perturbed_x.shape[:-3], augment_dim)) if augment_dim is not None else None

                # Get consistency function values
                online_consistency = self.model.apply(
                    {'params': params}, x=perturbed_x,
                    sigma=next_sigma, train=True, augment_labels=augment_labels, rngs={'dropout': dropout_key})
                
                target_consistency = self.model.apply(
                    {'params': target_model}, x=target_xn, 
                    sigma=sigma, train=True, augment_labels=augment_labels, rngs={'dropout': dropout_key})

                if diffusion_framework.loss == "lpips":
                    output_shape = (y.shape[0], 224, 224, y.shape[-1])
                    online_consistency = jax.image.resize(online_consistency, output_shape, "bilinear")
                    target_consistency = jax.image.resize(target_consistency, output_shape, "bilinear")
                    online_consistency = (online_consistency + 1) / 2.0
                    target_consistency = (target_consistency + 1) / 2.0
                    loss = jnp.mean(self.perceptual_loss(online_consistency, target_consistency))
                elif diffusion_framework.loss == "l2":
                    loss = jnp.mean((online_consistency - target_consistency) ** 2)
                elif diffusion_framework.loss == "l1":
                    loss = jnp.mean(jnp.abs(online_consistency - target_consistency))
                else:
                    NotImplementedError("Consistency model is only support lpips, l2, and l1 loss.")

                loss_dict = {}
                loss_dict['total_loss'] = loss
                return loss, loss_dict
        else:
            @jax.jit
            def loss_fn(params, params_ema, y, rng_key, n_timestep):
                rng_key, step_key, normal_key, dropout_key = jax.random.split(rng_key, 4)
                
                # Sample n ~ U[0, N-2]
                idx = jax.random.randint(step_key, (y.shape[0], ), minval=0, maxval=n_timestep-1)
                
                sigma = self.t_steps_fn(idx, n_timestep)[:, None, None, None]
                next_sigma = self.t_steps_fn(idx+1, n_timestep)[:, None, None, None]
                
                noise = jax.random.normal(rng_key, y.shape)
                
                augment_dim = config.model.diffusion.get("augment_dim", None)
                augment_labels = jnp.zeros((*y.shape[:-3], augment_dim)) if augment_dim is not None else None

                # Get consistency function values
                online_consistency = self.model.apply(
                    {'params': params}, x= y + next_sigma * noise,
                    sigma=next_sigma, train=True, augment_labels=augment_labels, rngs={'dropout': dropout_key})
                
                target_consistency = self.model.apply(
                    {'params': params_ema}, x= y + sigma * noise, 
                    sigma=sigma, train=True, augment_labels=augment_labels, rngs={'dropout': dropout_key})

                if diffusion_framework.loss == "lpips":
                    output_shape = (y.shape[0], 224, 224, y.shape[-1])
                    online_consistency = jax.image.resize(online_consistency, output_shape, "bilinear")
                    target_consistency = jax.image.resize(target_consistency, output_shape, "bilinear")
                    online_consistency = (online_consistency + 1) / 2.0
                    target_consistency = (target_consistency + 1) / 2.0
                    loss = jnp.mean(self.perceptual_loss(online_consistency, target_consistency))
                elif diffusion_framework.loss == "l2":
                    loss = jnp.mean((online_consistency - target_consistency) ** 2)
                elif diffusion_framework.loss == "l1":
                    loss = jnp.mean(jnp.abs(online_consistency - target_consistency))
                else:
                    NotImplementedError("Consistency model is only support lpips, l2, and l1 loss.")

                loss_dict = {}
                loss_dict['total_loss'] = loss
                return loss, loss_dict
        
        def update(carry_state, x0):
            (rng, state) = carry_state
            rng, new_rng = jax.random.split(rng)
            
            args = [state.params, jax.lax.stop_gradient(state.target_model), x0, rng]         
            if not self.is_distillation:
                # state.step is incremented by every call to 'apply.gradients'
                n_timestep = self.n_timestep_fn(state.step)
                self.target_model_ema_decay = self.ema_power_fn(n_timestep)
                args += [n_timestep.astype(float)]
            
            (_, loss_dict), grad = jax.value_and_grad(loss_fn, has_aux=True)(*args)

            grad = jax.lax.pmean(grad, axis_name=self.pmap_axis)
            # breakpoint()
            new_state = state.apply_gradients(grads=grad)
            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)

            # Update EMA for training (update target model)
            ema_updated_params = jax.tree_map(
                lambda x, y: self.target_model_ema_decay * x + (1 - self.target_model_ema_decay) * y,
                new_state.target_model, new_state.params)
            new_state = new_state.replace(target_model = ema_updated_params)

            # Update EMA for sampling
            new_state = self.ema_obj.ema_update(new_state)

            new_carry_state = (new_rng, new_state)
            return new_carry_state, loss_dict
        
        def heun_2nd_method(params, x_cur, rng_key, gamma, step):
            rng_key, dropout_key, dropout_key_2 = jax.random.split(rng_key, 3)

            t_cur = self.t_steps[step]
            t_next = jnp.where(
                step == 0, 
                jnp.zeros_like(t_cur), 
                self.t_steps[step - 1])
            t_cur = t_cur[:, None, None, None]
            t_next = t_next[:, None, None, None]

            # Increase noise temporarily.
            t_hat = t_cur + gamma * t_cur
            noise = jax.random.normal(rng_key, x_cur.shape) * self.S_noise
            x_hat = x_cur + noise * jnp.sqrt(t_hat ** 2 - t_cur ** 2)

            # Augment label
            augment_dim = config.model.diffusion.get("augment_dim", None)
            augment_labels = jnp.zeros((*x_cur.shape[:-3], augment_dim)) if augment_dim is not None else None

            # Euler step
            denoised = self.teacher_model.apply(
                {'params': params}, x=x_hat, sigma=t_hat, 
                train=False, augment_labels=augment_labels, rngs={'dropout': dropout_key})
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            def second_order_corrections(x_next, t_next, x_hat, t_hat, d_cur, rng_key, step):
                denoised = self.teacher_model.apply(
                    {'params': params}, x=x_next, sigma=t_next, 
                    train=False, augment_labels= augment_labels, rngs={'dropout': rng_key})
                d_prime = (x_next - denoised) / t_next
                x_corrected_one = x_hat + (0.5 * d_cur + 0.5 * d_prime) * (t_next - t_hat)
                return_val = jnp.where(step[:, None, None, None] == 0, x_next, x_corrected_one)
                return return_val
            
            x_result = second_order_corrections(x_next, t_next, x_hat, t_hat, d_cur, dropout_key_2, step)
            
            return x_result

        def p_sample_fn(params, x_cur, rng_key, step):
            dropout_key = rng_key
            # Augment label
            augment_dim = config.model.diffusion.get("augment_dim", None)
            augment_labels = jnp.zeros((*x_cur.shape[:-3], augment_dim)) if augment_dim is not None else None

            denoised = self.model.apply(
                {'params': params}, x=x_cur, sigma=step, 
                train=False, augment_labels=augment_labels, rngs={'dropout': dropout_key})
            return denoised

        self.update_fn = jax.pmap(partial(jax.lax.scan, update), axis_name=self.pmap_axis)
        self.p_sample_jit = jax.pmap(p_sample_fn, axis_name=self.pmap_axis, in_axes=(0, 0, 0, None))

    
    def p_sample(self, param, xt, t):
        # Sample from p_theta(x_{t-1}|x_t)
        self.rand_key, normal_key, dropout_key = jax.random.split(self.rand_key, 3)
        return self.p_sample_jit(param, xt, t, normal_key, dropout_key)
    
    def init_model_state(self, config: DictConfig):
        class TrainState(jax_utils.TrainState):
            target_model: Any = None
        
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        input_format = jnp.ones([1, *config.dataset.data_size])

        augment_dim = config.model.diffusion.get("augment_dim", None)
        augment_labels = jnp.zeros([1, augment_dim]) if augment_dim is not None else None
        params = self.model.init(rng_dict, x=input_format, sigma=jnp.ones([1,]), train=False, augment_labels=augment_labels)['params']

        # state: jax_utils.TrainState = jax_utils.create_train_state(config, 'diffusion', self.model.apply, params)
        tx = jax_utils.create_optimizer(config, "diffusion")
        new_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            params_ema=params,
            target_model=params, # NEW!
            tx=tx
        )
        return new_state

    def get_model_state(self):
        return [flax.jax_utils.unreplicate(self.model_state)]
    
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
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None):
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        # One-step generation
        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * self.sigma_max
        
        rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
        rng_key = jax.random.split(rng_key, jax.local_device_count())
        # latent_sample = self.p_sample_jit(self.model_state.params_ema, latent_sample, rng_key, self.sigma_max)
        latent_sample = self.p_sample_jit(self.model_state.params_ema, latent_sample, rng_key, self.sigma_max)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample

    def get_gamma(self, step):
        # gamma_val = jnp.minimum(jnp.sqrt(2) - 1, self.S_churn / self.n_timestep)
        # gamma = jnp.where(self.S_min <= self.t_steps[step] and self.t_steps[step] <= self.S_max,
        #                 gamma_val, 0)
        # return gamma
        return jnp.zeros_like(step) # This is possible because consistency model only consider deterministic sampling