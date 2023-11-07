import jax
import jax.numpy as jnp

import flax
from flax.training import checkpoints

from model.unetpp import iCMPrecond
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema.ema_cm import CMEMA
from framework.default_diffusion import DefaultModel
# import lpips
import lpips_jax

from tqdm import tqdm

from omegaconf import DictConfig
from functools import partial

import os
from typing import Any

class iCMFramework(DefaultModel):
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

        self.model = iCMPrecond(model_config, 
                               image_channels=model_config['image_channels'], 
                               model_type=model_type, 
                               sigma_min=diffusion_framework['sigma_min'],
                               sigma_max=diffusion_framework['sigma_max'])
        
        self.model_state = self.init_model_state(config)
        model_prefix = "diffusion"
        self.model_state = fs_obj.load_model_state(model_prefix, self.model_state)
        
        # Replicate states for training with pmap
        self.training_states = {"model_state": self.model_state}
        
        self.training_states = flax.jax_utils.replicate(self.training_states)

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

        # Add t_steps function to deal with EDM steps for CD.
        self.t_steps_inv_fn = lambda sigma: (sigma ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)) / (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)) * (self.n_timestep - 1)
        self.t_steps_fn = lambda idx: (self.sigma_min ** (1 / self.rho) + idx / (self.n_timestep - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        # Add CT training steps for torso training
        k_prime = jnp.floor(diffusion_framework['train']['total_step'] / (jnp.log2(jnp.floor(self.s_1 / self.s_0)) + 1))
        self.ct_maximum_step_fn = lambda cur_step: jnp.minimum(self.s_0 * jnp.power(2, jnp.floor(cur_step / k_prime)), self.s_1) + 1
        self.ct_t_steps_fn = lambda idx, N_k: (self.sigma_min ** (1 / self.rho) + idx / (N_k - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        # Create ema obj
        ema_config = config.ema
        self.ema_obj = CMEMA(**ema_config)

        # Define loss functions
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16', replicate=False)

        def edm_sigma_sampling_fn(rng_key, y):
            p_mean = -1.2
            p_std = 1.2
            sigma_data = 0.5
            rnd_normal = jax.random.normal(rng_key, (y.shape[0], 1, 1, 1))
            sigma = jnp.exp(rnd_normal * p_std + p_mean)
            sigma_idx = self.t_steps_inv_fn(sigma)
            prev_sigma = self.t_steps_fn(jnp.where((sigma_idx - 1) > 0, sigma_idx - 1, 0))
            return sigma, prev_sigma

        def cd_sigma_sampling_fn(rng_key, y):
            idx = jax.random.randint(rng_key, (y.shape[0], ), minval=1, maxval=self.n_timestep)
            sigma = self.t_steps[idx][:, None, None, None]
            prev_sigma = self.t_steps[idx-1][:, None, None, None]
            return sigma, prev_sigma
        
        def ct_sigma_sampling_fn(rng_key, y, step):
            N_k = self.ct_steps_fn(step)
            idx = jax.random.randint(rng_key, (y.shape[0], ), minval=1, maxval=N_k)
            sigma = self.ct_t_steps_fn(idx, N_k)[:, None, None, None]
            prev_sigma = sigma = self.ct_t_steps_fn(idx - 1, N_k)[:, None, None, None]
            return sigma, prev_sigma

        def ict_sigma_sampling_fn(rng_key, y, step):
            p_mean = -1.1
            p_std = 2.0
            N_k = self.ct_maximum_step_fn(cur_step=step)

            # First, prepare range list from 0 to self.s_1 (include)
            overall_idx = jnp.arange(self.s_1 + 1)

            # Then, if the value is larger than the maximum step value, set it to the maximum step value.
            overall_idx = jnp.where(overall_idx < N_k, overall_idx, N_k)

            # Calculate erf of standardizated sigma for sampling from categorical distribution 
            # This process is imitation of discrete lognormal distribution. (please refer to the paper)
            overall_standardized_sigma = (jnp.log(self.ct_t_steps_fn(overall_idx, N_k)) - p_mean) / (jnp.sqrt(2) * p_std)
            overall_erf_sigma = jax.scipy.special.erf(overall_standardized_sigma)

            # Calculate categorical distribution probability
            # erf(sigma_{i+1}) - erf(sigma_{i})
            # Note that the index after the maximum step value should be zero 
            # because the corresponding index value is clipped by N_k-1, which will have same sigma value
            # Therefore, the probability value will be 0 when subtract the same erf value
            categorical_prob = jnp.zeros((self.s_1 + 1,))
            categorical_prob = categorical_prob.at[:self.s_1].set(overall_erf_sigma[1:] - overall_erf_sigma[:-1])
            categorical_prob = categorical_prob.at[self.s_1].set(0)
            categorical_prob = categorical_prob / jnp.sum(categorical_prob) # Normalize to make sure the sum is 1.
            idx = jax.random.choice(rng_key, overall_idx, shape=(y.shape[0], ), p=categorical_prob)
            next_idx = idx + 1
            sigma = self.ct_t_steps_fn(next_idx, N_k)[:, None, None, None]
            prev_sigma = self.ct_t_steps_fn(idx, N_k)[:, None, None, None]
            return sigma, prev_sigma

        def get_sigma_sampling(sigma_sampling, rng_key, y, step=None):
            if sigma_sampling == "EDM":
                return edm_sigma_sampling_fn(rng_key, y)
            elif sigma_sampling == "CD":
                return cd_sigma_sampling_fn(rng_key, y)
            elif sigma_sampling == "CT":
                return ct_sigma_sampling_fn(rng_key, y, step)
            elif sigma_sampling == "iCT":
                return ict_sigma_sampling_fn(rng_key, y, step)
            else:
                NotImplementedError("sigma_sampling should be either EDM or CM for now.")

        def lpips_loss(original, denoised, weight=1.0):
            output_shape = (original.shape[0], 224, 224, original.shape[-1])
            original = jax.image.resize(original, output_shape, "bilinear")
            denoised = jax.image.resize(denoised, output_shape, "bilinear")
            original = (original + 1) / 2.0
            denoised = (denoised + 1) / 2.0
            return jnp.mean(weight * self.perceptual_loss(original, denoised))
    
        def huber_loss(original, denoised, weight=1.0):
            data_dim = original.shape[-1] * original.shape[-2] * original.shape[-3] # H * W * C
            c = 0.00054 * jnp.sqrt(data_dim)
            huber = jnp.sqrt(jnp.mean((original - denoised) ** 2, axis=(-1, -2, -3)) + c ** 2) - c
            huber = weight * huber
            return jnp.mean(huber)

        @jax.jit
        def monitor_metric_fn(params, y, rng_key):
            torso_params = params.get('torso_state', self.torso_state.params_ema)
            head_params = params['head_state']

            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)

            idx = jax.random.randint(step_key, (y.shape[0], ), minval=1, maxval=self.n_timestep)
            sigma = self.t_steps[idx][:, None, None, None]
            prev_sigma = self.t_steps[idx-1][:, None, None, None]
            
            noise = jax.random.normal(noise_key, y.shape)
            perturbed_x = y + sigma * noise

            # denoised, D_x, aux = model_default_output_fn(params, y, sigma, prev_sigma, noise, rng_key, eval_mode=True)
            # Get consistency function values
            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            D_x, aux = self.model.apply(
                {'params': torso_params}, x=perturbed_x, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key_2})
            
            # Monitor distillation loss (l2 loss)
            l2_dist = jnp.mean((D_x - prev_D_x) ** 2)
            
            # Monitor distillation loss (perceptual loss)
            output_shape = (y.shape[0], 224, 224, y.shape[-1])
            D_x = jax.image.resize(D_x, output_shape, "bilinear")
            prev_D_x = jax.image.resize(prev_D_x, output_shape, "bilinear")
            D_x = (D_x + 1) / 2.0
            prev_D_x = (prev_D_x + 1) / 2.0
            lpips_dist = jnp.mean(self.perceptual_loss(D_x, prev_D_x))

            # Monitor difference between original CM and trained (perturbed) CM
            original_D_x, aux = self.model.apply(
                {'params': self.torso_state.params_ema}, x=y, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            original_D_x = jax.image.resize(original_D_x, output_shape, "bilinear")
            lpips_dist_btw_training_and_original_cm = jnp.mean(self.perceptual_loss(D_x, original_D_x))

            loss_dict = {}
            loss_dict['eval/l2_dist'] = l2_dist
            loss_dict['eval/lpips_dist_for_training_cm'] = lpips_dist
            loss_dict['eval/lpips_dist_btw_training_and_original_cm'] = lpips_dist_btw_training_and_original_cm
            return loss_dict

        @jax.jit
        def model_loss_fn(update_params, total_states_dict, args, has_aux=False):
            model_params = update_params['model_state']
            target_model = jax.lax.stop_gradient(model_params)

            # Unzip arguments for loss_fn
            y, rng_key = args

            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)
            noise = jax.random.normal(noise_key, y.shape)
            sigma, prev_sigma = get_sigma_sampling(diffusion_framework['sigma_sampling'], step_key, y, total_states_dict['model_state'].step)

            # denoised, D_x, aux = model_default_output_fn(params, y, sigma, prev_sigma, noise, rng_key)
            perturbed_x = y + sigma * noise

            # Get consistency function values
            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            D_x, _ = self.model.apply(
                {'params': model_params}, x=perturbed_x, sigma=sigma,
                train=True, augment_labels=None, rngs={'dropout': dropout_key_2})
            
            prev_perturbed_x = y + prev_sigma * noise
            prev_D_x, _ = self.model.apply(
                {'params': target_model}, x=prev_perturbed_x, sigma=prev_sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})

            # Loss and loss dict construction
            total_loss = 0
            loss_dict = {}
            if diffusion_framework['loss'] == "lpips":
                loss = lpips_loss(D_x, prev_D_x)
                loss_dict['train/lpips_loss'] = loss
            elif diffusion_framework['loss'] == "huber":
                weight = 1 / (sigma - prev_sigma)
                loss = huber_loss(D_x, prev_D_x, weight=weight)
                loss_dict['train/huber_loss'] = loss
            total_loss += loss
            loss_dict['train/total_loss'] = total_loss
            loss_dict['train/N_k'] = jnp.mean(self.ct_maximum_step_fn(total_states_dict['model_state'].step))
            return total_loss, loss_dict


        def update_params_fn(
                states_dict: dict, 
                update_states_key_list: list, 
                loss_fn, 
                loss_fn_args, 
                loss_dict_tail: dict = {}):
            update_params_dict = {params_key: states_dict[params_key].params for params_key in update_states_key_list}
            (_, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(update_params_dict, states_dict, loss_fn_args)
            grads = jax.lax.pmean(grads, axis_name=self.pmap_axis)
            updated_states = {params_key: states_dict[params_key].apply_gradients(grads=grads[params_key]) 
                              for params_key in update_states_key_list}
            loss_dict_tail.update(loss_dict)
            states_dict.update(updated_states)
            return states_dict, loss_dict_tail
            

        # Define update function
        def update(carry_state, x0):
            (rng, states) = carry_state
            rng, new_rng = jax.random.split(rng)

            loss_dict = {}
            model_key = ['model_state']
            rng, torso_rng = jax.random.split(rng)
            states, loss_dict = update_params_fn(states, model_key, model_loss_fn, (x0, torso_rng), loss_dict)

            # Target model EMA (for consistency model training procedure)
            model_state = states['model_state']
            states['model_state'] = model_state

            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)
            
            # Update EMA for sampling
            states = {state_name: self.ema_obj.ema_update(state_content) for state_name, state_content in states.items()}
            new_carry_state = (new_rng, states)
            return new_carry_state, loss_dict

        def sample_cm_fn(params, x_cur, rng_key, gamma=None, t_cur=None, t_prev=None):
            dropout_key = rng_key
            denoised, aux = self.model.apply(
                {'params': params}, x=x_cur, sigma=t_cur, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            return denoised

        self.p_sample_cm = jax.pmap(sample_cm_fn)
        self.update_fn = jax.pmap(partial(jax.lax.scan, update), axis_name=self.pmap_axis)
        self.eval_fn = jax.pmap(monitor_metric_fn, axis_name=self.pmap_axis)
    
    def get_training_states_params(self):
        return {state_name: state_content.params for state_name, state_content in self.training_states.items()}
    
    def init_model_state(self, config: DictConfig):
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        input_format = jnp.ones([1, *config.dataset.data_size])
        
        torso_params = self.model.init(
            rng_dict, x=input_format, sigma=jnp.ones([1,]), train=False, augment_labels=None)['params']
        
        model_tx = jax_utils.create_optimizer(config, "diffusion")
        new_model_state = jax_utils.TrainState.create(
            apply_fn=self.model.apply,
            params=torso_params,
            params_ema=torso_params,
            tx=model_tx
        )
        return new_model_state

    def get_model_state(self):
        return {
            "diffusion": flax.jax_utils.unreplicate(self.training_states['model_state'])
        }
    
    def fit(self, x0, cond=None, step=0, eval_during_training=False):
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)
        # new_carry, loss_dict_stack = self.update_fn((dropout_key, self.head_state), x0)
        new_carry, loss_dict_stack = self.update_fn((dropout_key, self.training_states), x0)
        (_, training_states) = new_carry

        loss_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), loss_dict_stack)
        self.training_states = training_states

        # eval_dict = {}
        # if eval_during_training:
        #     eval_key, self.rand_key = jax.random.split(self.rand_key, 2)
        #     eval_key = jnp.asarray(jax.random.split(eval_key, jax.local_device_count()))
        #     # TODO: self.eval_fn is called using small batch size. This is not good for evaluation.
        #     # Need to use large batch size (e.g. using scan function.)
        #     eval_dict = self.eval_fn(self.get_training_states_params(), x0[:, 0], eval_key)
        #     eval_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), eval_dict)

        return_dict = {}
        return_dict.update(loss_dict)
        # return_dict.update(eval_dict)
        self.wandblog.update_log(return_dict)
        return return_dict

    def sampling_cm(self, num_image, img_size=(32, 32, 3), original_data=None, mode="cm_training"):
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        # One-step generation
        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * self.sigma_max
        
        rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
        rng_key = jax.random.split(rng_key, jax.local_device_count())
        gamma = jnp.asarray([0] * jax.local_device_count())
        t_max = jnp.asarray([self.sigma_max] * jax.local_device_count())
        t_min = jnp.asarray([self.sigma_min] * jax.local_device_count())

        sampling_params = self.training_states['model_state'].params_ema

        latent_sample = self.p_sample_cm(sampling_params, latent_sample, rng_key, gamma, t_max, t_min)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None, mode="edm"):
        return self.sampling_cm(num_image, img_size, original_data, mode)

    
    