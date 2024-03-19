import jax
import jax.numpy as jnp

import numpy as np

import flax
from flax.training import checkpoints

from model.unetpp import CMPrecond, ScoreDistillPrecond, EDMPrecond
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema.ema_cm import CMEMA
from utils.augment_utils import AugmentPipe
from utils.jax_utils import unreplicate_tree, create_environment_sharding
from framework.default_diffusion import DefaultModel
# import lpips
import lpips_jax

import orbax.checkpoint

from tqdm import tqdm

from omegaconf import DictConfig
from functools import partial

import os
from typing import Any

# from clu import parameter_overview

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
        self.distributed_training = config.get("distributed_training", False)

        # Create UNet and its state
        model_config = {**config.model.diffusion}
        model_type = model_config.pop("type")

        head_config = {**config.model.head}
        head_type = head_config.pop("type")
        self.head_type = head_type
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

        self.torso_state, self.head_state = self.init_model_state(config)
        
        states= {}
        if self.torso_state is not None:
            states["diffusion"] = self.torso_state
        if self.head_state is not None:
            states["head"] = self.head_state

        states = fs_obj.load_model_state(states)
        self.torso_state, self.head_state = states.get("diffusion"), states.get("head")
        # Replicate states for training with pmap
        self.training_states = {}
        self.training_states['torso_state'] = self.torso_state
        self.training_states["head_state"] = self.head_state
        
        # self.training_states = flax.jax_utils.replicate(self.training_states)
        # self.training_states = {}
        # XXX: Convert orbax.checkpoint.composite_checkpoint_handler.CompositeArgs to dict of flax.TrainState
        # replicated_devices = jax_utils.create_replicated_sharding() if config.get("distributed_training", False) else None 
        # if self.distributed_training:
        #     self.training_states = {model_key: jax.experimental.multihost_utils.broadcast_one_to_all(self.training_states[model_key])
        #                         for model_key in self.training_states.keys()}
        # else:                            
        self.training_states = {model_key: flax.jax_utils.replicate(self.training_states[model_key]) 
                            for model_key in self.training_states.keys()}
        if self.distributed_training:
            devices = np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count())
            axes_names = ('host', 'devices')
            global_mesh = jax.sharding.Mesh(devices, axes_names)
            pspecs = jax.sharding.PartitionSpec("host")
            self.training_states = {model_key: jax.experimental.multihost_utils.host_local_array_to_global_array(self.training_states[model_key], global_mesh, pspecs)
                                for model_key in self.training_states.keys()}
            # self.training_states = {model_key: jax.experimental.multihost_utils.broadcast_one_to_all(self.training_states[model_key])
            #                     for model_key in self.training_states.keys()}
            # self.training_states = jax.tree_map(lambda x: jnp.asarray(x), self.training_states)
            # self.training_states = {model_key: flax.jax_utils.replicate(self.training_states[model_key]) 
            #                 for model_key in self.training_states.keys()}
        # self.sharding = jax_utils.create_environment_sharding()
        
        # breakpoint()
        # Parameters
        self.sigma_min = diffusion_framework['sigma_min']
        self.sigma_max = diffusion_framework['sigma_max']
        self.rho = diffusion_framework['rho']
        self.mu_0 = diffusion_framework.params_ema_for_training[0]
        self.s_0 = diffusion_framework.params_ema_for_training[1]
        self.s_1 = diffusion_framework.params_ema_for_training[2]

        # Set step indices for distillation
        step_indices = jnp.arange(self.n_timestep)
        self.t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        # Add t_steps function to deal with EDM steps for CD.
        self.t_steps_inv_fn = lambda sigma: (sigma ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)) / (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)) * (self.n_timestep - 1)
        self.t_steps_fn = lambda idx: (self.sigma_min ** (1 / self.rho) + idx / (self.n_timestep - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        # Add iCT training steps
        k_prime = jnp.floor(diffusion_framework['train']['total_step'] / (jnp.log2(jnp.floor(self.s_1 / self.s_0)) + 1))
        self.ict_maximum_step_fn = lambda cur_step: jnp.minimum(self.s_0 * jnp.power(2, jnp.floor(cur_step / k_prime)), self.s_1) + 1
        self.ict_t_steps_fn = lambda idx, N_k: (self.sigma_min ** (1 / self.rho) + idx / (N_k - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        # Create ema obj
        ema_config = config.ema
        self.ema_obj = CMEMA(**ema_config)

        # Define loss functions
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16', replicate=False)


        def edm_sigma_sampling_fn(rng_key, y):
            p_mean = -1.2
            p_std = 1.2
            rnd_normal = jax.random.normal(rng_key, (y.shape[0], 1, 1, 1))
            sigma = jnp.exp(rnd_normal * p_std + p_mean)
            sigma_idx = self.t_steps_inv_fn(sigma)
            prev_sigma = self.t_steps_fn(jnp.where((sigma_idx - 1) > 0, sigma_idx - 1, 0))
            return sigma, prev_sigma
        
        def ict_sigma_sampling_fn(rng_key, y, step):
            p_mean = -1.1
            p_std = 2.0
            N_k = self.ict_maximum_step_fn(cur_step=step)

            # First, prepare range list from 0 to self.s_1 (include)
            overall_idx = jnp.arange(self.s_1 + 1)

            # Then, if the value is larger than the maximum step value, set it to the maximum step value.
            overall_idx = jnp.where(overall_idx < N_k, overall_idx, N_k - 1)

            # Calculate erf of standardizated sigma for sampling from categorical distribution 
            # This process is imitation of discrete lognormal distribution. (please refer to the paper)
            overall_standardized_sigma = (jnp.log(self.ict_t_steps_fn(overall_idx, N_k)) - p_mean) / (jnp.sqrt(2) * p_std)
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
            sigma = self.ict_t_steps_fn(next_idx, N_k)[:, None, None, None]
            prev_sigma = self.ict_t_steps_fn(idx, N_k)[:, None, None, None]
            return sigma, prev_sigma
    

        def get_sigma_sampling(sigma_sampling, rng_key, y, step=None):
            if sigma_sampling == "EDM":
                return edm_sigma_sampling_fn(rng_key, y)
            elif sigma_sampling == "iCT":
                return ict_sigma_sampling_fn(rng_key, y, step)
            else:
                NotImplementedError("sigma_sampling should be either EDM or CM for now.")
        
        def lpips_loss_fn(pred, target, loss_weight=1):
            # return jnp.mean(self.perceptual_loss(pred, target))
            output_shape = (target.shape[0], 224, 224, target.shape[-1])
            pred = jax.image.resize(pred, output_shape, "bilinear")
            target = jax.image.resize(target, output_shape, "bilinear")
            lpips_loss = jnp.mean(loss_weight * self.perceptual_loss(pred, target))
            return lpips_loss
        
        def pseudo_huber_loss_fn(pred, target, loss_weight=1):
            data_dim = pred.shape[1:]
            # c = 0.00054 * jnp.sqrt(data_dim[0] * data_dim[1] * data_dim[2])
            c = diffusion_framework.get("pseudo_huber_loss_c", 0.00054 * jnp.sqrt(data_dim[0] * data_dim[1] * data_dim[2]))
            # pseudo_huber = jnp.sqrt(jnp.sum((pred - target) ** 2, axis=(-1, -2, -3)) + c ** 2) - c
            pseudo_huber = jnp.sqrt((pred - target) ** 2 + c ** 2) - c
            pseudo_huber_loss = jnp.mean(loss_weight * pseudo_huber)
            return pseudo_huber_loss
        
        def get_loss(loss_type, pred, target, loss_weight=1, train=True, key_name=None):
            loss = 0
            if loss_type == "l2":
                loss = jnp.mean(loss_weight * (pred - target) ** 2)
            if loss_type == "lpips":
                loss = lpips_loss_fn(pred, target, loss_weight)
            elif loss_type in ["huber", "pseudo_huber"]:
                loss = pseudo_huber_loss_fn(pred, target, loss_weight)
            
            loss_dict = {}
            if train:
                loss_dict[f"train/{key_name if key_name is not None else loss_type}"] = loss
            else:
                loss_dict[f"eval/{key_name if key_name is not None else loss_type}"] = loss
            return loss, loss_dict

        def original_alignment_loss_fn(rng_key, torso_params, target_model, y, sigma, D_x, cm_dropout_key):
            alignment_batch_size = diffusion_framework['alignment_batch_size']
            num_data_samples = diffusion_framework['num_samples_for_alignment']

            noise_2_key, rng_key = jax.random.split(rng_key, 2)
            noise_2 = jax.random.normal(noise_2_key, (alignment_batch_size, *y.shape[1:]))
            
            sigma_samples = jax.random.choice(rng_key, sigma, shape=(num_data_samples,), replace=False)
            data_samples = jax.random.choice(rng_key, y, shape=(num_data_samples,), replace=False)
            D_x_samples = jax.random.choice(rng_key, D_x, shape=(num_data_samples,), replace=False)

            perturbed_D_x = jnp.reshape(jax.vmap(lambda x, t: x + t * noise_2)(D_x_samples, sigma_samples), (-1, *y.shape[1:]))
            sigma_samples = jnp.repeat(sigma_samples, alignment_batch_size)[:, None, None, None]

            new_D_x, _ = self.model.apply(
                {'params': torso_params}, x=perturbed_D_x, sigma=sigma_samples,
                train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})

            cm_mean = jnp.mean(jnp.reshape(new_D_x, (num_data_samples, alignment_batch_size, *y.shape[1:])), axis=1)
            return jnp.mean((cm_mean - data_samples) ** 2)

        # def original_alignment_loss_fn(rng_key, torso_params, target_model, y, sigma, D_x, cm_dropout_key):
        #     rng_key, noise_key = jax.random.split(rng_key, 2)

        #     noise = jax.random.normal(noise_key, y.shape)

        #     stopgrad_D_x = jax.lax.stop_gradient(D_x)
        #     perturbed_D_x = stopgrad_D_x + sigma * noise

        #     new_D_x_1, _ = self.model.apply(
        #         {'params': torso_params}, x=perturbed_D_x, sigma=sigma,
        #         train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})
        #     # return jnp.mean(new_D_x_1 * (new_D_x_2 - y))
        #     return jnp.mean((new_D_x_1 - y) ** 2)

        def pseudo_alignment_loss_fn(rng_key, torso_params, target_model, y, sigma, D_x, cm_dropout_key):
            rng_key, noise_1_key, noise_2_key = jax.random.split(rng_key, 3)

            noise_1 = jax.random.normal(noise_1_key, y.shape)
            noise_2 = jax.random.normal(noise_2_key, y.shape)

            stopgrad_D_x = jax.lax.stop_gradient(D_x)
            perturbed_D_x_1 = stopgrad_D_x + sigma * noise_1
            perturbed_D_x_2 = stopgrad_D_x + sigma * noise_2

            new_D_x_1, _ = self.model.apply(
                {'params': torso_params}, x=perturbed_D_x_1, sigma=sigma,
                train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})
            new_D_x_2, _ = self.model.apply(
                {'params': target_model}, x=perturbed_D_x_2, sigma=sigma,
                train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})
            return jnp.mean(new_D_x_1 * (new_D_x_2 - y))

        @jax.jit
        def monitor_metric_fn(params, y, rng_key, current_step):
            torso_params = jax.lax.stop_gradient(params.get('torso_state', self.torso_state.params_ema))

            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)

            # idx = jax.random.randint(step_key, (y.shape[0], ), minval=1, maxval=self.n_timestep)
            # sigma = self.t_steps[idx][:, None, None, None]
            # prev_sigma = self.t_steps[idx-1][:, None, None, None]
            # sigma, prev_sigma = get_sigma_sampling(diffusion_framework['sigma_sampling'], step_key, y, current_step)
            sigma, prev_sigma = get_sigma_sampling(diffusion_framework['sigma_sampling_joint'], step_key, y, current_step)
            
            noise = jax.random.normal(noise_key, y.shape)
            perturbed_x = y + sigma * noise

            # Get D_x
            D_x, aux = self.model.apply(
                {'params': torso_params}, x=perturbed_x, sigma=sigma,
                train=True, augment_labels=None, rngs={'dropout': dropout_key})
            

            # Pseudo alignment loss
            rng_key, pseudo_alignment_key = jax.random.split(rng_key, 2)
            pseudo_alignment_loss = pseudo_alignment_loss_fn(pseudo_alignment_key, torso_params, torso_params, y, sigma, D_x, dropout_key)

            # Original alignment loss
            rng_key, original_alignment_key = jax.random.split(rng_key, 2)
            original_alignment_loss = original_alignment_loss_fn(original_alignment_key, torso_params, torso_params, y, sigma, D_x, dropout_key)

            loss_dict = {}
            loss_dict['eval/pseudo_alignment_loss'] = pseudo_alignment_loss
            loss_dict['eval/original_alignment_loss'] = original_alignment_loss
            return loss_dict


        def joint_training_loss_fn(update_params, total_states_dict, args, has_aux=False):
            # Set loss dict and total loss
            loss_dict = {}
            total_loss = 0

            torso_params = update_params['torso_state']
            head_params = update_params['head_state']
            target_model = jax.lax.stop_gradient(torso_params) 

            # Unzip arguments for loss_fn
            y, rng_key = args
            
            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)
            noise = jax.random.normal(noise_key, y.shape)
            
            # Sigma sampling
            # current_step = jnp.floor(total_states_dict['torso_state'].step / self.step_scale).astype(int)
            current_step = jnp.floor(total_states_dict['torso_state'].opt_state.gradient_step).astype(int)
            sigma, prev_sigma = get_sigma_sampling(diffusion_framework['sigma_sampling_joint'], step_key, y, current_step)
            perturbed_x = y + sigma * noise
            prev_perturbed_x = y + prev_sigma * noise

            # Get consistency function values
            cm_dropout_key, dropout_key = jax.random.split(dropout_key, 2)
          
            def discrete_loss(torso_params, target_model, y, sigma, prev_sigma, perturbed_x, prev_perturbed_x, dropout_key):
                D_x, aux = self.model.apply(
                    {'params': torso_params}, x=perturbed_x, sigma=sigma,
                    train=True, augment_labels=None, rngs={'dropout': dropout_key})
                
                prev_D_x, _ = self.model.apply(
                    {'params': target_model}, x=prev_perturbed_x, sigma=prev_sigma,
                    train=True, augment_labels=None, rngs={'dropout': dropout_key})
                # Get consistency loss
                loss_weight = 1 / (sigma - prev_sigma)
                consistency_loss, consistency_loss_dict = get_loss(diffusion_framework['loss'], D_x, prev_D_x, loss_weight=loss_weight, train=True)
                return consistency_loss, consistency_loss_dict, (D_x, aux)

            consistency_loss, consistency_loss_dict, func_val = discrete_loss(torso_params, target_model, y, sigma, prev_sigma, perturbed_x, prev_perturbed_x, cm_dropout_key)
            total_loss += consistency_loss
            loss_dict.update(consistency_loss_dict)
            
            D_x, aux = func_val
            head_D_x = jax.lax.stop_gradient(D_x)
            head_t_emb, head_last_x_emb = jax.lax.stop_gradient(aux) if not diffusion_framework['gradient_flow_from_head'] else aux

            denoised, aux = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, 
                F_x=head_D_x, t_emb=head_t_emb, last_x_emb=head_last_x_emb,
                train=True, augment_labels=None, rngs={'dropout': dropout_key})

            # Get DSM
            sigma_data = 0.5
            weight = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
            joint_training_weight = diffusion_framework.get('joint_training_weight', 1)
            dsm_loss = jnp.mean(joint_training_weight * weight * (denoised - y) ** 2)
            total_loss += dsm_loss
            loss_dict['train/head_dsm_loss'] = dsm_loss

            if diffusion_framework['alignment_loss']:
                alignment_loss_fn = pseudo_alignment_loss_fn if diffusion_framework['alignment_type'] != "original" else original_alignment_loss_fn
                alignment_term = jax.lax.cond(
                    current_step <= diffusion_framework['alignment_threshold'],
                    lambda *args: 0.0,
                    # pseudo_alignment_loss_fn,
                    alignment_loss_fn,
                    rng_key, torso_params, target_model, y, sigma, D_x, cm_dropout_key
                )

                if diffusion_framework.get("alignment_loss_weight", "uniform") == "lognormal":
                    p_mean = -1.1
                    p_std = 2.0
                    alignment_loss_weight = jnp.exp(-0.5 * ((jnp.log(sigma) - p_mean) / p_std) ** 2)
                    alignment_loss_weight *= 1 / (p_std * jnp.sqrt(2 * jnp.pi))
                else:
                    alignment_loss_weight = 1
                alignment_loss = jnp.mean(alignment_loss_weight * alignment_term)

                total_loss += diffusion_framework['alignment_loss_scale'] * alignment_loss
                loss_dict['train/alignment_loss'] = alignment_loss

            return total_loss, loss_dict


        def update_params_fn(
                states_dict: dict, 
                update_states_key_list: list, 
                loss_fn, 
                loss_fn_args, 
                loss_dict_tail: dict = {}):
            update_params_dict = {params_key: states_dict[params_key].params for params_key in update_states_key_list}
            (_, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(update_params_dict, states_dict, loss_fn_args)
            grads_mean = {params_key: jax.lax.pmean(grads[params_key], axis_name=self.pmap_axis) for params_key in update_states_key_list}
            updated_states = {params_key: states_dict[params_key].apply_gradients(grads=grads_mean[params_key]) 
                              for params_key in update_states_key_list}
            loss_dict_tail.update(loss_dict)

            # Before weight update, measure the distance between current model and target model
            prev_param = jax.lax.stop_gradient(states_dict['torso_state'].params)
            current_param = jax.lax.stop_gradient(updated_states['torso_state'].params)
            distance = jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x),
                    jax.tree_util.tree_map(lambda x, y: (x - y) ** 2, 
                                           jax.tree_util.tree_leaves(prev_param), 
                                           jax.tree_util.tree_leaves(current_param)), 0)
            loss_dict_tail['train/weight_update_distance'] = jnp.sqrt(distance)

            # gradient norm
            torso_grads = grads_mean['torso_state']
            grad_l2_norms = jnp.sqrt(jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x ** 2), 
                jax.tree_util.tree_leaves(torso_grads), 0))
            loss_dict_tail['train/grad_l2_norms'] = grad_l2_norms

            states_dict.update(updated_states)
            return states_dict, loss_dict_tail
            

        # Define update function
        def update(carry_state, x0):
            (rng, states) = carry_state
            rng, new_rng = jax.random.split(rng)

            loss_dict = {}
            
            head_torso_key = ["head_state", "torso_state"]
            rng, head_torso_rng = jax.random.split(rng)
            states, loss_dict = update_params_fn(states, head_torso_key, joint_training_loss_fn, (x0, head_torso_rng), loss_dict)

            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)
            
            # Update EMA for sampling
            # states = {state_name: self.ema_obj.ema_update(state_content) for state_name, state_content in states.items()}
            def ema_update_fn(states, step):
                updated_states = {state_name: self.ema_obj.ema_update(state_content, step) for state_name, state_content in states.items()}
                return updated_states
            # effective_step = states[head_torso_key[0]].step // self.step_scale
            # states = jax.lax.cond(
            #     states[head_torso_key[0]].step % self.step_scale == 0,
            #     ema_update_fn, lambda x, step: x, states, effective_step)
            is_updated = states[head_torso_key[0]].tx.has_updated(states[head_torso_key[0]].opt_state)
            states = jax.lax.cond(
                is_updated,
                ema_update_fn, lambda x, step: x, states, states[head_torso_key[0]].opt_state.gradient_step)
            

            new_carry_state = (new_rng, states)
            return new_carry_state, loss_dict


        # Define p_sample_jit functions for sampling
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        def sample_edm_fn(params, x_cur, rng_key, gamma, t_cur, t_prev):
            # model_params = params.get('model_state', None)
            torso_params = params.get('torso_state', self.torso_state.params_ema)
            head_params = params['head_state']

            rng_key, dropout_key, dropout_key_2 = jax.random.split(rng_key, 3)

            # Increase noise temporarily.
            t_hat = t_cur + gamma * t_cur
            noise = jax.random.normal(rng_key, x_cur.shape)
            x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * noise

            # Euler step
            D_x, aux = self.model.apply(
                {'params': torso_params}, x=x_hat, sigma=t_hat, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            t_emb, last_x_emb = aux

            denoised, aux = self.head.apply(
                {'params': head_params}, x=x_hat, sigma=t_hat, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=False, augment_labels=None, rngs={'dropout': dropout_key}
            )

            # predicted_score = - (x_hat - denoised) ** 2 / (t_hat ** 2)
            d_cur = (x_hat - denoised) / t_hat
            euler_x_prev = x_hat + (t_prev - t_hat) * d_cur

            # Apply 2nd order correction.
            def second_order_corrections(euler_x_prev, t_prev, x_hat, t_hat, d_cur, rng_key):
                D_x, aux = self.model.apply(
                    {'params': torso_params}, x=euler_x_prev, sigma=t_prev,
                    train=False, augment_labels=None, rngs={'dropout': rng_key})
                t_emb, last_x_emb = aux

                denoised, aux = self.head.apply(
                    {'params': head_params}, x=euler_x_prev, sigma=t_prev, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                    train=False, augment_labels=None, rngs={'dropout': dropout_key}
                )

                d_prime = (euler_x_prev - denoised) / t_prev
                heun_x_prev = x_hat + 0.5 * (t_prev - t_hat) * (d_cur + d_prime)
                return heun_x_prev
            
            heun_x_prev = jax.lax.cond(t_prev != 0.0,
                                    second_order_corrections,
                                    lambda euler_x_prev, t_prev, x_hat, t_hat, dx_dt, rng_key: euler_x_prev,
                                    euler_x_prev, t_prev, x_hat, t_hat, d_cur, dropout_key_2)
            return heun_x_prev, D_x

        def sample_cm_fn(params, x_cur, rng_key, gamma=None, t_cur=None, t_prev=None):
            dropout_key = rng_key
            denoised, aux = self.model.apply(
                {'params': params}, x=x_cur, sigma=t_cur, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            return denoised

        self.p_sample_edm = jax.pmap(sample_edm_fn)
        self.p_sample_cm = jax.pmap(sample_cm_fn)
        self.update_fn = jax.pmap(partial(jax.lax.scan, update), 
                                  axis_name=self.pmap_axis)
        self.eval_fn = jax.pmap(monitor_metric_fn, axis_name=self.pmap_axis)
    
    def get_training_states_params(self):
        return {state_name: state_content.params for state_name, state_content in self.training_states.items()}
    
    def init_model_state(self, config: DictConfig):
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        input_format = jnp.ones([1, *config.dataset.data_size])
        
        torso_params = self.model.init(
            rng_dict, x=input_format, sigma=jnp.ones([1,]), train=False, augment_labels=None)['params']
        
        self.rand_key, dummy_rng = jax.random.split(self.rand_key, 2)
        D_x, aux = self.model.apply(
                {'params': torso_params}, x=input_format, sigma=jnp.ones([1,]), 
                train=False, augment_labels=None, rngs={'dropout': dummy_rng})
        model_tx = jax_utils.create_optimizer(config, "diffusion")
        new_torso_state = jax_utils.TrainState.create(
            apply_fn=self.model.apply,
            params=torso_params,
            params_ema=torso_params,
            tx=model_tx
        )
        
        t_emb, last_x_emb = aux
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        head_params = self.head.init(rng_dict, x=input_format, sigma=jnp.ones([1,]), F_x=D_x, last_x_emb=last_x_emb, t_emb=t_emb,
                                     train=False, augment_labels=None)['params']

        head_tx = jax_utils.create_optimizer(config, "diffusion")
        
        new_head_state = jax_utils.TrainState.create(
            apply_fn=self.head.apply,
            params=head_params,
            params_ema=head_params,
            tx=head_tx
        )
        return new_torso_state, new_head_state


    def get_model_state(self):
        if self.distributed_training:
            # training_states = {model_key: jax.tree_util.tree_map(lambda x: jax_utils.fully_replicated_host_local_array_to_global_array(x), self.training_states[model_key])
            #                     for model_key in self.training_states.keys()}
            # return {
            #     "diffusion": training_states['torso_state'], 
            #     "head": training_states['head_state']
            # }
            return {
                "diffusion": self.training_states['torso_state'], 
                "head": self.training_states['head_state']
            }
        else:
            training_states = self.training_states
            return {
                "diffusion": flax.jax_utils.unreplicate(training_states['torso_state']), 
                "head": flax.jax_utils.unreplicate(training_states['head_state'])
            }
        # return {
        #     "diffusion": jax_utils.unreplicate_tree(self.training_states['torso_state']),
        #     "head": jax_utils.unreplicate_tree(self.training_states['head_state'])
        # }
        # training_states = self.training_states
        # return {
        #         "diffusion": flax.jax_utils.unreplicate(training_states['torso_state']), 
        #         "head": flax.jax_utils.unreplicate(training_states['head_state'])
        #     }
    
    def fit(self, x0, cond=None, step=0, eval_during_training=False):
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)
        # new_carry, loss_dict_stack = self.update_fn((dropout_key, self.head_state), x0)
        x0 = jnp.asarray(x0)
        new_carry, loss_dict_stack = self.update_fn((dropout_key, self.training_states), x0)
        (_, training_states) = new_carry

        loss_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), loss_dict_stack)
        self.training_states = training_states

        eval_dict = {}
        if eval_during_training:
            eval_key, self.rand_key = jax.random.split(self.rand_key, 2)
            eval_key = jnp.asarray(jax.random.split(eval_key, jax.local_device_count()))
            current_step = jnp.asarray([step] * jax.local_device_count())
            # TODO: self.eval_fn is called using small batch size. This is not good for evaluation.
            # Need to use large batch size (e.g. using scan function.)
            eval_dict = self.eval_fn(self.get_training_states_params(), x0[:, 0], eval_key, current_step)
            eval_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), eval_dict)

        return_dict = {}
        return_dict.update(loss_dict)
        return_dict.update(eval_dict)
        self.wandblog.update_log(return_dict)
        return return_dict

    def sampling_edm(self, num_image, img_size=(32, 32, 3), original_data=None, mode="edm"):
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
            gamma = 0

            rng_key = jax.random.split(rng_key, jax.local_device_count())
            t_cur = jnp.asarray([t_cur] * jax.local_device_count())
            t_next = jnp.asarray([t_next] * jax.local_device_count())
            gamma = jnp.asarray([gamma] * jax.local_device_count())
            if mode == "edm":
                latent_sample, _ = self.p_sample_edm(
                    {state_name: state_content.params_ema for state_name, state_content in self.training_states.items()},
                    latent_sample, rng_key, gamma, t_cur, t_next)
        latent_sample = latent_sample.reshape(num_image, *img_size)
        return latent_sample

    def sampling_edm_and_cm(self, num_image, img_size=(32, 32, 3), original_data=None, mode="edm"):
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        step_indices = jnp.arange(self.n_timestep)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.n_timestep - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = jnp.append(t_steps, jnp.zeros_like(t_steps[0]))
        pbar = tqdm(zip(t_steps[:-1], t_steps[1:]))

        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * t_steps[0]
        cm_sample = None
        for t_cur, t_next in pbar:
            rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
            gamma = 0

            rng_key = jax.random.split(rng_key, jax.local_device_count())
            t_cur = jnp.asarray([t_cur] * jax.local_device_count())
            t_next = jnp.asarray([t_next] * jax.local_device_count())
            gamma = jnp.asarray([gamma] * jax.local_device_count())
            latent_sample, d_x = self.p_sample_edm(
                {state_name: state_content.params_ema for state_name, state_content in self.training_states.items()},
                latent_sample, rng_key, gamma, t_cur, t_next)
            if cm_sample is None:
                cm_sample = d_x

        latent_sample = latent_sample.reshape(num_image, *img_size)
        cm_sample = cm_sample.reshape(num_image, *img_size)
        return latent_sample, cm_sample

    def sampling_cm(self, num_image, img_size=(32, 32, 3), original_data=None, mode="cm-training"):
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        # One-step generation
        latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * self.sigma_max
        
        rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
        rng_key = jax.random.split(rng_key, jax.local_device_count())
        gamma = jnp.asarray([0] * jax.local_device_count())
        t_max = jnp.asarray([self.sigma_max] * jax.local_device_count())
        t_min = jnp.asarray([self.sigma_min] * jax.local_device_count())

        if mode == "cm-training" or mode == "one-step":
            sampling_params = self.training_states['torso_state'].params_ema
        elif mode == "cm-not-training":
            sampling_params = self.torso_state.params_ema
            sampling_params = flax.jax_utils.replicate(sampling_params)

        latent_sample = self.p_sample_cm(sampling_params, latent_sample, rng_key, gamma, t_max, t_min)


        latent_sample = latent_sample.reshape(num_image, *img_size)
        return latent_sample
    
    def sampling_cm_two_step(self, num_image, img_size=(32, 32, 3), original_data=None, mode="two-step"):
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        # One-step generation
        # latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * self.sigma_max
        
        # rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
        # rng_key = jax.random.split(rng_key, jax.local_device_count())
        # gamma = jnp.asarray([0] * jax.local_device_count())
        # t_max = jnp.asarray([self.sigma_max] * jax.local_device_count())
        # t_min = jnp.asarray([self.sigma_min] * jax.local_device_count())

        # if mode == "cm-training":
        #     sampling_params = self.training_states['torso_state'].params_ema
        # elif mode == "cm-not-training":
        #     sampling_params = self.torso_state.params_ema
        #     sampling_params = flax.jax_utils.replicate(sampling_params)

        # latent_sample = self.p_sample_cm(sampling_params, latent_sample, rng_key, gamma, t_max, t_min)
        ts = [self.sigma_max, 0.661, self.sigma_min]
        
        x = jax.random.normal(sampling_key, latent_sampling_tuple) * self.sigma_max

        # if mode == "cm-training":
            # sampling_params = self.training_states['torso_state'].params_ema
        # elif mode == "cm-not-training":
        #     sampling_params = self.torso_state.params_ema
        #     sampling_params = flax.jax_utils.replicate(sampling_params)    
        sampling_params = self.training_states['torso_state'].params_ema

        for i in range(len(ts) - 1):
            # t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            t = ts[i]

            # x0 = distiller(x, t * s_in)
            sampling_key, p_sample_key = jax.random.split(sampling_key, 2)
            p_sample_key = jax.random.split(p_sample_key, jax.local_device_count())
            
            t_param = jnp.asarray([t] * jax.local_device_count())
            t_min = jnp.asarray([self.sigma_min] * jax.local_device_count())
            gamma = jnp.zeros((jax.local_device_count(),))

            # x0 = sampling_fn(params, x, p_sample_key, gamma, t_param, t_min_param)
            # next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            x = self.p_sample_cm(sampling_params, x, p_sample_key, gamma, t_param, t_min)
            next_t = ts[i+1]
            next_t = jnp.clip(next_t, self.sigma_min, self.sigma_max)

            sampling_key, normal_rng = jax.random.split(sampling_key, 2)
            x = x + jax.random.normal(normal_rng, latent_sampling_tuple) * jnp.sqrt(next_t**2 - self.sigma_min**2)

        x = x.reshape(num_image, *img_size)
        return x

    def sampling_cm_intermediate(self, num_image, img_size=(32, 32, 3), original_data=None, sweep_timesteps=17, noise=None, sigma_scale=None):
        latent_sampling_tuple = (jax.local_device_count(), num_image // jax.local_device_count(), *img_size)
        sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)

        # One-step generation
        # latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * self.sigma_max
        sampling_t_steps = jnp.flip(self.t_steps)

        if noise is None:
            latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple) * sampling_t_steps[sweep_timesteps]
        else:
            latent_sample = noise
        
        if original_data is not None:
            latent_sample += original_data
        
        rng_key, self.rand_key = jax.random.split(self.rand_key, 2)
        rng_key = jax.random.split(rng_key, jax.local_device_count())
        gamma = jnp.asarray([0] * jax.local_device_count())
        current_t = jnp.asarray([sampling_t_steps[sweep_timesteps]] * jax.local_device_count())
        t_min = jnp.asarray([self.sigma_min] * jax.local_device_count())

        # sampling_params = self.torso_state.params_ema
        sampling_params = self.training_states["torso_state"].params_ema
        # sampling_params = flax.jax_utils.replicate(sampling_params)

        latent_sample = self.p_sample_cm(sampling_params, latent_sample, rng_key, gamma, current_t, t_min)
        latent_sample = latent_sample.reshape(num_image, *img_size)
        return latent_sample
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None, mode="edm"):
        # mode option: edm, cm_training, cm_not_training
        if mode == "edm":
            return self.sampling_edm(num_image, img_size, original_data, mode)
        elif "cm" in mode:
            return self.sampling_cm(num_image, img_size, original_data, mode)
        elif "one-step" in mode:
            return self.sampling_cm(num_image, img_size, original_data, mode)
        elif "two-step" in mode:
            return self.sampling_cm_two_step(num_image, img_size, original_data, mode)