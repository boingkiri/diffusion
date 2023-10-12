import jax
import jax.numpy as jnp
import flax

from model.unetpp import EDMPrecond, DummyCTMScore
from utils import jax_utils
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog
from utils.ema.ema_cm import CMEMA
from framework.default_diffusion import DefaultModel
import lpips_jax

from tqdm import tqdm
from typing import Any
from omegaconf import DictConfig
from functools import partial

import os

class CTMFramework(DefaultModel):
    def __init__(self, config: DictConfig, rand_key, fs_obj: FSUtils, wandblog: WandBLog):
        diffusion_framework: DictConfig = config.framework.diffusion
        self.n_timestep = diffusion_framework['n_timestep']
        self.type = diffusion_framework['type']
        self.rand_key = rand_key
        self.fs_obj = fs_obj
        self.wandblog = wandblog        
        self.pmap_axis = "batch"

        # Create UNet and its state
        model_config = {**config.model.diffusion}
        model_type = model_config.pop("type")
        self.model = DummyCTMScore(model_config, ## TODO: Fix this code when a precondition for CTM is implemented.
                               image_channels=model_config['image_channels'], 
                               model_type=model_type, 
                               sigma_min=diffusion_framework['sigma_min'],
                               sigma_max=diffusion_framework['sigma_max'])
        self.model_state = self.init_model_state(config)

        self.teacher_model = None
        if diffusion_framework.use_pretrained_score_model:
            # TODO: This code assumes that the teacher model has same structure with the training model.
            # Need to fix this code when we use different model structure for training and teacher model.
            # The restoration function will need teacher_model_config.
            teacher_model_prefix = "diffusion"
            teacher_model_checkpoint_dir = diffusion_framework['pretrained_score_path']
            self.teacher_model = EDMPrecond(model_config,
                            image_channels=model_config['image_channels'], 
                            model_type=model_type, 
                            sigma_min=diffusion_framework['sigma_min'],
                            sigma_max=diffusion_framework['sigma_max'])
            self.teacher_model_state = fs_obj.load_model_state(teacher_model_prefix, None, 
                                                checkpoint_dir=teacher_model_checkpoint_dir)

            tmp_params_ema = self.teacher_model_state["params_ema"]
            # TODO: for now, we assume that the only difference for teacher and student is the map_layer_target_noise.
            tmp_params_ema['UNetpp_0']['map_layer0_target_noise'] = self.model_state.params['UNetpp_0']['map_layer0_target_noise']
            tmp_params_ema['UNetpp_0']['map_layer1_target_noise'] = self.model_state.params['UNetpp_0']['map_layer1_target_noise']
            del tmp_params_ema['UNetpp_0']['map_augment']
            self.model_state = self.model_state.replace(params=tmp_params_ema)
            self.model_state = self.model_state.replace(params_ema=tmp_params_ema)
            self.model_state = self.model_state.replace(target_model=tmp_params_ema)

        # Replicate states for training with pmap
        self.training_states = {"torso_state": self.model_state}
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
        self.ct_steps_fn = lambda idx: jnp.ceil(jnp.sqrt(idx / diffusion_framework['train']["total_step"] * ((self.s_1 + 1) ** 2 - self.s_0 ** 2) + self.s_0 ** 2) - 1) + 1
        self.target_model_ema_decay_fn = lambda idx: jnp.exp(self.s_0 * jnp.log(self.mu_0) / self.ct_steps_fn(idx))
        self.ct_t_steps_fn = lambda idx, N_k: (self.sigma_min ** (1 / self.rho) + idx / (N_k - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        # Create ema obj
        ema_config = config.ema
        self.ema_obj = CMEMA(**ema_config)

        # Define loss functions
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16', replicate=False)

        def CTM_EDM_fn(rng_key, y):
            p_mean = -1.2
            p_std = 1.2
            rng_key, t_key = jax.random.split(rng_key, 2)
            rnd_normal = jax.random.normal(t_key, (y.shape[0], 1, 1, 1))
            t_sigma = jnp.exp(rnd_normal * p_std + p_mean)
            return t_sigma

        def CTM_late_stage_fn(rng_key, y):
            rng_key, uniform_key = jax.random.split(rng_key, 2)
            uniform_dist = jax.random.uniform(uniform_key, (y.shape[0], 1, 1, 1), maxval=0.7)
            t_sigma = self.sigma_max ** (1/self.rho) + uniform_dist * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))
            t_sigma = t_sigma ** self.rho
            return t_sigma


        @jax.jit
        def monitor_metric_fn(params, y, rng_key):
            torso_params = params.get('torso_state', self.model_state.params_ema)

            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)

            idx = jax.random.randint(step_key, (y.shape[0], ), minval=1, maxval=self.n_timestep)
            sigma = self.t_steps[idx][:, None, None, None]
            prev_sigma = self.t_steps[idx-1][:, None, None, None]
            
            noise = jax.random.normal(noise_key, y.shape)
            perturbed_x = y + sigma * noise

            # Get consistency function values
            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            D_x = self.model.apply(
                {'params': torso_params}, x=perturbed_x, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key_2})

            # Additional loss for monitoring training.
            dx_dt = (perturbed_x - D_x) / sigma
            prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * dx_dt # Euler step
            
            rng_key, dropout_key = jax.random.split(rng_key, 2)
            prev_D_x = self.model.apply(
                {'params': torso_params}, x=prev_perturbed_x, sigma=prev_sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})

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
            original_D_x = self.model.apply(
                {'params': self.model_state.params_ema}, x=y, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            original_D_x = jax.image.resize(original_D_x, output_shape, "bilinear")
            lpips_dist_btw_training_and_original_cm = jnp.mean(self.perceptual_loss(D_x, original_D_x))

            loss_dict = {}
            loss_dict['eval/l2_dist'] = l2_dist
            loss_dict['eval/lpips_dist_for_training_cm'] = lpips_dist
            loss_dict['eval/lpips_dist_btw_training_and_original_cm'] = lpips_dist_btw_training_and_original_cm
            return loss_dict
        
        @jax.jit
        def ctm_loss_fn(update_params, total_states_dict, args, has_aux=False):
            torso_params = update_params['torso_state']
            target_model = jax.lax.stop_gradient(total_states_dict['torso_state'].target_model)

            # Unzip arguments for loss_fn
            y, rng_key = args
            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)
            noise = jax.random.normal(noise_key, y.shape)

            # Get L_CTM loss
            step_key, t_key, u_key, s_key = jax.random.split(step_key, 4)
            t_sigma = jax.random.uniform(t_key, (y.shape[0], 1, 1, 1), maxval=self.sigma_max)
            s_sigma = jax.random.uniform(s_key, (y.shape[0], 1, 1, 1), maxval=t_sigma)
            u_sigma = jax.random.uniform(u_key, (y.shape[0], 1, 1, 1), minval=s_sigma, maxval=t_sigma)

            x_t = y + t_sigma * noise

            ## Get x_est
            dropout_key_tmp, dropout_key = jax.random.split(dropout_key, 2)
            x_t_s, _ = self.model.apply(
                {'params': torso_params}, x=x_t, start_sigma=t_sigma, target_sigma=s_sigma,
                train=True, augment_labels=None, rngs={'dropout': dropout_key_tmp})

            dropout_key_tmp, dropout_key = jax.random.split(dropout_key, 2)
            x_est, _ = self.model.apply(
                {'params': target_model}, x=x_t_s, start_sigma=s_sigma, target_sigma=self.sigma_min,
                train=False, augment_labels=None, rngs={'dropout': dropout_key_tmp})

            ## Get x_target
            rng_key, sampling_key = jax.random.split(rng_key, 2)
            if diffusion_framework.use_pretrained_score_model:
                x_t_u = heun_fn(self.teacher_model_state["params_ema"], x_t, sampling_key, 0, 
                                t_sigma, u_sigma, model=self.teacher_model)
            else:
                x_t_u = heun_fn(jax.lax.stop_gradient(target_model), x_t, sampling_key, 0, 
                                t_sigma, u_sigma, model=self.model)
            
            dropout_key_tmp, dropout_key = jax.random.split(dropout_key, 2)
            x_u_s, _ = self.model.apply(
                {'params': target_model}, x=x_t_u, start_sigma=u_sigma, target_sigma=s_sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key_tmp})

            dropout_key_tmp, dropout_key = jax.random.split(dropout_key, 2)
            x_target, _ = self.model.apply(
                {'params': target_model}, x=x_u_s, start_sigma=s_sigma, target_sigma=self.sigma_min,
                train=False, augment_labels=None, rngs={'dropout': dropout_key_tmp})
            
            output_shape = (y.shape[0], 224, 224, y.shape[-1])
            D_x = (jax.image.resize(x_est, output_shape, "bilinear") + 1) / 2.0
            prev_D_x = (jax.image.resize(x_target, output_shape, "bilinear") + 1) / 2.0
            lpips_loss = jnp.mean(self.perceptual_loss(D_x, prev_D_x))
            
            # Get L_DSM loss
            step_key, t_key = jax.random.split(step_key, 2)
            t_sigma = jnp.where(total_states_dict['torso_state'].step < diffusion_framework.train.total_step,
                                CTM_EDM_fn(t_key, y),
                                CTM_late_stage_fn(t_key, y))
            x_t = y + t_sigma * noise
            _, g_theta = self.model.apply(
                {'params': torso_params}, x=x_t, start_sigma=t_sigma, target_sigma=t_sigma,
                train=True, augment_labels=None, rngs={'dropout': dropout_key_tmp})
            dsm_loss = jnp.mean((y - g_theta) ** 2)

            # Loss and loss dict construction
            loss_dict = {}
            total_loss = lpips_loss + dsm_loss
            loss_dict['train/total_loss'] = total_loss
            loss_dict['train/lpips_loss'] = lpips_loss
            loss_dict['train/dsm_loss'] = dsm_loss
            
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

            # Target model EMA (for consistency model training procedure)
            torso_key = ['torso_state']
            rng, torso_rng = jax.random.split(rng)
            states, loss_dict = update_params_fn(states, torso_key, ctm_loss_fn, (x0, torso_rng), loss_dict)

            # Target model EMA (for consistency model training procedure)
            torso_state = states['torso_state']
            target_model_ema_decay = self.target_model_ema_decay
            ema_updated_params = jax.tree_map(
                lambda x, y: target_model_ema_decay * x + (1 - target_model_ema_decay) * y,
                torso_state.target_model, torso_state.params)
            torso_state = torso_state.replace(target_model = ema_updated_params)
            
            states['torso_state'] = torso_state
            
            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)
            
            # Update EMA for sampling
            states = {state_name: self.ema_obj.ema_update(state_content) for state_name, state_content in states.items()}

            new_carry_state = (new_rng, states)
            return new_carry_state, loss_dict

        # Define p_sample_jit functions for sampling
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        def heun_fn(params, x_cur, rng_key, gamma, t_cur, t_prev, model=self.model):
            torso_params = params.get('torso_state', self.model_state.params_ema)
            rng_key, dropout_key, dropout_key_2 = jax.random.split(rng_key, 3)

            # Increase noise temporarily.
            t_hat = t_cur + gamma * t_cur
            noise = jax.random.normal(rng_key, x_cur.shape) * self.S_noise
            x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * noise

            # Euler step
            D_x = model.apply(
                {'params': torso_params}, x=x_hat, sigma=t_hat, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})

            d_cur = (x_hat - D_x) / t_hat
            euler_x_prev = x_hat + (t_prev - t_hat) * d_cur

            # Apply 2nd order correction.
            def second_order_corrections(euler_x_prev, t_prev, x_hat, t_hat, d_cur, rng_key):
                D_x = model.apply(
                    {'params': torso_params}, x=euler_x_prev, sigma=t_prev,
                    train=False, augment_labels=None, rngs={'dropout': rng_key})

                d_prime = (euler_x_prev - D_x) / t_prev
                heun_x_prev = x_hat + 0.5 * (t_prev - t_hat) * (d_cur + d_prime)
                return heun_x_prev
            
            # heun_x_prev = jax.lax.cond(jnp.squeeze(t_prev) != 0.0,
            #                         second_order_corrections,
            #                         lambda euler_x_prev, t_prev, x_hat, t_hat, dx_dt, rng_key: euler_x_prev,
            #                         euler_x_prev, t_prev, x_hat, t_hat, d_cur, dropout_key_2)
            heun_x_prev = second_order_corrections(euler_x_prev, t_prev, x_hat, t_hat, d_cur, dropout_key_2)
            return heun_x_prev

        def sample_cm_fn(params, x_cur, rng_key, gamma=None, t_cur=None, t_prev=None):
            dropout_key = rng_key
            denoised = self.model.apply(
                {'params': params}, x=x_cur, start_sigma=t_cur, target_sigma=self.sigma_min, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            return denoised

        self.p_sample_edm = jax.pmap(heun_fn)
        self.p_sample_cm = jax.pmap(sample_cm_fn)
        self.update_fn = jax.pmap(partial(jax.lax.scan, update), axis_name=self.pmap_axis)
        self.eval_fn = jax.pmap(monitor_metric_fn, axis_name=self.pmap_axis)
    
    def get_training_states_params(self):
        return {state_name: state_content.params for state_name, state_content in self.training_states.items()}
    
    def init_model_state(self, config: DictConfig):
        class TrainState(jax_utils.TrainState):
            target_model: Any = None
        
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        input_format = jnp.ones([1, *config.dataset.data_size])

        torso_params = self.model.init(
            rng_dict, x=input_format, start_sigma=jnp.ones([1,]), target_sigma=jnp.ones([1,]), 
            train=False, augment_labels=None)['params']

        model_tx = jax_utils.create_optimizer(config, "diffusion")
        model_state = TrainState.create(
            apply_fn=self.model.apply,
            params=torso_params,
            params_ema=torso_params,
            target_model=torso_params, # NEW!
            tx=model_tx
        )
        return model_state

    def get_model_state(self):
        # return [flax.jax_utils.unreplicate(self.head_state)]
        return {"torso": flax.jax_utils.unreplicate(self.training_states['torso_state'])}
    
    def fit(self, x0, cond=None, step=0, eval_during_training=False):
        key, dropout_key = jax.random.split(self.rand_key, 2)
        self.rand_key = key

        # Apply pmap
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())
        dropout_key = jnp.asarray(dropout_key)
        new_carry, loss_dict_stack = self.update_fn((dropout_key, self.training_states), x0)
        (_, training_states) = new_carry

        loss_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), loss_dict_stack)
        self.training_states = training_states

        eval_dict = {}
        # if eval_during_training:
        #     eval_key, self.rand_key = jax.random.split(self.rand_key, 2)
        #     eval_key = jnp.asarray(jax.random.split(eval_key, jax.local_device_count()))
            # TODO: self.eval_fn is called using small batch size. This is not good for evaluation.
            # Need to use large batch size (e.g. using scan function.)
            # eval_dict = self.eval_fn(self.get_training_states_params(), x0[:, 0], eval_key)
            # eval_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), eval_dict)

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
            latent_sample = self.p_sample_edm(
                {state_name: state_content.params_ema for state_name, state_content in self.training_states.items()},
                latent_sample, rng_key, gamma, t_cur, t_next)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample

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

        if mode == "cm_training":
            sampling_params = self.training_states['torso_state'].params_ema
        elif mode == "cm_not_training":
            sampling_params = self.model_state.params_ema
            sampling_params = flax.jax_utils.replicate(sampling_params)

        latent_sample = self.p_sample_cm(sampling_params, latent_sample, rng_key, gamma, t_max, t_min)
        latent_sample = latent_sample[0] if type(latent_sample) == tuple else latent_sample

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None, mode="edm"):
        # mode option: edm, cm_training, cm_not_training
        if mode == "edm":
            return self.sampling_edm(num_image, img_size, original_data, mode)
        elif "cm" in mode:
            return self.sampling_cm(num_image, img_size, original_data, mode)

    
    