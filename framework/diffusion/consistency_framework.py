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
        
        self.model_state, self.head_state = self.init_model_state(config)
        # print(parameter_overview.get_parameter_overview(self.head_state.params))
        # self.model_state = fs_obj.load_model_state("diffusion", self.model_state, checkpoint_dir='pretrained_models/cd_750k')
        # breakpoint()
        checkpoint_dir = "experiments/cm_distillation_ported_from_torch_ve/checkpoints"
        for checkpoint in os.listdir(config.exp.checkpoint_dir):
            if "diffusion" in checkpoint:
                checkpoint_dir = config.exp.checkpoint_dir
                break
        self.model_state = fs_obj.load_model_state("diffusion", self.model_state, checkpoint_dir=checkpoint_dir)
        self.head_state = fs_obj.load_model_state("head", self.head_state)

        # Replicate states for training with pmap
        self.training_states = {"head_state": self.head_state}
        self.CM_freeze = diffusion_framework['CM_freeze']
        if not self.CM_freeze:
            self.training_states["model_state"] = self.model_state
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

        # Create ema obj
        ema_config = config.ema
        self.ema_obj = CMEMA(**ema_config)

        # Define loss functions
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16', replicate=False)
        
        @jax.jit
        def model_default_output_fn(params, y, sigma, prev_sigma, noise, rng_key, eval_mode=False):
            model_params = params.get('model_state', self.model_state.params_ema)
            head_params = params['head_state']

            rng_key, dropout_key = jax.random.split(rng_key, 2)

            perturbed_x = y + sigma * noise

            # Get consistency function values
            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            model_train_flag = True if "model_state" in params else False ## NEW!
            model_train_flag = model_train_flag and not eval_mode
            D_x, aux = self.model.apply(
                {'params': model_params}, x=perturbed_x, sigma=sigma,
                train=model_train_flag, augment_labels=None, rngs={'dropout': dropout_key_2})
            
            F_x, t_emb, last_x_emb = aux

            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            head_train_flag = not eval_mode
            denoised, head_aux = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=head_train_flag, augment_labels=None, rngs={'dropout': dropout_key_2})

            return denoised, D_x, head_aux

        @jax.jit
        def monitor_metric_fn(params, y, rng_key):
            model_params = params.get('model_state', self.model_state.params_ema)
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
                {'params': model_params}, x=perturbed_x, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': dropout_key_2})
            
            F_x, t_emb, last_x_emb = aux

            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            denoised, head_aux = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=False, augment_labels=None, rngs={'dropout': dropout_key_2})

            # Additional loss for monitoring training.
            dx_dt = (perturbed_x - denoised) / sigma
            prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * dx_dt # Euler step
            
            rng_key, dropout_key = jax.random.split(rng_key, 2)
            prev_D_x, aux = self.model.apply(
                {'params': model_params}, x=prev_perturbed_x, sigma=prev_sigma,
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
            original_D_x, aux = self.model.apply(
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
        def distill_loss_fn(params, y, rng_key, has_aux=False):
            model_params = params.get('model_state', self.model_state.params_ema)
            head_params = params['head_state']

            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)

            idx = jax.random.randint(step_key, (y.shape[0], ), minval=1, maxval=self.n_timestep)
            sigma = self.t_steps[idx][:, None, None, None]
            prev_sigma = self.t_steps[idx-1][:, None, None, None]
            noise = jax.random.normal(noise_key, y.shape)

            # denoised, D_x, aux = model_default_output_fn(params, y, sigma, prev_sigma, noise, rng_key)
            model_params = params.get('model_state', self.model_state.params_ema)
            head_params = params['head_state']

            rng_key, dropout_key = jax.random.split(rng_key, 2)

            perturbed_x = y + sigma * noise

            # Get consistency function values
            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            D_x, aux = self.model.apply(
                {'params': model_params}, x=perturbed_x, sigma=sigma,
                train=not self.CM_freeze, augment_labels=None, rngs={'dropout': dropout_key_2})
            
            F_x, t_emb, last_x_emb = aux

            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            denoised, aux = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=True, augment_labels=None, rngs={'dropout': dropout_key_2})

            if self.head_type == 'score_pde':
                dh_dx_inv, dh_dt = aux

            weight = None
            if self.head_type == 'score_pde':
                weight = 1 / (sigma ** 2) 
            else:
                sigma_data = 0.5
                weight = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
            dsm_loss = jnp.mean(weight * (denoised - y) ** 2)

            # Loss and loss dict construction
            total_loss = dsm_loss
            loss_dict = {}
            loss_dict['train/dsm_loss'] = dsm_loss

            # Get consistency loss
            if diffusion_framework['lpips_loss_training']:
                # dx_dt = (perturbed_x - denoised) / sigma
                prev_perturbed_x = perturbed_x + prev_sigma * noise # Euler step
                prev_D_x, aux = self.model.apply(
                    {'params': model_params}, x=prev_perturbed_x, sigma=prev_sigma,
                    train=True, augment_labels=None, rngs={'dropout': dropout_key})
                
                output_shape = (y.shape[0], 224, 224, y.shape[-1])
                D_x = jax.image.resize(D_x, output_shape, "bilinear")
                prev_D_x = jax.image.resize(prev_D_x, output_shape, "bilinear")
                D_x = (D_x + 1) / 2.0
                prev_D_x = (prev_D_x + 1) / 2.0
                lpips_loss = jnp.mean(self.perceptual_loss(D_x, prev_D_x))
                total_loss += lpips_loss
                loss_dict['train/lpips_loss'] = lpips_loss
            
            if self.head_type == 'score_pde' and diffusion_framework['score_pde_regularizer']:
                # Get dh_dx_inv loss
                dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
                def egrad(g): # diagonal of jacobian can be calculated by egrad (elementwise grad)
                    def wrapped(x, *rest):
                        y, g_vjp = jax.vjp(lambda x: g(x, *rest)[0], x)
                        x_bar, = g_vjp(jnp.ones_like(y))
                        return x_bar
                    return wrapped

                target_dh_dx_diag = egrad(
                    lambda data: self.model.apply(
                        {'params': self.model_state.params_ema}, x=data, sigma=sigma,
                        train=False, augment_labels=None, rngs={'dropout': dropout_key_2}
                    )
                )(perturbed_x, )
                dh_dx_inv_loss = jnp.mean((dh_dx_inv * target_dh_dx_diag - jnp.ones_like(dh_dx_inv)) ** 2)

                # Get dh_dt loss
                dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
                _, target_dh_dt = jax.jvp(
                    lambda sigma: self.model.apply(
                        {'params': self.model_state.params_ema}, x=perturbed_x, sigma=sigma,
                        train=False, augment_labels=None, rngs={'dropout': dropout_key_2}
                    ),
                    (sigma, ), (jnp.ones_like(sigma), ),
                )
                target_dh_dt = target_dh_dt[0]
                dh_dt_loss = jnp.mean((dh_dt - target_dh_dt) ** 2)

                # Add to total_loss
                dh_dx_inv_weight = diffusion_framework['dh_dx_inv_weight']
                dh_dt_weight = diffusion_framework['dh_dt_weight']
                total_loss += dh_dx_inv_weight * dh_dx_inv_loss
                total_loss += dh_dt_weight * dh_dt_loss
                loss_dict['train/dh_dx_inv_loss'] = dh_dx_inv_loss
                loss_dict['train/dh_dt_loss'] = dh_dt_loss
            return total_loss, loss_dict
        
        # Define update function
        def update(carry_state, x0):
            (rng, states) = carry_state
            rng, new_rng = jax.random.split(rng)
            
            # Update head (for multiple times)
            params = {state_name: state_content.params for state_name, state_content in states.items()}
            (_, loss_dict), grads = jax.value_and_grad(distill_loss_fn, has_aux=True)(params, x0, rng)
            grads = jax.lax.pmean(grads, axis_name=self.pmap_axis)
            new_states = {state_name: state_content.apply_gradients(grads=grads[state_name]) 
                          for state_name, state_content in states.items()}
            
            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)

            # Update EMA for sampling
            new_states = {state_name: self.ema_obj.ema_update(state_content)
                            for state_name, state_content in new_states.items()}

            # TODO: idk whether the target model of cm is used for training.
            if not self.CM_freeze:
                model_state = new_states['model_state']
                ema_updated_params = jax.tree_map(
                    lambda x, y: self.target_model_ema_decay * x + (1 - self.target_model_ema_decay) * y,
                    model_state.target_model, model_state.params)
                model_state = model_state.replace(target_model = ema_updated_params)
                new_states['model_state'] = model_state

            new_carry_state = (new_rng, new_states)
            return new_carry_state, loss_dict


        # Define p_sample_jit functions for sampling
        # Multistep sampling using the distilled score
        # Progress one step towards the sample
        def sample_edm_fn(params, x_cur, rng_key, gamma, t_cur, t_prev):
            # model_params = params.get('model_state', None)
            model_params = params.get('model_state', self.model_state.params_ema)
            head_params = params['head_state']

            rng_key, dropout_key, dropout_key_2 = jax.random.split(rng_key, 3)

            # Increase noise temporarily.
            t_hat = t_cur + gamma * t_cur
            noise = jax.random.normal(rng_key, x_cur.shape) * self.S_noise
            x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * noise

            # Euler step
            D_x, aux = self.model.apply(
                {'params': model_params}, x=x_hat, sigma=t_hat, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            
            F_x, t_emb, last_x_emb = aux

            denoised, aux = self.head.apply(
                {'params': head_params}, x=x_hat, sigma=t_hat, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=False, augment_labels=None, rngs={'dropout': dropout_key}
            )

            if self.head_type == 'score_pde':
                dh_dx_inv, dh_dt = aux

            # predicted_score = - (x_hat - denoised) ** 2 / (t_hat ** 2)
            d_cur = (x_hat - denoised) / t_hat
            euler_x_prev = x_hat + (t_prev - t_hat) * d_cur

            # Apply 2nd order correction.
            def second_order_corrections(euler_x_prev, t_prev, x_hat, t_hat, d_cur, rng_key):
                D_x, aux = self.model.apply(
                    {'params': model_params}, x=euler_x_prev, sigma=t_prev,
                    train=False, augment_labels=None, rngs={'dropout': rng_key})
                
                F_x, t_emb, last_x_emb = aux

                denoised = self.head.apply(
                    {'params': head_params}, x=euler_x_prev, sigma=t_prev, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                    train=False, augment_labels=None, rngs={'dropout': dropout_key}
                )

                if self.head_type == 'score_pde':
                    denoised, aux = denoised
                    dh_dx_inv, dh_dt = aux

                d_prime = (euler_x_prev - denoised) / t_prev
                heun_x_prev = x_hat + 0.5 * (t_prev - t_hat) * (d_cur + d_prime)
                return heun_x_prev
            
            heun_x_prev = jax.lax.cond(t_prev != 0.0,
                                    second_order_corrections,
                                    lambda euler_x_prev, t_prev, x_hat, t_hat, dx_dt, rng_key: euler_x_prev,
                                    euler_x_prev, t_prev, x_hat, t_hat, d_cur, dropout_key_2)
            return heun_x_prev

        def sample_cm_fn(params, x_cur, rng_key, gamma=None, t_cur=None, t_prev=None):
            dropout_key = rng_key
            denoised, aux = self.model.apply(
                {'params': params}, x=x_cur, sigma=t_cur, 
                train=False, augment_labels=None, rngs={'dropout': dropout_key})
            return denoised

        self.p_sample_edm = jax.pmap(sample_edm_fn)
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
        
        model_params = self.model.init(
            rng_dict, x=input_format, sigma=jnp.ones([1,]), train=False, augment_labels=None)['params']
        
        D_x, aux = self.model.apply(
                {'params': model_params}, x=input_format, sigma=jnp.ones([1,]), 
                train=False, augment_labels=None, rngs={'dropout': self.rand_key})
        
        F_x, t_emb, last_x_emb = aux
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        head_params = self.head.init(rng_dict, x=input_format, sigma=jnp.ones([1,]), F_x=D_x, last_x_emb=last_x_emb, t_emb=t_emb,
                                     train=False, augment_labels=None)['params']

        model_tx = jax_utils.create_optimizer(config, "diffusion")
        head_tx = jax_utils.create_optimizer(config, "diffusion")
        new_model_state = TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            params_ema=model_params,
            target_model=model_params, # NEW!
            tx=model_tx
        )
        new_head_state = TrainState.create(
            apply_fn=self.head.apply,
            params=head_params,
            params_ema=head_params,
            tx=head_tx
        )
        return new_model_state, new_head_state


    def get_model_state(self):
        # return [flax.jax_utils.unreplicate(self.head_state)]
        if self.CM_freeze:
            return {"head": flax.jax_utils.unreplicate(self.training_states['head_state'])}
        else:
            return {
                "diffusion": flax.jax_utils.unreplicate(self.training_states['model_state']), 
                "head": flax.jax_utils.unreplicate(self.training_states['head_state'])
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

        eval_dict = {}
        if eval_during_training:
            eval_key, self.rand_key = jax.random.split(self.rand_key, 2)
            eval_key = jnp.asarray(jax.random.split(eval_key, jax.local_device_count()))
            # TODO: self.eval_fn is called using small batch size. This is not good for evaluation.
            # Need to use large batch size (e.g. using scan function.)
            eval_dict = self.eval_fn(self.get_training_states_params(), x0[:, 0], eval_key)
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
            # gamma_val = jnp.minimum(jnp.sqrt(2) - 1, self.S_churn / self.n_timestep)
            # gamma = jnp.where(self.S_min <= t_cur and t_cur <= self.S_max,
            #                 gamma_val, 0)
            gamma = 0

            rng_key = jax.random.split(rng_key, jax.local_device_count())
            t_cur = jnp.asarray([t_cur] * jax.local_device_count())
            t_next = jnp.asarray([t_next] * jax.local_device_count())
            gamma = jnp.asarray([gamma] * jax.local_device_count())
            # latent_sample = self.p_sample_edm(self.head_state.params_ema, latent_sample, rng_key, gamma, t_cur, t_next)
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
            sampling_params = self.training_states['model_state'].params_ema
        elif mode == "cm_not_training":
            sampling_params = self.model_state.params_ema
            sampling_params = flax.jax_utils.replicate(sampling_params)

        latent_sample = self.p_sample_cm(sampling_params, latent_sample, rng_key, gamma, t_max, t_min)

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

    
    