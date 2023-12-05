import jax
import jax.numpy as jnp

import flax
from flax.training import checkpoints

from model.unetpp import CMPrecond, ScoreDistillPrecond, EDMPrecond
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

        if diffusion_framework['is_distillation']:
            self.teacher_model = EDMPrecond(model_config, 
                                image_channels=model_config['image_channels'], 
                                model_type=model_type, 
                                sigma_min=diffusion_framework['sigma_min'],
                                sigma_max=diffusion_framework['sigma_max'])
            teacher_model_path = diffusion_framework['distillation_path']
            self.teacher_state = checkpoints.restore_checkpoint(teacher_model_path, None, prefix="diffusion_")

        self.torso_state, self.head_state = self.init_model_state(config)
        # print(parameter_overview.get_parameter_overview(self.head_state.params))
        # self.model_state = fs_obj.load_model_state("diffusion", self.model_state, checkpoint_dir='pretrained_models/cd_750k')
        torso_checkpoint_dir = diffusion_framework['torso_checkpoint_path']
        torso_prefix = "torso"
        if torso_checkpoint_dir is not None:
            torso_prefix = "diffusion"
        else:
            for checkpoint in os.listdir(config.exp.checkpoint_dir):
                if torso_prefix in checkpoint:
                    torso_checkpoint_dir = config.exp.checkpoint_dir
                    break
        # self.torso_state = fs_obj.load_model_state("torso", self.torso_state, checkpoint_dir=checkpoint_dir)
        # FIXME: For now, "diffusion" prefix is used for torso_state because of convension. 
        self.torso_state = fs_obj.load_model_state(torso_prefix, self.torso_state, 
                                                   checkpoint_dir=torso_checkpoint_dir)
        
        
        # checkpoint_dir = "experiments/0906_verification_unet_block_1/checkpoints"
        head_checkpoint_dir = diffusion_framework['head_checkpoint_path']
        head_prefix = "head"
        for checkpoint in os.listdir(config.exp.checkpoint_dir):
            if "head" in checkpoint:
                head_checkpoint_dir = config.exp.checkpoint_dir
                break
        self.head_state = fs_obj.load_model_state(head_prefix, self.head_state, 
                                                  checkpoint_dir=head_checkpoint_dir)
        
        # Set optimizer newly if initialize_previous_training_step is True
        if diffusion_framework['initialize_previous_training_step']:
            torso_tx = jax_utils.create_optimizer(config, "diffusion")
            head_tx = jax_utils.create_optimizer(config, "head")
            self.torso_state = self.torso_state.replace(tx=torso_tx)
            self.head_state = self.head_state.replace(tx=head_tx)
            self.torso_state = self.torso_state.replace(step=0)
            self.head_state = self.head_state.replace(step=0)

        # Replicate states for training with pmap
        self.training_states = {}
        self.CM_freeze = diffusion_framework['CM_freeze']
        self.only_cm_training = diffusion_framework['only_cm_training']
        if not self.CM_freeze:
            self.training_states["torso_state"] = self.torso_state
        if not self.only_cm_training:
            self.training_states["head_state"] = self.head_state
        
        self.training_states = flax.jax_utils.replicate(self.training_states)

        # Determine if get_sigma_sampling requires current step as input
        requires_steps = ["CT", "Bernoulli"]
        self.head_sigma_requires_current_step = any(list(map(lambda s: s in diffusion_framework['sigma_sampling_head'], requires_steps)))
        self.torso_sigma_requires_current_step = any(list(map(lambda s: s in diffusion_framework['sigma_sampling_torso'], requires_steps)))

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
        self.ct_step_num_fn = lambda step: jnp.ceil(jnp.sqrt(step / diffusion_framework['train']["total_step"] * ((self.s_1 + 1) ** 2 - self.s_0 ** 2) + self.s_0 ** 2) - 1) + 1
        self.target_model_ema_decay_fn = lambda step: jnp.exp(self.s_0 * jnp.log(self.mu_0) / self.ct_step_num_fn(step))
        self.ct_t_steps_fn = lambda idx, N_k: (self.sigma_min ** (1 / self.rho) + idx / (N_k - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

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
        
        def cd_sigma_sampling_fn(rng_key, y):
            idx = jax.random.randint(rng_key, (y.shape[0], ), minval=1, maxval=self.n_timestep)
            sigma = self.t_steps[idx][:, None, None, None]
            prev_sigma = self.t_steps[idx-1][:, None, None, None]
            return sigma, prev_sigma
        
        def ct_sigma_sampling_fn(rng_key, y, step):
            N_k = self.ct_step_num_fn(step)
            idx = jax.random.randint(rng_key, (y.shape[0], ), minval=1, maxval=N_k)
            sigma = self.ct_t_steps_fn(idx, N_k)[:, None, None, None]
            prev_sigma = self.ct_t_steps_fn(idx - 1, N_k)[:, None, None, None]
            return sigma, prev_sigma
        
        def cm_uniform_sigma_sampling_fn(rng_key, y):
            sigma = jax.random.uniform(
                rng_key, (y.shape[0], 1, 1, 1), 
                minval=self.sigma_min ** (1 / self.rho), maxval=self.sigma_max ** (1 / self.rho)) ** self.rho
            sigma_idx = self.t_steps_inv_fn(sigma)
            prev_sigma = self.t_steps_fn(jnp.where((sigma_idx - 1) > 0, sigma_idx - 1, 0))
            return sigma, prev_sigma
            
        
        def bernoulli_sigma_sampling_fn(rng_key, y, step):
            bernoulli_key, step_key = jax.random.split(rng_key, 2)
            bernoulli_probability = 1 - step / diffusion_framework['train']["total_step"]
            bernoulli_idx = jax.random.bernoulli(bernoulli_key, p=bernoulli_probability, shape=(y.shape[0], 1, 1, 1))
            
            # if bernoulli == 1, use CD, else EDM
            cm_sigma_tuple = cd_sigma_sampling_fn(step_key, y)
            edm_sigma_tuple = edm_sigma_sampling_fn(step_key, y)

            sigma = jnp.where(bernoulli_idx == 1, cm_sigma_tuple[0], edm_sigma_tuple[0])
            prev_sigma = jnp.where(bernoulli_idx == 1, cm_sigma_tuple[1], edm_sigma_tuple[1])
            return sigma, prev_sigma

        def quantization_edm_sigma_sampling_fn(rng_key, y):
            # Extract EDM sigma 
            edm_sigma_tuple = edm_sigma_sampling_fn(rng_key, y)
            edm_sigma = edm_sigma_tuple[0]
            edm_sigma_idx = self.t_steps_inv_fn(edm_sigma)
            # Quantize it to get previous edm sigma
            prev_edm_sigma_idx = jnp.floor(edm_sigma_idx).astype(int)
            prev_edm_sigma = self.t_steps_fn(prev_edm_sigma_idx)
            return edm_sigma, prev_edm_sigma
        
        def quantization_uniform_sigma_sampling_fn(rng_key, y):
            # Extract EDM sigma 
            cm_uniform_sigma_tuple = cm_uniform_sigma_sampling_fn(rng_key, y)
            uniform_sigma = cm_uniform_sigma_tuple[0]
            edm_sigma_idx = self.t_steps_inv_fn(uniform_sigma)
            # Quantize it to get previous edm sigma
            prev_uniform_sigma_idx = jnp.floor(edm_sigma_idx).astype(int)
            prev_uniform_sigma = self.t_steps_fn(prev_uniform_sigma_idx)
            return uniform_sigma, prev_uniform_sigma

        def bernoulli_quantization_sigma_sampling_fn(rng_key, y, step):
            bernoulli_key, step_key = jax.random.split(rng_key, 2)
            bernoulli_probability = 1 - step / diffusion_framework['train']["total_step"]
            bernoulli_idx = jax.random.bernoulli(bernoulli_key, p=bernoulli_probability, shape=(y.shape[0], 1, 1, 1))
            
            # if bernoulli == 1, use CD, else EDM
            cm_sigma_tuple = cd_sigma_sampling_fn(step_key, y)
            
            # Extract Quantization sigma
            edm_sigma_tuple = quantization_edm_sigma_sampling_fn(step_key, y)

            # result_sigmas = jnp.where(bernoulli_idx == 1, cm_sigma_tuple, edm_sigma_tuple)
            sigma = jnp.where(bernoulli_idx == 1, cm_sigma_tuple[0], edm_sigma_tuple[0])
            prev_sigma = jnp.where(bernoulli_idx == 1, cm_sigma_tuple[1], edm_sigma_tuple[1])
            return sigma, prev_sigma

    
        def bernoulli_uniform_quantization_sigma_sampling_fn(rng_key, y, step):
            bernoulli_key, step_key = jax.random.split(rng_key, 2)
            bernoulli_probability = 1 - step / diffusion_framework['train']["total_step"]
            bernoulli_idx = jax.random.bernoulli(bernoulli_key, p=bernoulli_probability, shape=(y.shape[0], 1, 1, 1))
            
            # if bernoulli == 1, use CD, else EDM
            cm_step_key, step_key = jax.random.split(step_key, 2)
            cm_sigma_tuple = cd_sigma_sampling_fn(step_key, y)
            
            # Extract EDM sigma
            quantization_step_key, step_key = jax.random.split(step_key, 2)
            edm_sigma_tuple = quantization_uniform_sigma_sampling_fn(quantization_step_key, y)
            
            # result_sigmas = jnp.where(bernoulli_idx == 1, cm_sigma_tuple, edm_sigma_tuple)
            sigma = jnp.where(bernoulli_idx == 1, cm_sigma_tuple[0], edm_sigma_tuple[0])
            prev_sigma = jnp.where(bernoulli_idx == 1, cm_sigma_tuple[1], edm_sigma_tuple[1])
            return sigma, prev_sigma

        def get_sigma_sampling(sigma_sampling, rng_key, y, step=None):
            if sigma_sampling == "EDM":
                return edm_sigma_sampling_fn(rng_key, y)
            elif sigma_sampling == "CD":
                return cd_sigma_sampling_fn(rng_key, y)
            elif sigma_sampling == "CT":
                return ct_sigma_sampling_fn(rng_key, y, step)
            elif sigma_sampling == "EDM_Quantization":
                return quantization_edm_sigma_sampling_fn(rng_key, y)
            elif sigma_sampling == "Uniform_Quantization":
                return quantization_uniform_sigma_sampling_fn(rng_key, y)
            elif sigma_sampling == "Bernoulli":
                return bernoulli_sigma_sampling_fn(rng_key, y, step)
            elif sigma_sampling == "Bernoulli_Quantization":
                return bernoulli_quantization_sigma_sampling_fn(rng_key, y, step)
            elif sigma_sampling == "Bernoulli_uniform_Quantization":
                return bernoulli_uniform_quantization_sigma_sampling_fn(rng_key, y, step)
            else:
                NotImplementedError("sigma_sampling should be either EDM or CM for now.")

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
        def head_loss_fn(update_params, total_states_dict, args, has_aux=False):
            total_loss = 0
            loss_dict = {}

            head_params = update_params['head_state']
            torso_params = jax.lax.stop_gradient(total_states_dict.get('torso_state', self.torso_state).params_ema)
            y, rng_key = args
            
            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)
            noise = jax.random.normal(noise_key, y.shape)
            
            # Sigma sampling
            step = total_states_dict['head_state'].step if self.head_sigma_requires_current_step else None
            sigma, prev_sigma = get_sigma_sampling(diffusion_framework['sigma_sampling_head'], step_key, y, step)
            perturbed_x = y + sigma * noise

            # Get consistency function values
            cm_dropout_key, dropout_key = jax.random.split(dropout_key, 2)
            D_x, aux = self.model.apply(
                {'params': torso_params}, x=perturbed_x, sigma=sigma,
                train=False, augment_labels=None, rngs={'dropout': cm_dropout_key})
            
            D_x = jax.lax.stop_gradient(D_x)
            aux = jax.lax.stop_gradient(aux)
            F_x, t_emb, last_x_emb = aux

            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            denoised, aux = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                train=True, augment_labels=None, rngs={'dropout': dropout_key_2})

            weight = None
            if self.head_type == 'score_pde':
                weight = 1 / (sigma ** 2) 
            else:
                sigma_data = 0.5
                weight = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
            dsm_loss = jnp.mean(weight * (denoised - y) ** 2)

            # Loss and loss dict construction
            total_loss += dsm_loss
            loss_dict['train/head_dsm_loss'] = dsm_loss

            if diffusion_framework['head_connection_loss']:
                rng_key, noise_key = jax.random.split(rng_key, 2)
                noise = jax.random.normal(noise_key, y.shape)
                D_x = jax.lax.stop_gradient(D_x)
                perturbed_D_x = D_x + sigma * noise
                new_D_x, aux = self.model.apply(
                    {'params': torso_params}, x=perturbed_D_x, sigma=sigma,
                    train=False, augment_labels=None, rngs={'dropout': cm_dropout_key})
                connection_loss = jnp.mean((new_D_x - denoised) ** 2)

                total_loss += connection_loss
                loss_dict['train/head_connection_loss'] = connection_loss

            if self.head_type == 'score_pde' and diffusion_framework['score_pde_regularizer']:
                # Extract dh_dx_inv, dh_dt
                dh_dx_inv, dh_dt = aux

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
                        {'params': self.torso_state.params_ema}, x=data, sigma=sigma,
                        train=False, augment_labels=None, rngs={'dropout': dropout_key_2}
                    )
                )(perturbed_x, )
                dh_dx_inv_loss = jnp.mean((dh_dx_inv * target_dh_dx_diag - jnp.ones_like(dh_dx_inv)) ** 2)

                # Get dh_dt loss
                dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
                _, target_dh_dt = jax.jvp(
                    lambda sigma: self.model.apply(
                        {'params': self.torso_state.params_ema}, x=perturbed_x, sigma=sigma,
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
                loss_dict['train/head_dh_dx_inv_loss'] = dh_dx_inv_loss
                loss_dict['train/head_dh_dt_loss'] = dh_dt_loss
            return total_loss, loss_dict
        
        @jax.jit
        def torso_loss_fn(update_params, total_states_dict, args, has_aux=False):
            # Set loss dict and total loss to return
            loss_dict = {}
            total_loss = 0

            torso_params = update_params['torso_state']
            head_params = total_states_dict.get('head_state', None)
            if head_params is not None:
                head_params = jax.lax.stop_gradient(head_params.params_ema)
            target_model = jax.lax.stop_gradient(total_states_dict['torso_state'].target_model)

            # Unzip arguments for loss_fn
            y, rng_key = args

            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)
            noise = jax.random.normal(noise_key, y.shape)

            # Sigma sampling
            step = total_states_dict['torso_state'].step if self.torso_sigma_requires_current_step else None
            sigma, prev_sigma = get_sigma_sampling(diffusion_framework['sigma_sampling_torso'], step_key, y, step)

            # denoised, D_x, aux = model_default_output_fn(params, y, sigma, prev_sigma, noise, rng_key)
            perturbed_x = y + sigma * noise

            # Get consistency function values
            cm_dropout_key, dropout_key = jax.random.split(dropout_key, 2)
            D_x, aux = self.model.apply(
                {'params': torso_params}, x=perturbed_x, sigma=sigma,
                train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})

            F_x, t_emb, last_x_emb = aux
            # Get score value with consistency function value
            def head_fn(head_params, y, sigma, noise, D_x, t_emb, last_x_emb, dropout_key):
                dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
                perturbed_x = y + sigma * noise

                denoised, aux = self.head.apply(
                    {'params': head_params}, x=perturbed_x, sigma=sigma, 
                    F_x=jax.lax.stop_gradient(D_x), t_emb=jax.lax.stop_gradient(t_emb), last_x_emb=jax.lax.stop_gradient(last_x_emb),
                    train=False, augment_labels=None, rngs={'dropout': dropout_key_2})
                return denoised

            # Use the score head output as the guidance for consistency model
            if diffusion_framework['score_feedback']:
                denoised = head_fn(head_params, y, sigma, noise, D_x, t_emb, last_x_emb, dropout_key)
                score_mul_sigma = (perturbed_x - denoised) / sigma # score * sigma
                score_feedback_ratio = total_states_dict['torso_state'].step / diffusion_framework['train']["total_step"]
                unbiased_score_mul_sigma = (perturbed_x - y)  / sigma
                if diffusion_framework['score_feedback_type'] == "interpolation":
                    score_mul_sigma = score_feedback_ratio * score_mul_sigma + \
                                        (1 - score_feedback_ratio) * unbiased_score_mul_sigma
                    prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * score_mul_sigma # Euler step
                elif diffusion_framework['score_feedback_type'] == "threshold":
                    selected_score_mul_sigma = jnp.where(
                        score_feedback_ratio <= diffusion_framework['score_feedback_threshold'],
                        unbiased_score_mul_sigma,
                        score_mul_sigma
                    )
                    prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * selected_score_mul_sigma # Euler step
            elif diffusion_framework['is_distillation']:
                assert diffusion_framework['score_feedback'] == False
                dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
                teacher_denoised = self.teacher_model.apply(
                    {'params': self.teacher_state["params_ema"]}, x=perturbed_x, sigma=sigma,
                    train=False, augment_labels=None, rngs={'dropout': dropout_key_2})
                teacher_score = (perturbed_x - teacher_denoised) / sigma     
                prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * teacher_score
            else:
                prev_perturbed_x = y + prev_sigma * noise # Unbiased score estimator
            
            # Connect the consistency output and denoiser output with connection loss
            if diffusion_framework['torso_connection_loss']:
                current_step = total_states_dict['torso_state'].step
                rng_key, noise_key = jax.random.split(rng_key, 2)
                noise = jax.random.normal(noise_key, y.shape)
                perturbed_D_x = D_x + sigma * noise
                new_D_x, aux = self.model.apply(
                    {'params': jax.lax.stop_gradient(torso_params)}, x=perturbed_D_x, sigma=sigma,
                    train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})
                unbiased_denoiser_fn = lambda head_params, y, sigma, noise, D_x, t_emb, last_x_emb, dropout_key: y
                connection_loss_denoised = jax.lax.cond(
                    current_step <= diffusion_framework['torso_connection_threshold'],
                    unbiased_denoiser_fn, head_fn, head_params, y, sigma, noise, D_x, t_emb, last_x_emb, dropout_key)
                connection_loss = jnp.mean((new_D_x - connection_loss_denoised) ** 2)

                total_loss += connection_loss
                loss_dict['train/torso_connection_loss'] = connection_loss
            
            prev_perturbed_x = jax.lax.stop_gradient(prev_perturbed_x)
            prev_D_x, aux = self.model.apply(
                {'params': target_model}, x=prev_perturbed_x, sigma=prev_sigma,
                train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})
            
            output_shape = (y.shape[0], 224, 224, y.shape[-1])
            D_x = jax.image.resize(D_x, output_shape, "bilinear")
            prev_D_x = jax.image.resize(prev_D_x, output_shape, "bilinear")
            D_x = (D_x + 1) / 2.0
            prev_D_x = (prev_D_x + 1) / 2.0
            lpips_loss = jnp.mean(self.perceptual_loss(D_x, prev_D_x))

            total_loss += lpips_loss
            loss_dict['train/torso_lpips_loss'] = lpips_loss
            
            return total_loss, loss_dict

        def joint_training_loss_fn(update_params, total_states_dict, args, has_aux=False):
            # Set loss dict and total loss
            loss_dict = {}
            total_loss = 0

            torso_params = update_params['torso_state']
            head_params = update_params['head_state']
            target_model = jax.lax.stop_gradient(total_states_dict['torso_state'].target_model)

            # Unzip arguments for loss_fn
            y, rng_key = args

            rng_key, step_key, noise_key, dropout_key = jax.random.split(rng_key, 4)
            noise = jax.random.normal(noise_key, y.shape)
            
            # Sigma sampling
            sigma, prev_sigma = get_sigma_sampling(diffusion_framework['sigma_sampling_joint'], step_key, y, total_states_dict['torso_state'].step)
            perturbed_x = y + sigma * noise

            # Get consistency function values
            cm_dropout_key, dropout_key = jax.random.split(dropout_key, 2)
            D_x, aux = self.model.apply(
                {'params': torso_params}, x=perturbed_x, sigma=sigma,
                train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})
            
            head_D_x = jax.lax.stop_gradient(D_x) if not diffusion_framework['gradient_flow_from_head'] else D_x
            head_F_x, head_t_emb, head_last_x_emb = jax.lax.stop_gradient(aux) if not diffusion_framework['gradient_flow_from_head'] else aux

            dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            denoised, aux = self.head.apply(
                {'params': head_params}, x=perturbed_x, sigma=sigma, 
                F_x=head_D_x, t_emb=head_t_emb, last_x_emb=head_last_x_emb,
                train=True, augment_labels=None, rngs={'dropout': dropout_key_2})

            if diffusion_framework['score_feedback']:
                score_feedback_ratio = total_states_dict['torso_state'].step / diffusion_framework['train']["total_step"]
                unbiased_score_estimator = jax.lax.stop_gradient(perturbed_x - y) / sigma
                learned_score_estimator = jax.lax.stop_gradient(perturbed_x - denoised) / sigma
                one_step_forward = jnp.where(
                    score_feedback_ratio <= diffusion_framework['score_feedback_threshold'],
                    unbiased_score_estimator,
                    learned_score_estimator
                )
                score_diff = unbiased_score_estimator - learned_score_estimator
                loss_dict['train/diff_btw_unbiased_estimator_and_learned_estimator'] = jnp.mean(jnp.mean(score_diff ** 2, axis=(1, 2, 3)))
            else:
                # one_step_forward = jax.lax.stop_gradient(perturbed_x - y) / sigma
                one_step_forward = noise

            prev_perturbed_x = perturbed_x + (prev_sigma - sigma) * one_step_forward
            prev_D_x, prev_aux = self.model.apply(
                {'params': target_model}, x=prev_perturbed_x, sigma=prev_sigma,
                train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})

            # prev_F_x, prev_t_emb, prev_last_x_emb = jax.lax.stop_gradient(prev_aux)
            # prev_denoising_D_x = jax.lax.stop_gradient(prev_D_x)

            # dropout_key_2, dropout_key = jax.random.split(dropout_key, 2)
            # prev_denoised, aux = self.head.apply(
            #     {'params': head_params}, x=prev_perturbed_x, 
            #     sigma=prev_sigma, F_x=prev_denoising_D_x, t_emb=prev_t_emb, last_x_emb=prev_last_x_emb,
            #     train=True, augment_labels=None, rngs={'dropout': dropout_key_2})

            # Get consistency loss
            output_shape = (y.shape[0], 224, 224, y.shape[-1])
            lpips_D_x = (jax.image.resize(D_x, output_shape, "bilinear") + 1) / 2.0
            prev_lpips_D_x = (jax.image.resize(prev_D_x, output_shape, "bilinear") + 1) / 2.0
            lpips_loss = jnp.mean(self.perceptual_loss(lpips_D_x, prev_lpips_D_x))
            total_loss += lpips_loss
            loss_dict['train/torso_lpips_loss'] = lpips_loss

            # Get DSM
            weight = None
            if self.head_type == 'score_pde':
                weight = 1 / (sigma ** 2) 
            else:
                sigma_data = 0.5
                weight = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
            dsm_loss = jnp.mean(weight * (denoised - y) ** 2)

            # Loss and loss dict construction
            total_loss += dsm_loss
            loss_dict['train/head_dsm_loss'] = dsm_loss

            if diffusion_framework['joint_connection_loss']:
                rng_key, noise_key = jax.random.split(rng_key, 2)
                noise = jax.random.normal(noise_key, y.shape)
                perturbed_D_x = D_x + sigma * noise
                new_D_x, aux = self.model.apply(
                    {'params': jax.lax.stop_gradient(torso_params)}, x=perturbed_D_x, sigma=sigma,
                    train=True, augment_labels=None, rngs={'dropout': cm_dropout_key})
                current_step = total_states_dict['torso_state'].step

                # Connection unbiased denoiser type
                if diffusion_framework['joint_connection_denoiser_type'] == "unbiased":
                    connection_loss_denoised = y
                elif diffusion_framework['joint_connection_denoiser_type'] == "STF":
                    NotImplementedError("STF is not implemented yet.") # TODO
                else:
                    NotImplementedError("joint_connection_denoiser_type should be either unbiased or STF for now.")

                # Retrieve connection loss denoised
                if type(diffusion_framework['joint_connection_threshold']) is int:
                    connection_loss_denoised = jnp.where(
                        current_step <= diffusion_framework['joint_connection_threshold'], connection_loss_denoised, denoised)
                elif diffusion_framework['joint_connection_threshold'] in ["linear_interpolate", 'li']:
                    li_ratio = current_step / diffusion_framework['train']["total_step"]
                    connection_loss_denoised = (1 - li_ratio) * connection_loss_denoised + li_ratio * denoised
                connection_loss = jnp.mean((new_D_x - connection_loss_denoised) ** 2)
                total_loss += connection_loss
                loss_dict['train/connection_loss'] = connection_loss

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
            # grads = jax.lax.pmean(grads, axis_name=self.pmap_axis)
            # updated_states = {params_key: states_dict[params_key].apply_gradients(grads=grads[params_key]) 
            #                   for params_key in update_states_key_list}
            loss_dict_tail.update(loss_dict)
            states_dict.update(updated_states)
            return states_dict, loss_dict_tail
            

        # Define update function
        def update(carry_state, x0):
            (rng, states) = carry_state
            rng, new_rng = jax.random.split(rng)

            loss_dict = {}

            if diffusion_framework['alternative_training']:
                # Update head (for multiple times)
                head_key = ["head_state"]
                rng, head_rng = jax.random.split(rng)
                states, loss_dict = update_params_fn(states, head_key, head_loss_fn, (x0, head_rng), loss_dict)
                
                if not self.CM_freeze:
                    torso_key = ['torso_state']
                    rng, torso_rng = jax.random.split(rng)
                    states, loss_dict = update_params_fn(states, torso_key, torso_loss_fn, (x0, torso_rng), loss_dict)

                    # Target model EMA (for consistency model training procedure)
                    torso_state = states['torso_state']
                    target_model_ema_decay = jnp.where(
                        diffusion_framework['sigma_sampling_torso'] == "CT", 
                        self.target_model_ema_decay_fn(torso_state.step),
                        self.target_model_ema_decay)
                    ema_updated_params = jax.tree_map(
                        lambda x, y: target_model_ema_decay * x + (1 - target_model_ema_decay) * y,
                        torso_state.target_model, torso_state.params)
                    torso_state = torso_state.replace(target_model = ema_updated_params)
                    states['torso_state'] = torso_state
            
            elif diffusion_framework['joint_training']:
                head_torso_key = ["head_state", "torso_state"]
                rng, head_torso_rng = jax.random.split(rng)
                states, loss_dict = update_params_fn(states, head_torso_key, joint_training_loss_fn, (x0, head_torso_rng), loss_dict)

                # Target model EMA (for consistency model training procedure)
                torso_state = states['torso_state']
                target_model_ema_decay = jnp.where(
                        diffusion_framework['sigma_sampling_joint'] == "CT", 
                        self.target_model_ema_decay_fn(torso_state.step),
                        self.target_model_ema_decay)
                ema_updated_params = jax.tree_map(
                    lambda x, y: target_model_ema_decay * x + (1 - target_model_ema_decay) * y,
                    torso_state.target_model, torso_state.params)
                torso_state = torso_state.replace(target_model = ema_updated_params)
                states['torso_state'] = torso_state
            
            elif diffusion_framework['only_cm_training']:
                torso_key = ['torso_state']
                rng, torso_rng = jax.random.split(rng)
                states, loss_dict = update_params_fn(states, torso_key, torso_loss_fn, (x0, torso_rng), loss_dict)

                # Target model EMA (for consistency model training procedure)
                torso_state = states['torso_state']
                if diffusion_framework['sigma_sampling_torso'] == "CT":
                    target_model_ema_decay = self.target_model_ema_decay_fn(torso_state.step)
                else:
                    target_model_ema_decay = self.target_model_ema_decay
                ema_updated_params = jax.tree_map(
                    lambda x, y: target_model_ema_decay * x + (1 - target_model_ema_decay) * y,
                    torso_state.target_model, torso_state.params)
                torso_state = torso_state.replace(target_model = ema_updated_params)
                
                states['torso_state'] = torso_state

            else:
                NotImplementedError("Training procedure should be either alternative training, joint training or only_cm_training.")
            
            for loss_key in loss_dict:
                loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key], axis_name=self.pmap_axis)
            
            # Update EMA for sampling
            states = {state_name: self.ema_obj.ema_update(state_content) for state_name, state_content in states.items()}

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
            noise = jax.random.normal(rng_key, x_cur.shape) * self.S_noise
            x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * noise

            # Euler step
            D_x, aux = self.model.apply(
                {'params': torso_params}, x=x_hat, sigma=t_hat, 
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
                    {'params': torso_params}, x=euler_x_prev, sigma=t_prev,
                    train=False, augment_labels=None, rngs={'dropout': rng_key})
                
                F_x, t_emb, last_x_emb = aux

                denoised, aux = self.head.apply(
                    {'params': head_params}, x=euler_x_prev, sigma=t_prev, F_x=D_x, t_emb=t_emb, last_x_emb=last_x_emb,
                    train=False, augment_labels=None, rngs={'dropout': dropout_key}
                )

                if self.head_type == 'score_pde':
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
        
        torso_params = self.model.init(
            rng_dict, x=input_format, sigma=jnp.ones([1,]), train=False, augment_labels=None)['params']
        
        D_x, aux = self.model.apply(
                {'params': torso_params}, x=input_format, sigma=jnp.ones([1,]), 
                train=False, augment_labels=None, rngs={'dropout': self.rand_key})
        
        F_x, t_emb, last_x_emb = aux
        self.rand_key, param_rng, dropout_rng = jax.random.split(self.rand_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        head_params = self.head.init(rng_dict, x=input_format, sigma=jnp.ones([1,]), F_x=D_x, last_x_emb=last_x_emb, t_emb=t_emb,
                                     train=False, augment_labels=None)['params']

        model_tx = jax_utils.create_optimizer(config, "diffusion")
        head_tx = jax_utils.create_optimizer(config, "diffusion")
        new_torso_state = TrainState.create(
            apply_fn=self.model.apply,
            params=torso_params,
            params_ema=torso_params,
            target_model=torso_params, # NEW!
            tx=model_tx
        )
        new_head_state = TrainState.create(
            apply_fn=self.head.apply,
            params=head_params,
            params_ema=head_params,
            tx=head_tx
        )
        return new_torso_state, new_head_state


    def get_model_state(self):
        # return [flax.jax_utils.unreplicate(self.head_state)]
        if self.CM_freeze:
            return {"head": flax.jax_utils.unreplicate(self.training_states['head_state'])}
        elif self.only_cm_training:
            return {"torso": flax.jax_utils.unreplicate(self.training_states['torso_state'])}
        else:
            return {
                "torso": flax.jax_utils.unreplicate(self.training_states['torso_state']), 
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
        # if eval_during_training:
        #     eval_key, self.rand_key = jax.random.split(self.rand_key, 2)
        #     eval_key = jnp.asarray(jax.random.split(eval_key, jax.local_device_count()))
        #     # TODO: self.eval_fn is called using small batch size. This is not good for evaluation.
        #     # Need to use large batch size (e.g. using scan function.)
        #     eval_dict = self.eval_fn(self.get_training_states_params(), x0[:, 0], eval_key)
        #     eval_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), eval_dict)

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
            # latent_sample = self.p_sample_edm(self.head_state.params_ema, latent_sample, rng_key, gamma, t_cur, t_next)
            latent_sample = self.p_sample_edm(
                {state_name: state_content.params_ema for state_name, state_content in self.training_states.items()},
                latent_sample, rng_key, gamma, t_cur, t_next)

        if original_data is not None:
            rec_loss = jnp.mean((latent_sample - original_data) ** 2)
            self.wandblog.update_log({"Diffusion Reconstruction loss": rec_loss})
        return latent_sample

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

        if mode == "cm-training":
            sampling_params = self.training_states['torso_state'].params_ema
        elif mode == "cm-not-training":
            sampling_params = self.torso_state.params_ema
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

    
    