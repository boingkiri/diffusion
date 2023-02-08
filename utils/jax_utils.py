from flax.training import train_state

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
import flax.linen as nn

from framework.autoencoder.distribution import DiagonalGaussianDistribution

from typing import Any
from functools import partial

class TrainState(train_state.TrainState):
  params_ema: Any = None

class GradModule(nn.Module):
  module: nn.Module
  kwargs: dict
  argnums: int = 0
  has_aux: bool = False
  holomorphic: bool = False
  allow_int: bool = False
  reduce_axes=()
  
  def setup(self):
    self.created_module = self.module(**self.kwargs)
  
  def __call__(self, input_kwargs):
    return jax.grad(
      self.created_module,
      argnums=self.argnums,
      has_aux=self.has_aux,
      holomorphic=self.holomorphic,
      allow_int=self.allow_int,
      reduce_axes=self.reduce_axes)(**input_kwargs)
  
def get_framework_config(config, model_type):
  if model_type in ['diffusion', 'ddpm']:
    framework_config = config['framework']['diffusion']
  # elif model_type in ['autoencoder']:
  elif model_type in ['autoencoder', 'discriminator']:
    framework_config = config['framework']['autoencoder']
  elif model_type in ['ldm']:
    if config['framework']['train_idx'] == 1:
      framework_config = config['framework']['autoencoder']
    elif config['framework']['train_idx'] == 2:
      framework_config = config['framework']['diffusion']
  else:
    breakpoint()
  return framework_config

def get_learning_rate_schedule(config, model_type):
  tmp_config = get_framework_config(config, model_type)
  learning_rate = tmp_config['train']['learning_rate']
  if "warmup" in tmp_config['train']:
    # start_step = fs_utils.get_start_step_from_checkpoint(config)
    learning_rate = optax.warmup_exponential_decay_schedule(
      init_value=0.0,
      peak_value=learning_rate,
      warmup_steps=tmp_config['train']['warmup'],
      decay_rate=1,
      transition_steps=1
    )
  else:
    learning_rate = optax.constant_schedule(learning_rate)
  return learning_rate

def create_train_state(config, model_type, model, rng, aux_data=None, dataset='cifar10'):
  """
  Creates initial 'TrainState'
  """
  rng, param_rng, dropout_rng = jax.random.split(rng, 3)
  if config['dataset'] == 'cifar10':
    input_format = jnp.ones([1, 32, 32, 3]) 
  
  if model_type == "ddpm":
    if 'train_idx' in config['framework'].keys() and config['framework']['train_idx'] == 2:
      f_value = len(config['model']['autoencoder']['ch_mults']) # TODO: Too naive
      z_dim = config['model']['autoencoder']['embed_dim']
      input_format_shape = input_format.shape
      input_format = jnp.ones(
        [input_format_shape[0], 
         input_format_shape[1] // f_value, 
         input_format_shape[2] // f_value, 
         z_dim])
    rng_dict = {"params": param_rng, 'dropout': dropout_rng}
    params = model.init(rng_dict, x=input_format, t=jnp.ones([1,]), train=False)['params']
  elif model_type == "autoencoder":
    rng, gaussian_rng = jax.random.split(rng, 2)
    rng_dict = {"params": param_rng, 'dropout': dropout_rng, 'gaussian': gaussian_rng}
    params = model.init(rng_dict, x=input_format, train=False)['params']
  elif model_type == "discriminator":
    # aux_data : generator_model, generator_params
    kl_rng, rng = jax.random.split(rng, 2)
    input_format3 = jnp.ones([1, 32, 32, 3]) 

    generator_model: nn.Module = aux_data[0]
    generator_params = aux_data[1]
    conv_out_params = generator_params['decoder_model']['conv_out']
  
    def kl_model_init():
      reconstructions, posteriors = generator_model.apply(
        {"params": generator_params},
        x=input_format,
        train=False,
        rngs={'gaussian': kl_rng},
        method=generator_model.forward_before_conv_out
      )
      rng_dict = {"params": param_rng, 'dropout': dropout_rng}
      posteriors_kl = posteriors.kl()
      return model.init(rng_dict, inputs=input_format3, reconstructions=reconstructions, 
                          posteriors_kl=posteriors_kl, optimizer_idx=0, global_step=0, conv_out_params=conv_out_params)
    def vq_model_init():
      reconstructions, quantization_diff, ind = generator_model.apply(
        {"params": generator_params},
        x=input_format,
        train=False,
        rngs={'gaussian': kl_rng},
        method=generator_model.forward_before_conv_out
      )
      rng_dict = {"params": param_rng, 'dropout': dropout_rng}
      return model.init(rng_dict, inputs=input_format3, reconstructions=reconstructions, 
                          codebook_loss=quantization_diff, optimizer_idx=0, global_step=0, 
                          conv_out_params=conv_out_params, predicted_indices=ind)
    if config['framework']['autoencoder']['mode'] == 'KL':
      # experiment_fn_jit = jax.jit(kl_model_init)
      experiment_fn_jit = kl_model_init
    elif config['framework']['autoencoder']['mode'] == 'VQ':
      # experiment_fn_jit = jax.jit(vq_model_init)
      experiment_fn_jit = vq_model_init
    params = experiment_fn_jit()['params']
  # Initialize the Adam optimizer
  learning_rate = get_learning_rate_schedule(config, model_type)
  framework_config = get_framework_config(config, model_type)

  # Setting Optimizer
  optax_chain = []
  if "gradient_clip" in framework_config['train']:
    optax_chain.append(optax.clip(framework_config['train']['gradient_clip']))
  optimizer_config = framework_config['train']['optimizer']
  if optimizer_config['type'] == "Adam":
    betas = [0.9, 0.999] if "betas" not in optimizer_config else optimizer_config['betas']
    optax_chain.append(optax.adam(learning_rate, b1=betas[0], b2=betas[1]))
  tx = optax.chain(
    *optax_chain
  )

  # Return the training state
  return TrainState.create(
      apply_fn=model.apply,
      params=params,
      params_ema=params,
      tx=tx
  )

def save_train_state(state, checkpoint_dir, step, prefix=None):
  if prefix is None:
    prefix = "checkpoint_"
  checkpoints.save_checkpoint(checkpoint_dir, state, step, prefix=prefix)
  print(f"Saving {step} complete.")


def load_state_from_checkpoint_dir(checkpoint_dir, state, step, checkpoint_prefix="checkpoint_"):
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, prefix=checkpoint_prefix, step=step)
  print(f"{checkpoint_prefix}{state.step} loaded")
  return state

def save_best_state(state, best_checkpoint_dir, step, prefix):
  checkpoints.save_checkpoint(best_checkpoint_dir, state, step, overwrite=True, prefix=prefix)
  print(f"Best {step} steps! Saving {step} in best checkpoint dir complete.")

