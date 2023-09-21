from flax.training import train_state

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import checkpoints

from omegaconf import DictConfig
from typing import Any

from functools import partial

class TrainState(train_state.TrainState):
  params_ema: Any = None

def get_framework_config(config: DictConfig, model_type):
  if model_type in ['autoencoder', 'discriminator']:
    framework_config = config.framework.autoencoder
  elif model_type in ['ldm']:
    if config['framework']['train_idx'] == 1:
      framework_config = config.framework.autoencoder
    elif config['framework']['train_idx'] == 2:
      framework_config = config.framework.diffusion
  else:
    framework_config = config.framework.diffusion
    # framework_config = config.framework[model_type]
  return framework_config

def get_learning_rate_schedule(config: DictConfig, model_type):
  tmp_config = get_framework_config(config, model_type)
  learning_rate = tmp_config['train']['learning_rate']
  if "warmup" in tmp_config['train']:
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


def create_optimizer(config: DictConfig, model_type):
  # Initialize the optimizer
  learning_rate = get_learning_rate_schedule(config, model_type)
  framework_config = get_framework_config(config, model_type)
  # Gradient Clipping
  optax_chain = []
  if "gradient_clip" in framework_config['train']:
    optax_chain.append(optax.clip(framework_config['train']['gradient_clip']))
  optimizer_config = framework_config['train']['optimizer']

  # Setting Optimizer
  if optimizer_config['type'] == "Adam":
    betas = [0.9, 0.999] if "betas" not in optimizer_config else optimizer_config['betas']
    optax_chain.append(optax.adam(learning_rate, b1=betas[0], b2=betas[1]))
  elif optimizer_config['type'] == "radam":
    optax_chain.append(optax.radam(learning_rate))
  tx = optax.chain(
    *optax_chain
  )
  return tx


# @partial(jax.pmap, static_broadcasted_argnums=(0, 1, 2, 4))
# def create_train_state(config: DictConfig, model_type, model, rng, aux_data=None):
def create_train_state(config: DictConfig, model_type, apply_fn, params):
  """
  Creates initial 'TrainState'
  """
  tx = create_optimizer(config, model_type)

  # Return the training state
  return TrainState.create(
      apply_fn=apply_fn,
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
    print(f"Checkpoint {state.step} loaded")
    return state

def save_best_state(state, best_checkpoint_dir, step, checkpoint_prefix):
  assert type(state) is dict
  # state = state[0] # TODO: This code assume the state is give as list. Too naive.
  # checkpoints.save_checkpoint(best_checkpoint_dir, state, step, prefix=checkpoint_prefix, overwrite=True)
  for key in state:
    checkpoints.save_checkpoint(best_checkpoint_dir, state[key], step, prefix=key + "_", overwrite=True)
  print(f"Best {step} steps! Saving {step} in best checkpoint dir complete.")


