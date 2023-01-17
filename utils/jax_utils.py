from flax.training import train_state

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints

from typing import Any

class TrainState(train_state.TrainState):
  params_ema: Any = None

def get_framework_config(config, model_type):
  if model_type in ['diffusion', 'ddpm']:
    framework_config = config['framework']['diffusion']
  elif model_type in ['autoencoder']:
    framework_config = config['framework']['autoencoder']
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

def create_train_state(config, model_type, model, rng):
  """
  Creates initial 'TrainState'
  """
  rng, param_rng, dropout_rng = jax.random.split(rng, 3)
  input_format = jnp.ones([1, 32, 32, 3]) 
  if model_type == "ddpm":
    rng_dict = {"params": param_rng, 'dropout': dropout_rng}
    params = model.init(rng_dict, x=input_format, t=jnp.ones([64,]), train=False)['params']
  else:
    rng_dict = {"params": param_rng, 'dropout': dropout_rng}
    params = model.init(rng_dict, x=input_format, train=False)['params']
  
  
  # Initialize the Adam optimizer
  learning_rate = get_learning_rate_schedule(config, model_type)
  framework_config = get_framework_config(config, model_type)

  optax_chain = []
  if "gradient_clip" in framework_config['train']:
    optax_chain.append(optax.clip(framework_config['train']['gradient_clip']))
  optax_chain.append(optax.adam(learning_rate))
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
    # if start_num != 0:
    # breakpoint()
    state = checkpoints.restore_checkpoint(checkpoint_dir, state, prefix=checkpoint_prefix, step=step)
    print(f"Checkpoint {state.step} loaded")
    return state

def save_best_state(state, best_checkpoint_dir, step):
  checkpoints.save_checkpoint(best_checkpoint_dir, state, step, overwrite=True)
  print(f"Best {step} steps! Saving {step} in best checkpoint dir complete.")

