from flax.training import train_state

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints

from typing import Any

class TrainState(train_state.TrainState):
  params_ema: Any = None

def get_learning_rate_schedule(config):
  learning_rate = config['train']['learning_rate']
  if "warmup" in config['train']:
    # start_step = fs_utils.get_start_step_from_checkpoint(config)
    learning_rate = optax.warmup_exponential_decay_schedule(
      init_value=0.0,
      peak_value=learning_rate,
      warmup_steps=config['train']['warmup'],
      decay_rate=1,
      transition_steps=1
    )
  else:
    learning_rate = optax.constant_schedule(learning_rate)
  return learning_rate

def create_train_state(config, model, rng):
  """
  Creates initial 'TrainState'
  """
  rng, param_rng, dropout_rng = jax.random.split(rng, 3)
  input_format = jnp.ones([64, 32, 32, 3]) 
  params = model.init({"params": param_rng, 'dropout': dropout_rng}, 
    x=input_format, t=jnp.ones([64,]), train=False)['params']
  
  # Initialize the Adam optimizer
  # learning_rate = config['train']['learning_rate']
  learning_rate = get_learning_rate_schedule(config)
  
  optax_chain = []
  if "gradient_clip" in config['train']:
    optax_chain.append(optax.clip(config['train']['gradient_clip']))
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

def save_train_state(state, checkpoint_dir, step):
  checkpoints.save_checkpoint(checkpoint_dir, state, step)
  print(f"Saving {step} complete.")

  
def save_best_state(state, best_checkpoint_dir, step):
  checkpoints.save_checkpoint(best_checkpoint_dir, state, step, overwrite=True)
  print(f"Best {step} steps! Saving {step} in best checkpoint dir complete.")

