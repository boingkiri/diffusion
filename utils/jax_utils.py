from flax.training import train_state

import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax

import logging

from typing import Any

class TrainState(train_state.TrainState):
  params_ema: Any = None

def create_train_state(config, model, rng):
  """
  Creates initial 'TrainState'
  """
  rng, param_rng, dropout_rng = jax.random.split(rng, 3)
  input_format = jnp.ones([64, 32, 32, 3]) 
  params = model.init({"params": param_rng, 'dropout': dropout_rng}, 
    x=input_format, t=jnp.ones([64,]), train=False)['params']
  
  # Initialize the Adam optimizer
  learning_rate = config['train']['learning_rate']
  tx = optax.adam(learning_rate)

  logging.info("Creating train state complete.")

  # Return the training state
  return TrainState.create(
      apply_fn=model.apply,
      params=params,
      params_ema=params,
      tx=tx
  )

def save_train_state(state, params_ema, checkpoint_dir, step):
  if checkpoint_dir is None:
      checkpoint_dir = './checkpoints'
  saved_state = TrainState.create(
      apply_fn=state.apply_fn,
      params=state.params,
      params_ema=params_ema,
      tx=state.tx
  )
  checkpoints.save_checkpoint(checkpoint_dir, saved_state, step)
  print(f"Saving {step} complete.")


