from flax.training import checkpoints
from flax.training import train_state
import optax
import orbax.checkpoint


from omegaconf import DictConfig
from typing import Any

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

  # Setting gradient accumulation if necessary
  if framework_config['train'].get("gradient_accumulation_step", 1) > 1:
    tx = optax.MultiSteps(
      tx, every_k_schedule=framework_config['train']['gradient_accumulation_step'])
  return tx

def save_train_state(state, checkpoint_dir, prefix=None):
  ocp = orbax.checkpoint.PyTreeCheckpointer()
  if prefix is None:
    prefix = "checkpoint"
  checkpoint_dir = checkpoint_dir / f"{prefix}"
  ocp.save(checkpoint_dir, state)
  print(f"Saving {state.step} complete.")


