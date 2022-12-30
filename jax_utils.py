from flax.training import train_state

import jax
import jax.numpy as jnp
import optax
import numpy as np

import logging
import os

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

from typing import Any

class TrainState(train_state.TrainState):
  params_ema: Any = None

def create_train_state(model):
  """
  Creates initial 'TrainState'
  """
  rng = jax.random.PRNGKey(42)
  rng, param_rng, dropout_rng = jax.random.split(rng, 3)
  input_format = jnp.ones([64, 32, 32, 3]) 
  params = model.init({"params": param_rng, 'dropout': dropout_rng}, 
    x=input_format, t=jnp.ones([64,]), train=False)['params']
  
  # Initialize the Adam optimizer
  # tx = optax.adam(2e-5)
  tx = optax.adam(2e-4)
  # tx = optax.adam(5e-5)

  logging.info("Creating train state complete.")

  # Return the training state
  return TrainState.create(
      apply_fn=model.apply,
      params=params,
      params_ema=params,
      tx=tx
  )



# def create_train_state(model):
#   """
#   Creates initial 'TrainState'
#   """
#   rng = jax.random.PRNGKey(42)
#   rng, param_rng, dropout_rng = jax.random.split(rng, 3)
#   input_format = jnp.ones([64, 32, 32, 3]) 
#   params = model.init({"params": param_rng, 'dropout': dropout_rng}, 
#     x=input_format, t=jnp.ones([64,]), train=False)['params']
  
#   # Initialize the Adam optimizer
#   # tx = optax.adam(2e-5)
#   tx = optax.adam(2e-4)
#   # tx = optax.adam(5e-5)

#   logging.info("Creating train state complete.")

#   # Return the training state
#   return train_state.TrainState.create(
#       apply_fn=model.apply,
#       params=params,
#       tx=tx
#   )

def load_dataset_from_tfds(dataset_name="cifar10", batch_size=128):
  
  AUTOTUNE = tf.data.experimental.AUTOTUNE

  def normalize_channel_scale(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    # return image * 2 - 1
    return image, label
  
  def augmentation(image, label):
    image, label = normalize_channel_scale(image, label)
    image = tf.image.random_flip_left_right(image)
    return image, label

  
  ds = tfds.load(dataset_name, as_supervised=True)
  train_ds, test_ds = ds['train'], ds['test']

  augmented_train_ds = (
    train_ds
    .shuffle(1000)
    .repeat()
    .map(augmentation, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
  )

  return augmented_train_ds

def save_images(images, steps, savepath):
  n_images = len(images)
  # fig = plt.figure()
  f, axes = plt.subplots(n_images // 4, 4)

  images = np.clip(images, 0, 1)

  axes = np.concatenate(axes)

  for img, axis in zip(images, axes):
    axis.imshow(img)
    axis.axis('off')
  
  save_filename = os.path.join(savepath, f"{steps}.png")
  f.savefig(save_filename)

if __name__=="__main__":
  # from tqdm import tqdm
  # dataset = load_dataset_from_tfds()
  # pbar = tqdm(dataset)
  # breakpoint()
  sample = jnp.zeros((16, 32, 32, 3))
  save_images(sample, 0, "sampling")
