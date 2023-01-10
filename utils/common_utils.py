import os
import yaml
import requests
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from unet import UNet
from ddpm import DDPM
from ema import EMA
from . import jax_utils, fs_utils

import tensorflow as tf
import tensorflow_datasets as tfds

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


def get_config_from_yaml(config_dir):
    if not os.path.exists(config_dir):
        raise ValueError
    
    with open(config_dir) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_state_from_checkpoint_dir(config, state):
    checkpoint_dir = fs_utils.get_checkpoint_dir(config)
    start_num = fs_utils.get_start_step_from_checkpoint(config)
    if start_num != 0:
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, start_num)
        print(f"Checkpoint {start_num} loaded")
    return state


def init_setting(config, rng):
    fs_utils.verifying_or_create_workspace(config)
    state_rng, ddpm_rng, rng = jax.random.split(rng, 3)
    model = UNet(**config['model'])

    start_step = fs_utils.get_start_step_from_checkpoint(config)
    state = jax_utils.create_train_state(config, model, state_rng)
    state = load_state_from_checkpoint_dir(config, state)
    
    # ema_obj = EMA(**config['ema'], ema_params=state.params_ema)
    ema_obj = EMA(**config['ema'], ema_params=state.params)


    ddpm = DDPM(model, ddpm_rng, **config['ddpm'])
    return state, ddpm, start_step, ema_obj, rng

def normalize_to_minus_one_to_one(image):
    return image * 2 - 1

def unnormalize_minus_one_to_one(images):
    return (images + 1) * 0.5 

def load_dataset_from_tfds(dataset_name="cifar10", batch_size=128):
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  def normalize_channel_scale(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = normalize_to_minus_one_to_one(image)
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
  images = unnormalize_minus_one_to_one(images)

  n_images = len(images)
  f, axes = plt.subplots(n_images // 4, 4)
  images = np.clip(images, 0, 1)
  axes = np.concatenate(axes)

  for img, axis in zip(images, axes):
    axis.imshow(img)
    axis.axis('off')
  
  save_filename = os.path.join(savepath, f"{steps}.png")
  f.savefig(save_filename)
  plt.close()


def get_image_size_from_dataset(dataset):
  if dataset == "cifar10":
    return [32, 32, 3]
  else:
    raise NotImplementedError

if __name__=="__main__":
  sample = jnp.zeros((16, 32, 32, 3))
  save_images(sample, 0, "sampling")