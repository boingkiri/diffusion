import os
import requests

import flax
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pathlib


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

def get_dataset_size(dataset_name):
  if dataset_name == "cifar10":
     return (32, 32, 3)
  else:
     NotImplementedError("get_dataset_size: Not implemented.")

def normalize_to_minus_one_to_one(image):
    return image * 2 - 1

def unnormalize_minus_one_to_one(images):
    return (images + 1) * 0.5 

# def load_dataset_from_tfds(config, dataset_name="cifar10", batch_size=128, n_jitted_steps=1, x_flip=True, stf=False):
def load_dataset_from_tfds(config, dataset_name=None, batch_size=None, n_jitted_steps=None, x_flip=True):

  dataset_name = config["dataset"]["name"] if dataset_name is None else dataset_name
  batch_size = config["framework"]["diffusion"]["train"]["batch_size_per_rounds"] if batch_size is None else batch_size

  n_jitted_steps = config["n_jitted_steps"] if n_jitted_steps is None else n_jitted_steps
  x_flip = config["dataset"].get("x_flip", x_flip)

  assert n_jitted_steps >= 1

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  def normalize_channel_scale(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = normalize_to_minus_one_to_one(image)
    return image, label
  
  def augmentation(image, label):
    image, label = normalize_channel_scale(image, label)
    if x_flip is True:
      image = tf.image.random_flip_left_right(image)
    return image, label

  if dataset_name == "cifar10":
    ds = tfds.load("cifar10", as_supervised=True)
  elif dataset_name == "imagenet_64": # TODO
    ds = tfds.load("imagenet2012", split="train", as_supervised=True)
  train_ds, _ = ds['train'], ds['test']

  device_count = jax.local_device_count()
  batch_dims= [device_count, n_jitted_steps, batch_size // device_count] 

  train_ds = train_ds.shuffle(1000)
  train_ds = train_ds.repeat()
  train_ds = train_ds.map(augmentation, num_parallel_calls=AUTOTUNE)
  for dim in reversed(batch_dims):
    train_ds = train_ds.batch(dim)
  augmented_train_ds = train_ds.prefetch(AUTOTUNE)
  # it = tfds.as_numpy(augmented_train_ds)
  it = map(lambda data: jax.tree_map(lambda x: x._numpy(), data), augmented_train_ds)
  if xla_bridge.get_backend().platform == "gpu":
    it = flax.jax_utils.prefetch_to_device(it, 2)

  return it, batch_dims


def load_dataset_from_local_file(config, dataset_name=None, batch_size=None, n_jitted_steps=None, x_flip=False):
  dataset_name = config["dataset"]["name"] if dataset_name is None else dataset_name
  batch_size = config["framework"]["diffusion"]["train"]["batch_size_per_rounds"] if batch_size is None else batch_size

  n_jitted_steps = config["n_jitted_steps"] if n_jitted_steps is None else n_jitted_steps
  x_flip = config["dataset"].get("x_flip", x_flip)

  assert n_jitted_steps >= 1

  def add_dummy_label(data):
    return data, []

  def convert_path_to_numpy(data): ## Problematic
    np_data = np.load(data)
    return np_data

  if dataset_name == "celebahq":
    numpy_dir = pathlib.Path(config["dataset"]["dataset_path"])
    train_ds = tf.data.Dataset.list_files(str(numpy_dir/'*.npy'), shuffle=False)
    
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  device_count = jax.local_device_count()
  batch_dims= [device_count, n_jitted_steps, batch_size // device_count] 

  train_ds = train_ds.shuffle(1000)
  train_ds = train_ds.repeat()
  train_ds = train_ds.map(lambda path: tf.numpy_function(convert_path_to_numpy, [path], [tf.float32]), num_parallel_calls=AUTOTUNE)
  train_ds = train_ds.map(add_dummy_label, num_parallel_calls=AUTOTUNE)
  for dim in reversed(batch_dims):
    train_ds = train_ds.batch(dim)
  train_ds = train_ds.prefetch(AUTOTUNE)
  train_ds = map(lambda data: (data[0]._numpy(), data[1]), train_ds)

  if xla_bridge.get_backend().platform == "gpu":
    train_ds = flax.jax_utils.prefetch_to_device(train_ds, 2)
  return train_ds, batch_dims

def get_image_size_from_dataset(dataset):
  if dataset == "cifar10":
    return [32, 32, 3]
  else:
    raise NotImplementedError

if __name__=="__main__":
  sample = jnp.zeros((16, 32, 32, 3))
  # save_images(sample, 0, "sampling")

