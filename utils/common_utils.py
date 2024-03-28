import os
import requests
import pickle

import numpy as np
import flax
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge

import tensorflow as tf
import tensorflow_datasets as tfds

# from . import jax_utils 

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

def load_dataset_from_pickled_file(dataset_dir, data_size):
  train_list = os.listdir(dataset_dir)
  image = [] 
  labels = []
  img_size2 = data_size[0] * data_size[1]
  for filename in train_list:
    filename = os.path.join(dataset_dir, filename)
    with open(filename, "rb") as f:
      data = pickle.load(f)
      x = data["data"]
      x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
      x = x.reshape((x.shape[0], data_size[0], data_size[1], 3))
      image.append(x)
      labels.append(data["labels"])
  
  image = np.concatenate(image)
  labels = np.concatenate(labels).astype(int) - 1
  for image_elem, labels_elem in zip(image, labels):
    yield image_elem, labels_elem


def load_dataset_from_tfds(config, dataset_name=None, batch_size=None, n_jitted_steps=None, x_flip=True, shuffle=True, for_pae=False):

  dataset_name = config["dataset"]["name"] if dataset_name is None else dataset_name
  batch_size = config["framework"]["diffusion"]["train"]["batch_size_per_rounds"] if batch_size is None else batch_size
  if config.get("distributed_training", False) and not for_pae:
    batch_size = batch_size // (jax.device_count() // jax.local_device_count())
    print_format = f'Total global batch size: {config["framework"]["diffusion"]["train"]["total_batch_size"]}\n'
    print_format += f'Global batch size per round: {config["framework"]["diffusion"]["train"]["batch_size_per_rounds"]}\n'
    print_format += f'Local batch size: {batch_size}'
    print(print_format)

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
    train_ds, _ = ds['train'], ds['test']
  elif dataset_name == "imagenet_64":
    generator = load_dataset_from_pickled_file(config["dataset"]["dataset_path"], data_size=config["dataset"]["data_size"])
    ds = tf.data.Dataset.from_generator(lambda: map(tuple, generator), (tf.uint8, tf.uint32), ((64, 64, 3), ()))
    train_ds = ds

  global_device_count = jax.device_count()
  device_count = jax.local_device_count()
  batch_dims= [device_count, n_jitted_steps, batch_size // device_count] 
  # batch_dims = [global_device_count // device_count, device_count, n_jitted_steps, batch_size // global_device_count]

  # global_mesh, pspec = jax_utils.create_environment_sharding()


  if shuffle:
    train_ds = train_ds.shuffle(1000)
  train_ds = train_ds.repeat()
  train_ds = train_ds.map(augmentation, num_parallel_calls=AUTOTUNE)
  for dim in reversed(batch_dims):
    train_ds = train_ds.batch(dim)
  augmented_train_ds = train_ds.prefetch(AUTOTUNE)
  # it = tfds.as_numpy(augmented_train_ds)
  it = map(lambda data: jax.tree_map(lambda x: x._numpy(), data), augmented_train_ds)
  it = map(lambda data: jax.tree_map(lambda x: jnp.asarray(x), data), it)
  # it = map(lambda data: jax.tree_map(lambda x: jax.device_put(x, sharding), data), it)
  # it = map(lambda data: jax.tree_map(lambda x: jax.experimental.multihost_utils.host_local_array_to_global_array(x, global_mesh, pspec), data), it)
  if xla_bridge.get_backend().platform == "gpu":
    it = flax.jax_utils.prefetch_to_device(it, 2)

  return it


def get_image_size_from_dataset(dataset):
  if dataset == "cifar10":
    return [32, 32, 3]
  else:
    raise NotImplementedError

if __name__=="__main__":
  # sample = jnp.zeros((16, 32, 32, 3))
  # save_images(sample, 0, "sampling")
  from hydra import initialize, compose
  config_path = "../configs"
  config_yaml = "config_imagenet"
  with initialize(version_base=None, config_path=config_path) as cfg:
    config = compose(config_name=config_yaml)
    config["dataset"]["dataset_path"] = "../imagenet_64"
    it = load_dataset_from_tfds(
       config, dataset_name=config["dataset"]["name"], 
       batch_size=config["framework"]["diffusion"]["train"]["batch_size_per_rounds"], 
       n_jitted_steps=config.n_jitted_steps, x_flip=True, shuffle=True, 
       for_pae=False)
    for x, y in it:
      x = (x[0, 0, 0] + 1) / 2 * 255
      x = np.asarray(x.astype(np.uint8))
      import PIL.Image as Image
      im = Image.fromarray(x)
      print(y[0, 0, 0])
      im.save("sample.png")
      breakpoint()
