import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import requests

import flax
import jax
import jax.numpy as jnp

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

def get_dataset_size(dataset_name):
  if dataset_name == "cifar10":
     return (32, 32, 3)
  else:
     NotImplementedError("get_dataset_size: Not implemented.")

def normalize_to_minus_one_to_one(image):
    return image * 2 - 1

def unnormalize_minus_one_to_one(images):
    return (images + 1) * 0.5 

def load_dataset_from_tfds(dataset_name="cifar10", batch_size=128, n_jitted_steps=1, x_flip=True):
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

  ds = tfds.load(dataset_name, as_supervised=True)
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
  it = flax.jax_utils.prefetch_to_device(it, 2)

  return it


def get_image_size_from_dataset(dataset):
  if dataset == "cifar10":
    return [32, 32, 3]
  else:
    raise NotImplementedError

if __name__=="__main__":
  sample = jnp.zeros((16, 32, 32, 3))
  # save_images(sample, 0, "sampling")

