import os
import yaml
import os
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

def load_dataset_from_tfds(dataset_name="cifar10", batch_size=128, pmap=False):
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
  train_ds, _ = ds['train'], ds['test']

  if pmap:
    batch_size = batch_size // jax.process_count()

  augmented_train_ds = (
    train_ds
    .shuffle(1000)
    .repeat()
    .map(augmentation, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
  )

  # return augmented_train_ds
  if pmap:
    def reshape(xs):
      local_device_count = jax.local_device_count()
      def _reshape(x):
          x = x.numpy()
          return x.reshape((local_device_count, -1) + x.shape[1:])
      return jax.tree_map(_reshape, xs)
      # return jax.tree_map(lambda x: x.numpy(), xs)
    
  else:
    def reshape(xs):
      return jax.tree_map(lambda x: x.numpy(), xs)
  it = map(reshape, augmented_train_ds)
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

