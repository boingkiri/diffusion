import jax
import jax.numpy as jnp
import numpy as np

from .fid import inception, fid
from . import fs_utils, common_utils

import functools
import os

class FIDFramework():
    def __init__(self, config) -> None:
        self.rng = jax.random.PRNGKey(42)
        self.config = config
        self.model, self.params, self.apply_fn = self.load_fid_model()
        self.img_size = (299, 299)
    
    def load_fid_model(self):
        model = inception.InceptionV3(pretrained=True)
        params = model.init(self.rng, jnp.ones((1, 256, 256, 3)))
        apply_fn = jax.jit(functools.partial(model.apply, train=False))
        return model, params, apply_fn
    
    def get_tmp_dir(self):
        in_process_dir = fs_utils.get_in_process_dir(self.config)
        tmp_dir = os.path.join(in_process_dir, "tmp")
        return tmp_dir
    
    def precompute_dataset(self, dataset_name):
        if dataset_name == "cifar10" and "stats.npz" not in os.listdir(dataset_name):
            print("Precomputing CIFAR10 statistics")
            dataset_path = os.path.join(dataset_name, "train")
            mu, sigma = fid.compute_statistics(dataset_path, self.params, self.apply_fn, 50, self.img_size)
            statistics_file = os.path.join(dataset_name, "stats")
            np.savez(statistics_file, mu=mu, sigma=sigma)
            return mu, sigma
        print(f"Loading {dataset_name} statistics")
        statistics_file = os.path.join(dataset_name, "stats.npz")
        mu, sigma = fid.compute_statistics(statistics_file, self.params, self.apply_fn, 50, self.img_size)
        return mu, sigma

    def calculate_statistics(self, img_path):
        mu, sigma = fid.compute_statistics(img_path, self.params, self.apply_fn, 50, self.img_size)
        return mu, sigma
    
    def calculate_fid(self, src_img_path, des_img_path=None):
        if des_img_path is None:
            dataset_name = fs_utils.get_dataset_name(self.config)
            dest_mu, dest_sigma = self.precompute_dataset(dataset_name)
        else:
            dest_mu, dest_sigma = self.calculate_statistics(des_img_path)
        src_mu, src_sigma = self.calculate_statistics(src_img_path)
        fid_score = fid.compute_frechet_distance(src_mu, dest_mu, src_sigma, dest_sigma)
        return fid_score
