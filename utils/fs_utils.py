
import os
import shutil
import yaml

from . import common_utils, jax_utils
from omegaconf import DictConfig, OmegaConf

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

class FSUtils():
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def verify_and_create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"{dir_path} created.")

    def verify_and_create_workspace(self):
        # Creating current exp dir
        current_exp_dir = self.config.exp.current_exp_dir
        self.verify_and_create_dir(current_exp_dir)
        
        # Creating config file
        config_filepath = os.path.join(current_exp_dir, 'config.yaml')
        with open(config_filepath, 'w') as f:
            yaml.dump(OmegaConf.to_container(self.config, resolve=True), f)
        
        # Copying python file to current exp dir
        python_filepath = os.path.join(current_exp_dir, 'python_files')
        workspace_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        for walking_path in os.walk(workspace_path):
            files = walking_path[2]
            walking_path = walking_path[0]
            walking_rel_path = os.path.relpath(walking_path, workspace_path)
            saving_filepath = os.path.join(python_filepath, walking_rel_path)
            if self.config.exp.exp_dir in walking_rel_path:
                continue
            elif os.path.isdir(walking_path) and not os.path.exists(saving_filepath):
                os.makedirs(saving_filepath)
            for file in files:
                if ".py" in file:
                    shutil.copy(os.path.join(walking_path, file), saving_filepath)
        
        # Creating checkpoint dir
        checkpoint_dir = self.config.exp.checkpoint_dir
        self.verify_and_create_dir(checkpoint_dir)
        
        # Creating best checkpoint dir
        best_checkpoint_dir = self.config.exp.best_dir
        self.verify_and_create_dir(best_checkpoint_dir)
        
        # Creating sampling dir
        sampling_dir = self.config.exp.sampling_dir
        self.verify_and_create_dir(sampling_dir)
        
        # Creating in_process dir
        in_process_dir = self.config.exp.in_process_dir
        self.verify_and_create_dir(in_process_dir)
        
        # Creating dataset dir
        dataset_name = self.config.dataset.name
        if not os.path.exists(dataset_name):
            print("Creating dataset dir")
            os.makedirs(dataset_name)
            if dataset_name == "cifar10":
                import tarfile
                common_utils.download("http://pjreddie.com/media/files/cifar.tgz", dataset_name)
                filepath = os.path.join(dataset_name, "cifar.tgz")
                file = tarfile.open(filepath)
                file.extractall(dataset_name)
                file.close()
                os.remove(filepath)
                train_dir_path = os.path.join(dataset_name, "cifar", "train")
                dest_dir_path = os.path.join(dataset_name, "train")
                os.rename(train_dir_path, dest_dir_path)

    def get_start_step_from_checkpoint(self, model_type):
        if model_type == "diffusion":
            checkpoint_format = self.config.exp.diffusion_prefix
        elif model_type == "autoencoder":
            checkpoint_format = self.config.exp.autoencoder_prefix
        elif model_type == "discriminator": # idk this is useful though.
            checkpoint_format = self.config.exp.discriminator_prefix
        checkpoint_dir = self.config.exp.checkpoint_dir
        max_num = 0
        for content in os.listdir(checkpoint_dir):
            if checkpoint_format in content:
                _, num = content.split(checkpoint_format)
                num = int(num)
                if num > max_num:
                    max_num = num
        return max_num
    
    def save_comparison(self, images, steps, savepath):
        # Make in process dir first
        self.verify_and_create_dir(savepath)

        images = common_utils.unnormalize_minus_one_to_one(images)
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
        return save_filename
    
    def save_images_to_dir(self, images, save_path_dir=None, starting_pos=0):
        current_sampling = 0
        if save_path_dir is None:
            save_path_dir = self.config.exp.sampling_dir
        images = common_utils.unnormalize_minus_one_to_one(images)
        images = np.clip(images, 0, 1)
        images = images * 255
        images = np.array(images).astype(np.uint8)
        for image in images:
            im = Image.fromarray(image)
            sample_path = os.path.join(save_path_dir, f"{starting_pos + current_sampling}.png")
            im.save(sample_path)
            current_sampling += 1
        return current_sampling

    def get_state_prefix(self, model_type):
        if model_type == 'diffusion':
            prefix = self.config.exp.diffusion_prefix
        elif model_type == "autoencoder":
            prefix = self.config.exp.autoencoder_prefix
        elif model_type == "discriminator":
            prefix = self.config.exp.discriminator_prefix
        return prefix

    def load_model_state(self, model_type, state, checkpoint_dir=None):
        prefix = self.get_state_prefix(model_type)
        if checkpoint_dir is None:
            checkpoint_dir = self.config.exp.checkpoint_dir
        state = jax_utils.load_state_from_checkpoint_dir(checkpoint_dir, state, None, prefix)
        return state
    
    def get_best_fid(self):
        best_fid = None
        in_process_dir = self.config.exp.in_process_dir
        fid_log_file = os.path.join(in_process_dir, "fid_log.txt")

        if not os.path.exists(fid_log_file):
            return float('inf')

        with open(fid_log_file, 'r') as f:
            txt = f.read()
            logs = txt.split('\n')
            for log in logs:
                if len(log) == 0:
                    continue
                frag = log.split(' ')
                value = float(frag[-1])
                if best_fid is None:
                    best_fid = value
                elif best_fid >= value:
                    best_fid = value
        return best_fid

