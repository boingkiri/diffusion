
import os
import yaml

from . import common_utils, jax_utils
from .config_utils import ConfigContainer

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

class FSUtils():
    def __init__(self, config: ConfigContainer) -> None:
        self.config = config

    def verify_and_create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"{dir_path} created.")

    def verify_and_create_workspace(self):
        exp_config = self.config.get_exp_config()
        
        # Creating current exp dir
        current_exp_dir = self.config.get_current_exp_dir()
        self.verify_and_create_dir(current_exp_dir)
        
        # Creating config file
        config_filepath = os.path.join(current_exp_dir, 'config.yml')
        with open(config_filepath, 'w') as f:
            yaml.dump(self.config, f)
        
        # Creating checkpoint dir
        checkpoint_dir = os.path.join(current_exp_dir, exp_config['checkpoint_dir'])
        self.verify_and_create_dir(checkpoint_dir)
        
        # Creating best checkpoint dir
        best_checkpoint_dir = os.path.join(checkpoint_dir, 'best')
        self.verify_and_create_dir(best_checkpoint_dir)
        
        # Creating sampling dir
        sampling_dir = os.path.join(current_exp_dir, exp_config['sampling_dir'])
        self.verify_and_create_dir(sampling_dir)
        
        # Creating in_process dir
        in_process_dir = os.path.join(current_exp_dir, exp_config['in_process_dir'])
        self.verify_and_create_dir(in_process_dir)
        
        # Creating dataset dir
        dataset_name = self.config.get_dataset_name()
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
            checkpoint_format = self.config.get_diffusion_prefix()
        elif model_type == "autoencoder":
            checkpoint_format = self.config.get_autoencoder_prefix()
        elif model_type == "discriminator": # idk this is useful though.
            checkpoint_format = self.config.get_discriminator_prefix()
        checkpoint_dir = self.config.get_checkpoint_dir()
        max_num = 0
        for content in os.listdir(checkpoint_dir):
            if checkpoint_format in content:
                _, num = content.split(checkpoint_format)
                num = int(num)
                if num > max_num:
                    max_num = num
        return max_num + 1
    
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
            save_path_dir = self.config.get_sampling_dir()
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

    def load_model_state(self, model_type, state, checkpoint_dir=None):
        prefix = self.config.get_state_prefix(model_type)
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get_checkpoint_dir()
        state = jax_utils.load_state_from_checkpoint_dir(checkpoint_dir, state, None, prefix)
        return state
    
    def get_best_fid(self):
        best_fid = None
        in_process_dir = self.config.get_in_process_dir()
        fid_log_file = os.path.join(in_process_dir, "fid_log.txt")
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

