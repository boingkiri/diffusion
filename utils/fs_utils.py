
import os
import yaml

from . import common_utils, jax_utils

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

class FSUtils():
    def __init__(self, config) -> None:
        self.__config = config
    
    def get_exp_config(self):
        return self.__config['exp']

    def get_current_exp_dir(self):
        exp_config = self.get_exp_config()
        current_exp_dir = os.path.join(exp_config['exp_dir'], exp_config['exp_name'])
        return current_exp_dir

    def get_checkpoint_dir(self):
        exp_config= self.get_exp_config()
        current_exp_dir = self.get_current_exp_dir()
        current_checkpoint_dir = os.path.join(current_exp_dir, exp_config['checkpoint_dir'])
        return current_checkpoint_dir

    def get_best_checkpoint_dir(self):
        checkpoint_dir = self.get_checkpoint_dir()
        best_checkpoint_dir = os.path.join(checkpoint_dir, 'best')
        return best_checkpoint_dir

    def get_sampling_dir(self):
        exp_config = self.get_exp_config()
        current_exp_dir = self.get_current_exp_dir()
        sampling_dir = os.path.join(current_exp_dir, exp_config['sampling_dir'])
        return sampling_dir

    def get_in_process_dir(self):
        exp_config= self.get_exp_config()
        current_exp_dir = self.get_current_exp_dir()
        in_process_dir = os.path.join(current_exp_dir, exp_config['in_process_dir'])
        return in_process_dir
    
    def get_checkpoint_prefix(self):
        exp_config= self.get_exp_config()
        return exp_config['checkpoint_prefix']

    def get_dataset_name(self):
        return self.__config['dataset']

    def verifying_or_create_workspace(self):
        exp_config = self.get_exp_config()
        current_exp_dir = self.get_current_exp_dir()

        # Creating current exp dir
        if not os.path.exists(current_exp_dir):
            os.makedirs(current_exp_dir)
            print("Creating experiment dir")
        
        # Creating config file
        config_filepath = os.path.join(current_exp_dir, 'config.yml')
        with open(config_filepath, 'w') as f:
            yaml.dump(self.__config, f)
        
        # Creating checkpoint dir
        checkpoint_dir = os.path.join(current_exp_dir, exp_config['checkpoint_dir'])
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print("Creating checkpoint dir")
        
        # Creating best checkpoint dir
        best_checkpoint_dir = os.path.join(checkpoint_dir, 'best')
        if not os.path.exists(best_checkpoint_dir):
            os.makedirs(best_checkpoint_dir)
            print("Creating best checkpoint dir")
        
        # Creating sampling dir
        sampling_dir = os.path.join(current_exp_dir, exp_config['sampling_dir'])
        if not os.path.exists(sampling_dir):
            os.makedirs(sampling_dir)
            print("Creating sampling dir")
        
        # Creating in_process dir
        in_process_dir = os.path.join(current_exp_dir, exp_config['in_process_dir'])
        if not os.path.exists(in_process_dir):
            os.makedirs(in_process_dir)
            print("Creating in process dir")
        
        # Creating dataset dir
        dataset_name = self.get_dataset_name()
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

    def get_start_step_from_checkpoint(self, idx=0):
        # checkpoint_format = "checkpoint_"
        checkpoint_format = self.get_checkpoint_prefix()[idx]
        checkpoint_dir = self.get_checkpoint_dir()
        max_num = 0
        for content in os.listdir(checkpoint_dir):
            if checkpoint_format in content:
                _, num = content.split(checkpoint_format)
                num = int(num)
                if num > max_num:
                    max_num = num
        return max_num + 1
    
    def save_comparison(self, images, steps, savepath):
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
    
    def save_images_to_dir(self, images, save_path_dir=None, starting_pos=0):
        current_sampling = 0

        if save_path_dir is None:
            save_path_dir = self.get_sampling_dir()

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
        if model_type == "ddpm":
            prefix = "ddpm_"
        elif model_type in ["autoencoder", "ae"]:
            prefix = "autoencoder_"
        elif model_type == "diffusion":
            prefix = "diffusion_"
        return prefix
    
    # def get_state_step(self, model_type, checkpoint_path=None):
    #     if checkpoint_path is None:
    #         checkpoint_path = self.get_checkpoint_dir()
        


    def load_model_state(self, model_type, state):
        prefix = self.get_state_prefix(model_type)
        checkpoint_dir = self.get_checkpoint_dir()
        # step = self.get_state_step(model_type)
        state = jax_utils.load_state_from_checkpoint_dir(checkpoint_dir, state, None, prefix)
        return state
    
    def get_best_fid(self):
        best_fid = None
        in_process_dir = self.get_in_process_dir()
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

