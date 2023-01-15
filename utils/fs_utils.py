
import os
import yaml

from . import common_utils

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

    def get_start_step_from_checkpoint(self):
        checkpoint_format = "checkpoint_"
        checkpoint_dir = self.get_checkpoint_dir()
        max_num = 0
        for content in os.listdir(checkpoint_dir):
            if checkpoint_format in content:
                _, num = content.split(checkpoint_format)
                num = int(num)
                if num > max_num:
                    max_num = num
        return max_num