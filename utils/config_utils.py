import yaml
import os

from omegaconf import DictConfig, OmegaConf 
import hydra




class ConfigContainer():
    # def __init__(self, config_dir: str) -> None:
    def __init__(self) -> None:
        """
        config (str): Path of config file. 
        
        """
        # split_list = config_dir.split("/")
        # config_dir_path = "/".join(split_list[:-1])
        # config_file = split_list[-1]
        
        # @hydra.main(version_base=None, config_path="configs", config_name="config")
        @hydra.main(version_base=None)
        def get_configuration_from_yaml(cfg: DictConfig):
            print(OmegaConf.to_yaml(cfg))
            # return cfg
        # config_ttt = get_configuration_from_yaml()
        get_configuration_from_yaml()
        breakpoint()
    
    
    
    ################ Experiment Config ################
    def get_exp_config(self):
        return self.config['exp']

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
    
    def get_autoencoder_prefix(self):
        return self.get_exp_config()['autoencoder_prefix']
    
    def get_discriminator_prefix(self):
        return self.get_exp_config()['discriminator_prefix']

    def get_diffusion_prefix(self):
        return self.get_exp_config()['diffusion_prefix']
    
    def get_state_prefix(self, model_type):
        if model_type in ["ddpm", 'diffusion']:
            prefix = self.get_diffusion_prefix()
        elif model_type == "autoencoder":
            prefix = self.get_autoencoder_prefix()
        elif model_type == "discriminator":
            prefix = self.get_discriminator_prefix()
        return prefix
    
    def set_sampling_dir(self, sampling_dir):
        self.config['exp']['sampling_dir'] = sampling_dir


    ################ Framework Config ################
    def get_framework_config(self):
        return self.config['framework']
    
    def get_diffusion_framework_config(self):
        return self.config['framework']['diffusion']

    def get_autoencoder_framework_config(self):
        return self.config['framework']['autoencoder']
    
    def get_do_fid_during_training(self):
        return self.config['framework']['fid_during_training']

    ################ Model Config ################
    def get_model_config(self):
        return self.config['model']
    
    def get_autoencoder_model_config(self):
        return self.get_model_config()['autoencoder']
    
    def get_discriminator_model_config(self):
        return self.get_model_config()['disriminator']
    
    def get_diffusion_model_config(self):
        return self.get_model_config()['diffusion']

    def get_ch_mults(self):
        return self.get_autoencoder_model_config()['ch_mults']
    
    def get_z_dim(self):
        return self.get_autoencoder_model_config()['embed_dim']

    ################ Sampling Config ################
    def get_sampling_config(self):
        return self.config['samplig']

    def get_sample_batch_size(self):
        return self.get_sampling_config()['batch_size']

    
    ################ Miscellaneous Config ################
    def get_dataset_name(self):
        return self.config['dataset']
    
    def get_config_dict(self):
        return self.config
    
    def get_random_seed(self):
        return self.config['rand_seed']

    def get_ema_config(self):
        return self.config['ema']

    def write_yaml_file(self, file_path: str):
        with open(file_path, 'w') as f:
            yaml.dump(self.config, f)

