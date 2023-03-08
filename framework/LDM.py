# from framework.DDPM.ddpm import DDPM
from framework.diffusion.ddpm_framework import DDPMFramework
from framework.autoencoder.autoencoder import AutoEncoder

from utils.log_utils import WandBLog
from utils.fs_utils import FSUtils

import jax

from framework.default_diffusion import DefaultModel 

from omegaconf import DictConfig

class LDM(DefaultModel):
    def __init__(self, config, rand_key, fs_obj: FSUtils, wandblog: WandBLog):
        super().__init__(config, rand_key)
        self.framework_config = config['framework']
        self.random_key = rand_key
        self.first_stage_model = None
        self.diffusion_model = None
        self.f_scale = len(config['model']['autoencoder']['ch_mults'])
        self.z_dim = config['model']['autoencoder']['embed_dim']

        self.fs_obj = fs_obj
        self.wandblog = wandblog

        ae_key, self.random_key = jax.random.split(self.random_key, 2)
        self.first_stage_model = AutoEncoder(config, ae_key, fs_obj, wandblog)
        if self.get_train_order() == 2:
            ddpm_key, self.random_key = jax.random.split(self.random_key, 2)
            # self.diffusion_model = DDPM(config, ddpm_key, fs_obj, wandblog)
            self.diffusion_model = DDPMFramework(config, ddpm_key, fs_obj, wandblog)
    
    def get_sampling_size(self):
        # Assume we're using CIFAR-10 dataset
        img_size = 32
        if self.get_train_order() == 1:
            return (img_size, img_size, 3)
        elif self.get_train_order() == 2:
            img_size = img_size // self.f_scale
            return (img_size, img_size, self.z_dim)
        
    def diffusion_sampling(self, num_img, original_data=None):
        # if original_data is not None:
        #     original_data_encoding = self.first_stage_model.encoder_forward(original_data)
        # else:
        #     original_data_encoding = None
        original_data_encoding = None
        
        diffusion_img_size = self.get_sampling_size()
        sample = self.diffusion_model.sampling(num_img, diffusion_img_size, original_data_encoding)
        sample = self.first_stage_model.decoder_forward(sample)
        return sample

    def get_model_state(self):
        if self.get_train_order() == 1:
            return self.first_stage_model.get_model_state()
        elif self.get_train_order() == 2:
            return self.diffusion_model.get_model_state()
        else:
            NotImplementedError("LDM has only 2 stages.")
    
    def init_model_state(self, config: DictConfig, model_type, model, rng):
        # return super().init_model_state(config, model_type, model, rng)
        NotImplementedError("LDM do not use 'init_model_state' directly.")
    
    def get_train_order(self):
        return self.framework_config['train_idx']

    def fit(self, x0, cond=None, step=0):
        if self.get_train_order() == 1:
            log_dict = self.first_stage_model.fit(x0, cond, step)
            return log_dict
        elif self.get_train_order() == 2:
            z = self.first_stage_model.encoder_forward(x0)
            loss = self.diffusion_model.fit(z, cond, step)
            return loss
    
    def sampling(self, num_image, original_data=None):
        if self.get_train_order() == 1:
            assert original_data is not None
            sample = self.first_stage_model.reconstruction(original_data)
        elif self.get_train_order() == 2:
            sample = self.diffusion_sampling(num_image, original_data)
        else:
            NotImplementedError("Train order should have only 1 or 2 for its value.")
        return sample
    