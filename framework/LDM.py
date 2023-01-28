from framework.DDPM.ddpm import DDPM
from framework.autoencoder.autoencoder import AutoEncoder

import jax

from framework.default_diffusion import DefaultModel 

class LDM(DefaultModel):
    def __init__(self, config, rand_key, fs_obj):
        super().__init__(config, rand_key)
        self.framework_config = config['framework']
        self.random_key = rand_key
        self.first_stage_model = None
        self.diffusion_model = None
        self.f_scale = len(config['model']['autoencoder']['ch_mults'])

        self.fs_obj = fs_obj

        ae_key, self.random_key = jax.random.split(self.random_key, 2)
        self.first_stage_model = AutoEncoder(config, ae_key, fs_obj)

        if self.get_train_order() == 2:
            ddpm_key, self.random_key = jax.random.split(self.random_key, 2)
            self.diffusion_model = DDPM(config, ddpm_key, fs_obj)
        
        # def sample_fn(num_img, img_size=(32, 32, 3)):
        #     sample = self.diffusion_model.sampling(num_img, img_size)
        #     sample = self.first_stage_model.decoder_forward(sample)
        #     return sample
        
        # self.sampling_jit = jax.jit(sample_fn)
        # self.sampling = sampl
        
    def diffusion_sampling(self, num_img, img_size=(32, 32, 3)):
        diffusion_img_size = (img_size[0] // self.f_scale, img_size[1] // self.f_scale, img_size[2])
        sample = self.diffusion_model.sampling(num_img, diffusion_img_size)
        sample = self.first_stage_model.decoder_forward(sample)
        return sample

    def get_model_state(self):
        if self.get_train_order() == 1:
            return self.first_stage_model.get_model_state()
        elif self.get_train_order == 2:
            return self.diffusion_model.get_model_state()
        else:
            NotImplementedError("LDM has only 2 stages.")
    
    def get_train_order(self):
        return self.framework_config['train_idx']

    def fit(self, x0, cond=None, step=0):
        if self.get_train_order() == 1:
            # GAN will be used
            log_dict = self.first_stage_model.fit(x0, cond, step)
            return log_dict
        elif self.get_train_order() == 2:
            z = self.first_stage_model.encoder_forward(x0)
            loss = self.diffusion_model.fit(z, cond)
            return loss
    
    def sampling(self, num_image, img_size=(32, 32, 3), original_data=None):
        # assert self.get_train_order() == 2
        if self.get_train_order() == 1:
            assert original_data is not None
            sample = self.first_stage_model.reconstruction(original_data)
        elif self.get_train_order() == 2:
            # sample = self.sampling_jit(num_image, img_size)
            sample = self.diffusion_sampling(num_image, img_size)
        else:
            NotImplementedError("Train order should have only 1 or 2 for its value.")
        return sample
    

