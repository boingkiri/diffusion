from DDPM.ddpm import DDPM
from autoencoder.autoencoder import AutoEncoder

from utils import jax_utils

import jax
import jax.numpy as jnp

from framework.default_diffusion import DefaultModel 

class LDM(DefaultModel):
    def __init__(self, config, rand_key, fs_obj):
        super().__init__()
        self.framework_config = config['framework']
        self.random_key = rand_key
        self.first_stage_model = None
        self.diffusion_model = None

        ae_key, self.random_key = self.random_key.split(2)

        self.first_stage_model = AutoEncoder(config, ae_key)

        if self.get_train_order() == 2:
            ddpm_key, self.random_key = self.random_key.split(2)
            self.diffusion_model = DDPM(config, ddpm_key)
        
        def sample_fn(num_img, img_size=(32, 32, 3)):
            sample = self.diffusion_model.sampling(num_img, img_size)
            sample = self.first_stage_model.decoder_forward(sample)

            return sample
        
        self.sampling_jit = jax.jit(sample_fn)
        
   
    def get_model_state(self):
        if self.get_train_order() == 1:
            return self.first_stage_model_state
        elif self.get_train_order == 2:
            return self.diffusion_model_state
        else:
            NotImplementedError("LDM has only 2 stages.")
    
    def get_train_order(self):
        return self.framework_config['train_order']

    def fit(self, x0, cond=None):
        if self.get_train_order() == 1:
            # GAN will be used
            g_log, d_log = self.first_stage_model.fit(x0, cond)
            return g_log, d_log
        elif self.get_train_order() == 2:
            # self.
            z = self.first_stage_model.encoder_forward(x0)
            loss = self.diffusion_model.fit(z, cond)
            return loss
    
    def sampling(self, num_image, img_size=(32, 32, 3)):
        # latent_sampling_tuple = (num_image, *img_size)
        # sampling_key, self.rand_key = jax.random.split(self.rand_key, 2)
        # latent_sample = jax.random.normal(sampling_key, latent_sampling_tuple)

        # for t in reversed(range(self.n_timestep)):
        #     normal_key, dropout_key, self.rand_key = jax.random.split(self.rand_key, 3)
        #     latent_sample = self.p_sample_jit(self.model_state.params, latent_sample, t, normal_key, dropout_key)
        assert self.get_train_order() == 2
        sample = self.sampling_jit(num_image, img_size)
        return sample
    


