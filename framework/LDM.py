from DDPM.ddpm import DDPM
from autoencoder.autoencoder import AutoEncoder

from utils import jax_utils

class LDM():
    def __init__(self, config, rand_key):
        self.__config = config
        self.framework_config = config['framework']
        self.random_key = rand_key
        self.first_stage_model = None
        self.diffusion_model = None

        ae_key, self.random_key = self.random_key.split(2)

        self.first_stage_model = AutoEncoder(config)
        self.first_stage_model_state = jax_utils.create_train_state(config, self.first_stage_model, ae_key)

        if self.get_train_order() == 2:
            ddpm_key, ddpm_init_key, self.random_key = self.random_key.split(3)
            self.diffusion_model = DDPM(ddpm_key, config)
            self.diffusion_model_state = jax_utils.create_train_state(config, self.diffusion_model, ddpm_init_key)
        
   
    def get_model_state(self):
        if self.get_train_order() == 1:
            return self.first_stage_model_state
        elif self.get_train_order == 2:
            return self.diffusion_model_state
        else:
            NotImplementedError("LDM has only 2 stages.")
    
    def get_first_stage_state(self):
        return self.first_stage_model_state
    
    def get_diffusion_state(self):
        return self.diffusion_model_state
    
    def get_train_order(self):
        return self.framework_config['train_order']

    def fit(self, x0, cond=None):
        if self.get_train_order() == 1:
            # GAN will be used

        elif self.get_train_order() == 2:
            self.
        e_x = self.first_stage_model.encoder_fit(x0)
        z = self.diffusion_model(e_x)

