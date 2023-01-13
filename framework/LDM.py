from DDPM.ddpm import DDPM
from autoencoder.autoencoder import AutoEncoder

class LDM():
    def __init__(self, config, rand_key):
        self.config = config
        self.framework_config = config['framework']
        self.random_key = rand_key
        if self.framework_config['type'] in ['ldm']:
            self.diffusion_model = AutoEncoder(config)

        if self.framework_config['type'] in ['ldm', 'ddpm']:
            ddpm_key, key = self.random_key.split(2)
            self.random_key = key
            self.diffusion_model = DDPM(ddpm_key, config)

    def fit(self, state, x0):
        
