from DDPM.ddpm import DDPM
from LDM import LDM

import jax

from utils.fs_utils import FSUtils
from utils import jax_utils 

class DiffusionFramework():
    """
        This framework contains overall methods for training and sampling

    """
    def __init__(self, config, random_rng) -> None:
        self.type = config['framework']['type']
        self.random_rng = random_rng
        if self.type == 'ddpm':
            ddpm_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.model = DDPM(config, ddpm_rng)
        elif self.type == "LDM":
            ldm_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.model = LDM(config, ldm_rng)
        self.fs_utils = FSUtils(config)
        self.step = self.fs_utils.
    
    def fit(self, x):
        log = self.model.fit(x)
        return log
    
    def save_model(self):
        checkpoint_dir = self.fs_utils.get_checkpoint_dir()
        state_dict = self.model.get_model_state() # Dictionary of state
        
        for key in state_dict:
            state = state_dict[key]
            jax_utils.save_train_state(state, checkpoint_dir,)



    # def get_model_state(self):
    #     state_dict = self.model.get_model_state()
    #     return state_dict
    