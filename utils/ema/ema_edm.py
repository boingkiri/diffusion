import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

# import copy
# from default_ema import DefaultEMA 
from utils.ema.default_ema import DefaultEMA 

from flax.training.train_state import TrainState


class EDMEMA(DefaultEMA):
    def __init__(
        self, 
        beta=0.9999, 
        update_every=1,
        update_after_step=1,
        ema_rampup_ratio= None,
        ema_halflife_number= None,
        ):

        super().__init__(beta, update_every, update_after_step)

        # For EMA
        self.ema_rampup_ratio = ema_rampup_ratio
        self.ema_halflife_number = ema_halflife_number

    def get_current_decay(self, step):
        effective_step = self._clamp(step - self.update_after_step - 1, min_value=0.)
        value = self.beta ** self.get_power(step)
        result_value = jax.numpy.where(effective_step <= 0, 0, value)
        return result_value
    
    def get_power(self, step):
        batch_size = 512
        ema_halflife_number = jnp.minimum(self.ema_halflife_number, step * batch_size * self.ema_rampup_ratio) # Naive
        return batch_size / jnp.maximum(ema_halflife_number, 1e-8)

if __name__=="__main__":
    import time
    count = 0
    sample_dict = {
        "beta": 0.5,
        "update_every": 1,
        "update_after_step": 0,
        "power": 2/3,
        "ema_rampup_ratio": 0.05,
        "ema_halflife_number": 500000
    }
    ema_obj = EDMEMA(**sample_dict)
    while True:
        print(count)
        print(ema_obj.get_current_decay(count))
        print(ema_obj.get_power(count))
        time.sleep(0.1)
        count += 1
    