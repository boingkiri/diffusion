import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

import copy

from flax.training.train_state import TrainState

class EMA():
    def __init__(
        self, 
        # ema_params, 
        beta=0.999, 
        update_every=1,
        update_after_step=1,
        power=2/3,
        ema_rampup_ratio= None,
        ema_halflife_number= None
        ):

        # self.ema_params = copy.deepcopy(ema_params)
        self.beta = beta
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.power = power

        # For EMA
        self.ema_rampup_ratio = ema_rampup_ratio
        self.ema_halflife_number = ema_halflife_number

        def ema_update_pmap_fn(state):
            step = state.step
            current_decay = jax.lax.cond(self.ema_halflife_number is not None, 
                                         self.get_current_decay_edm, 
                                         self.get_current_decay_ddpm,
                                         step)
            ema_updated_params = jax.tree_map(
                lambda x, y: current_decay * x + (1 - current_decay) * y,
                state.params_ema, state.params)
            state = state.replace(params_ema = ema_updated_params)
            return state
        
        # self.ema_update_pmap = jax.pmap(ema_update_pmap_fn, in_axes=(0, None))
        self.ema_update_pmap = jax.jit(ema_update_pmap_fn)

    def ema_update(self, state: TrainState):
        new_state = self.ema_update_pmap(state)
        # self.ema_params = ema_updated_state
        return new_state
    
    def _clamp(self, value, min_value=None, max_value=None):
        # assert min_value is not None or max_value is not None
        if min_value is not None:
            value = jax.numpy.where(min_value > value, min_value, value)
        if max_value is not None:
            value = jax.numpy.where(max_value > value, value, max_value)
        return value

    def get_current_decay_ddpm(self, step):
        effective_step = self._clamp(step - self.update_after_step - 1, min_value=0.)
        # value = 1 - (1 + effective_step) ** -self.power
        # value = self._clamp(value, 0.0, self.beta)
        result_value = jax.numpy.where(effective_step <= 0, 0, self.beta)
        return result_value
    
    def get_current_decay_edm(self, step):
        effective_step = self._clamp(step - self.update_after_step - 1, min_value=0.)
        value = self.beta ** self.get_power(step)
        # value = self._clamp(value, 0.0, self.beta)
        result_value = jax.numpy.where(effective_step <= 0, 0, value)
        return result_value
    
    def get_power(self, step):
        if self.ema_rampup_ratio is not None and self.ema_halflife_number is not None:
            batch_size = 512
            ema_halflife_number = jnp.minimum(self.ema_halflife_number, step * batch_size * self.ema_rampup_ratio) # Naive
            return batch_size / jnp.maximum(ema_halflife_number, 1e-8)
        else:
            return -self.power

    # def get_ema_params(self):
    #     return self.ema_params

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
    ema_obj = EMA(**sample_dict)
    while True:
        print(count)
        print(ema_obj.get_current_decay(count))
        print(ema_obj.get_power(count))
        time.sleep(0.1)
        count += 1
    