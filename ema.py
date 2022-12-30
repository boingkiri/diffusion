import flax
import flax.linen as nn
import jax

import copy

class EMA():
    def __init__(
        self, 
        ema_params, 
        beta=0.999, 
        update_every=10,
        update_after_step=100,
        power=2/3):
        self.ema_params = copy.deepcopy(ema_params)
        self.beta = beta
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.power = power
    
    def ema_update(self, params, step):
        if step <= self.update_after_step:
            return 
        if step % self.update_every != 0:
            return
        current_decay = self.get_current_decay(step)
        # current_decay = self.beta
        # breakpoint()
        # diff = jax.tree_map(lambda x, y: (1 - current_decay) * (x - y), self.ema_params, params)
        # self.ema_params = jax.tree_map(lambda x, y: x - y, self.ema_params, diff)
        # self.ema_params = jax.tree_map(
        #     lambda x, y: self.beta * x + (1 - self.beta) * y,
        #     self.ema_params, params)
        self.ema_params = jax.tree_map(
            lambda x, y: current_decay * x + (1 - current_decay) * y,
            self.ema_params, params)
        
        return current_decay
        # return self.ema_params

    
    def _clamp(self, value, min_value=None, max_value=None):
        # assert min_value is not None or max_value is not None
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value

    def get_current_decay(self, step):
        effective_step = self._clamp(step - self.update_after_step - 1, min_value=0.)
        if effective_step <= 0:
            return 0.
        value = 1 - (1 + effective_step) ** -self.power
        return self._clamp(value, 0.0, self.beta)

    def get_ema_params(self):
        return self.ema_params

if __name__=="__main__":
    count = 0
    ema_obj = EMA(None)
    while True:
        print(count)
        print(ema_obj.get_current_decay(count))
        count += 1
    