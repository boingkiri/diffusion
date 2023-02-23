import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

import copy

class EMA():
    def __init__(
        self, 
        # ema_params, 
        beta=0.999, 
        update_every=10,
        update_after_step=100,
        power=2/3,
        pmap=False):

        # self.ema_params = copy.deepcopy(ema_params)
        self.beta = beta
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.power = power
        self.pmap = pmap
        self.step = -1

        def ema_update_pmap_fn(state):
            current_decay = self.get_current_decay(self.step)
            # ema_updated_params = jax.tree_map(
            #     lambda x, y: current_decay * x + (1 - current_decay) * y,
            #     self.ema_params, state.params)
            ema_updated_params = jax.tree_map(
                lambda x, y: current_decay * x + (1 - current_decay) * y,
                state.params_ema, state.params)
            state = state.replace(params_ema = ema_updated_params)
            return state, current_decay
        
        # self.ema_update_pmap = jax.pmap(ema_update_pmap_fn, in_axes=(0, None))
        self.ema_update_pmap = jax.pmap(ema_update_pmap_fn)
        # self.ema_update_vmap = jax.vmap(ema_update_pmap_fn, in_axes=(0, None))
        # self.ema_update_pmap = jax.jit(ema_update_pmap_fn)

    def ema_update(self, state, step):
        if step <= self.update_after_step:
            return state, step
        if step % self.update_every != 0:
            return state, step
        self.step = step
        new_state, current_decay = self.ema_update_pmap(state)
        # self.ema_params = ema_updated_state
        return new_state, current_decay
    
    def _clamp(self, value, min_value=None, max_value=None):
        # assert min_value is not None or max_value is not None
        if min_value is not None:
            # value = max(min_value, value)
            value = jax.numpy.where(min_value > value, value, min_value)
        if max_value is not None:
            value = jax.numpy.where(max_value > value, max_value, value)
            # value = min(max_value, value)
        return value

    def get_current_decay(self, step):
        effective_step = self._clamp(step - self.update_after_step - 1, min_value=0.)
        value = 1 - (1 + effective_step) ** -self.power
        value = self._clamp(value, 0.0, self.beta)
        result_value = jax.numpy.where(effective_step <= 0, 0, value)
        return result_value

    def get_ema_params(self):
        return self.ema_params

if __name__=="__main__":
    count = 0
    ema_obj = EMA(None)
    while True:
        print(count)
        print(ema_obj.get_current_decay(count))
        count += 1
    