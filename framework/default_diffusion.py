import jax
import jax.numpy as jnp
import flax

from abc import *
from omegaconf import DictConfig
from functools import partial

from model.modelContainer import ModelContainer
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog

class DefaultModel():
    def __init__(self, 
                config: DictConfig, 
                rand_key: jax.random.PRNGKeyArray, 
                fs_obj: FSUtils, 
                wandblog: WandBLog) -> None:

        # Set various objects
        self.config = config
        self.fs_obj = fs_obj
        self.wandblog = wandblog

        # Create model container for adapting various models
        self.model_container = ModelContainer(config, rand_key, train_state_template=None)
        self.model_keys = self.model_container.model_keys

        self.update_fn = jax.pmap(partial(jax.lax.scan, jax.jit(self.manual_update_fn)), axis_name="batch", donate_argnums=1)

    def manual_update_fn(self, carry_state: dict, data):
        """
        raw update function.
        The naive update operations are implemented in `manual_update_fn`. 
        If you expect more complex behavior for update, you should override this function.
        Args:
            carry_state (dict): 
                carry state. The ingredients for update should be included in this dictionary.
                Two of the elements are mandatory: `train_state` and `rng`.
            data (dict): data for training
        Returns:
            new_carry_state (dict): new carry state
            loss_dict (dict): loss dictionary
        """
        state: dict = carry_state['train_state']
        rng: jax.random.PRNGKeyArray = carry_state['rng']

        (state, rng) = carry_state
        rng, new_rng = jax.random.split(rng)
        params_dict = jax.tree_util.tree_map(lambda x: x.params, state)
        (_, loss_dict), grad = jax.value_and_grad(self.loss_fn, has_aux=True)(params_dict, data, rng)

        grad = jax.lax.pmean(grad)
        new_state = jax.tree_util.tree_map(lambda state, grad: state.apply_gradients(grads=grad), state, grad)
        for loss_key in loss_dict:
            loss_dict[loss_key] = jax.lax.pmean(loss_dict[loss_key])
        
        new_state = self.ema_obj.ema_update(new_state)
        new_carry_state = (new_state, new_rng)
        return new_carry_state, loss_dict

    def update(self, carry_state: dict, x0):
        pmap_carry_state = {}
        for key in carry_state:
            if key == "rng":
                pmap_carry_state[key] = jax.random.split(carry_state[key])
            else:
                pmap_carry_state[key] = flax.jax_utils.replicate(carry_state[key])

        return self.update_fn(pmap_carry_state, x0)
    
    def ema_update(self, model_states):  ## TODO: should be implemented in model container part.
        pass
    

    def fit(self, x0, rng: jax.random.PRNGKeyArray, cond=None, step=0):
        model_states = self.model_container.model_states

        # Construct carry state
        carry_state = {}
        carry_state['train_state'] = model_states
        carry_state['rng'] = rng

        new_carry, loss_dict_stack = self.update((model_states, rng), x0)
        (_, new_states) = new_carry

        loss_dict = flax.jax_utils.unreplicate(loss_dict_stack)
        for loss_key in loss_dict:
            loss_dict[loss_key] = jnp.mean(loss_dict[loss_key])

        self.model_container.model_states = new_states

        return_dict = {}
        return_dict.update(loss_dict)
        self.wandblog.update_log(return_dict)
        return return_dict
    
    def loss_fn(self, params, x0, rng):
        raise NotImplementedError
    
    def sampling(self, num_image, rng_key: jax.random.PRNGKeyArray, img_size=None):
        raise NotImplementedError

    def get_model_state(self) -> dict:
        return self.model_container.model_states
        
