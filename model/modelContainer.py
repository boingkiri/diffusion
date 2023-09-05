import jax
import jax.numpy as jnp
import flax
import orbax.checkpoint

import inspect

from omegaconf import DictConfig

from utils.fs_utils import FSUtils
from utils.common_utils import load_class_from_config_for_model
from utils.jax_utils import get_learning_rate_schedule, create_optimizer, TrainState

class ModelContainer():
    """
    Container for model and model state.
    The framework should access model and model_state through this container.
    This container is responsible for creating model and model state.
    """
    def __init__(
            self, 
            config: dict, 
            rng_key: jax.random.PRNGKeyArray,
            train_state_template: flax.struct.dataclass = None) -> None:
        """
        Args:
            config (dict): configuration for model
            fs_obj (FSUtils): file system object
        """
        model_configs = config['model']

        self.models: dict = {}
        self.model_states: dict = {}

        # Create train state template
        self.train_state_template = TrainState if train_state_template is None else train_state_template

        # Create model and model state
        self.model_keys = model_configs.keys()
        for key in model_configs:
            model_config = model_configs[key]
            rng_key, create_model_key = jax.random.split(rng_key, 2)
            model, model_state = self.create_model(model_config, create_model_key)
            self.models[key] = model
            self.model_states[key] = model_state

            # Add model to self as attributes
            setattr(self, key, model)
        
        # Load model state
        # checkpoint_dir = self.config.exp.checkpoint_dir
        for model_key in self.models:
            self.model_states = self.load_model_state(model_key, self.model_states) 
        # FIXME: string "diffusion" should be replaced to model_type

    @property
    def model_keys(self):
        return self.model_keys

    @property
    def models(self):
        return self.models

    @property
    def model_states(self):
        return self.model_states

    @model_states.setter
    def model_states(self, model_states_dict):
        self.model_states = model_states_dict


    def model(self):
        assert len(self.models) == 1, "Model Container has more than one model. Please use `models` instead of `model`"
        return self.models[list(self.models.keys())[0]]

    def model_state(self):
        assert len(self.model_states) == 1, "Model Container has more than one model state. Please use `model_states` instead of `model_state`"
        return self.model_states[list(self.model_states.keys())[0]]
    
    def create_model(self, model_config: DictConfig, rng_key: jax.random.PRNGKeyArray):
        # Prepare model class for inflate obj
        model_class = load_class_from_config_for_model(model_config)

        # Construct model config dict
        model_type = model_config.pop("type")
        model_class_name=  model_config.pop("model_class")
        model_config_dict = {**model_config}

        # Create model
        model = model_class(**model_config)
        model_state = self.init_model_state(model, model_config_dict, rng_key)

        return model, model_state

    def load_model_state(self, model_key, state, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.config.exp.checkpoint_dir
        state = self.load_state_from_checkpoint_dir(checkpoint_dir, model_key, state, None)
        return state

    def init_model_state(self, model, config: DictConfig, rng_key: jax.random.PRNGKeyArray):
        rng_key, param_rng, dropout_rng = jax.random.split(rng_key, 3)
        rng_dict = {"params": param_rng, 'dropout': dropout_rng}
        if config["type"] == "ldm" and config["framework"]['train_idx'] == 2:
            f_value = len(config['model']['autoencoder']['ch_mults'])
            z_dim = config['model']['autoencoder']['embed_dim']
            input_format_shape = input_format.shape
            input_format = jnp.ones(
                [input_format_shape[0], 
                input_format_shape[1] // f_value, 
                input_format_shape[2] // f_value, 
                z_dim])
        input_format = jnp.ones([1, *config.dataset.data_size])
        params = model.init(rng_dict, x=input_format, t=jnp.ones([1,]), train=False)['params']

        return self.create_train_state(config, 'diffusion', self.model.apply, params)
    
    def create_train_state(
            self, 
            config: DictConfig, 
            model_type, 
            apply_fn, 
            params, 
            additional_args: dict=None):
        """
        Creates initial 'TrainState'

        Args:
            config (dict): configuration for model. This information is used for creating optimizer
            model_type (str): type of model
            apply_fn (function): function for applying model
            params (dict): parameters of model
            additional_args (dict): 
                additional arguments for train state
                Some arguments are used frequently in legacy code. 
                If we encounter the arguments, we allocate the value automatically. 
        Returns:
            train_state (flax.struct.dataclass): initial train state
        """
        # TODO: Remove `model_type` from argument.
        # I think `model_type` is not that necessary to make optimizer.
        tx = create_optimizer(config, model_type) 

        # FIXME: Value allocation of additional argument of train state is too naive in this implementation.
        create_argument = {
            "apply_fn": apply_fn,
            "params": params,
            "tx": tx,
        }

        default_arguments = inspect.signature(flax.train_state.TrainState).parameters
        modified_arguments = inspect.signature(TrainState).parameters
        additional_args =  [arg for arg in modified_arguments if arg not in default_arguments]

        for arg in additional_args:
            if arg == "params_ema":
                create_argument[arg] = params
            elif arg == "target_model":
                create_argument[arg] = params
            else:
                raise NotImplementedError(f"Additional argument {arg} is not implemented.")

        # Return the training state
        return self.train_state_template.create(**create_argument)


    def get_model_states(self):
        """
        Returns:
            model_state (dict): unreplicated model state 
        """
        return_value = {}
        for key in self.model_states:
            return_value[key] = flax.jax_utils.unreplicate(self.model_states[key])
        return return_value

    def load_state_from_checkpoint_dir(self, model_key, checkpoint_dir, state):
        ocp = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_dir = checkpoint_dir / f"{model_key}"
        state = ocp.restore(checkpoint_dir)
        print(f"Checkpoint {state.step} loaded")
        return state