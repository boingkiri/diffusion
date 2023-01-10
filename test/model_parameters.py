import sys
sys.path.append("..")

from clu import parameter_overview

import jax

from utils import common_utils
jax.config.update('jax_platform_name', 'cpu')

config = common_utils.get_config_from_yaml("../config.yml")

state, ddpm, start_step, ema_obj, rng = common_utils.init_setting(config, jax.random.PRNGKey(42))
print(parameter_overview.get_parameter_overview(state.params))