import sys
sys.path.append("..")

from clu import parameter_overview

import jax
import flax
from flax import linen
from flax.training.checkpoints import restore_checkpoint

from utils import common_utils


jax.config.update('jax_platform_name', 'cpu')


# Write checkpoint directory for the model you want to inspect
checkpoint_dir = "../experiments/0828_verification_denoised/checkpoints/diffusion_700000"
state = restore_checkpoint(checkpoint_dir, target=None)

print(parameter_overview.get_parameter_overview(state['params']))