# if __name__ == "__main__":
#     import sys
#     sys.path.append("..")

import jax.random as random
import jax.numpy as jnp
import jax

from utils.fid import fid, inception
from utils import common_utils, fs_utils, fid_utils

config = common_utils.get_config_from_yaml("config.yml")
fid_obj = fid_utils.FIDUtils(config)
image_size = common_utils.get_image_size_from_dataset(config['dataset'])
tmp_dir = fid_obj.get_tmp_dir()
fid_obj.calculate_fid(tmp_dir)
