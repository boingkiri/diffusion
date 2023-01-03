from tqdm import tqdm

import torch
import jax
import jax.numpy as jnp

import numpy as np

from utils import jax_utils, fs_utils, common_utils
from sampling import sampling

def train(config, state, ddpm, start_step, ema_obj, rng):
    train_config = config['train']
    dataset = config['dataset']

    dataloader = common_utils.load_dataset_from_tfds(dataset, train_config['batch_size'])
    total_step = train_config['total_step']
    data_bar = tqdm(dataloader, total=total_step - start_step)
    
    in_process_dir = fs_utils.get_in_process_dir(config)
    checkpoint_dir = fs_utils.get_checkpoint_dir(config)
    image_size = common_utils.get_image_size_from_dataset(dataset)

    step = start_step + 1
    ema_decay = 0
    for i, (x, _) in enumerate(data_bar):
        x = jax.device_put(x.numpy())
        loss, state = ddpm.learning_from(state, x)
        
        loss_ema = loss.item()
        current_ema_decay = ema_obj.ema_update(state.params, step)
        if current_ema_decay is not None:
            ema_decay = current_ema_decay
        data_bar.set_description(f"Step: {step} loss: {loss_ema:.4f} EMA decay: {ema_decay:.4f}")

        if step % 1000 == 0:
            x_t, rng = sampling(ddpm, ema_obj.get_ema_params(), 8, image_size, rng)
            xset = jnp.concatenate([x_t[:8], x[:8]], axis=0)
            xset = torch.from_numpy(np.array(xset))
            common_utils.save_images(xset, step, in_process_dir)

        if step % 10000 == 0:
            state = state.replace(params_ema = ema_obj.get_ema_params())
            jax_utils.save_train_state(state, checkpoint_dir, step)
        
        if step >= total_step:
            break
        step += 1


