from typing import Dict, Optional, Type
from tqdm import tqdm

# from torch.utils.data import Dataset, DataLoader

# from torchvision.datasets import CIFAR10
# from torchvision import transforms
# from torchvision.utils import save_image, make_grid

import torch
import jax
import jax.numpy as jnp
import numpy as np

from unet import UNet
from ddpm import DDPM
# from jax_utils import create_train_state, load_dataset_from_tfds
import jax_utils
from ema import EMA

from flax.training import checkpoints

import logging
import os
import pickle

def init_cifar10(
        n_timestep:int = 1000,
        checkpoint_dir:Optional[str]=None,
        checkpoint_step:Optional[int]=None
    ):
    UNet_obj = UNet()
    # state = jax_utils.create_train_state(UNet_obj)
    state = jax_utils.create_train_state(UNet_obj)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    ema_obj = None
    
    # if checkpoint_dir is not None and "checkpoint_" in os.listdir(checkpoint_dir):
    if checkpoint_dir is not None:
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, checkpoint_step)
        print(checkpoint_dir + " Loaded.")
        checkpoint_step += 1
        
    else:
        checkpoint_step = 0
    ddpm = DDPM(UNet_obj, n_timestep)
    return state, ddpm, checkpoint_step, ema_obj


def run_cifar10(
        n_timestep:int = 1000, 
        n_step:int = 100, 
        batch_size:int = 128, 
        checkpoint_dir:Optional[str]=None,
        checkpoint_epoch:Optional[int]=None,
        sampling_dir: str="sampling"
    ):
    # tf.profiler.experimental.server.start(6000)

    # image_size = (3, 32, 32)
    image_size = (32, 32, 3)
    state, ddpm, next_step, ema_obj = init_cifar10(n_timestep, checkpoint_dir, checkpoint_epoch)
    if ema_obj is None:
        ema_obj = EMA(state.params_ema)
    
    dataloader = jax_utils.load_dataset_from_tfds()

    sampling_key = jax.random.PRNGKey(42)

    # logging.info("Start to train CIFAR10")
    print("Start to train CIFAR10")

    if not os.path.exists(sampling_dir):
        os.makedirs(sampling_dir)
    
    save_dir = os.path.join(sampling_dir, "save")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    step = next_step
    data_bar = tqdm(dataloader, total=n_step - step)
    ema_decay = 0
    for i, (x, _) in enumerate(data_bar):
        loss_ema = None
        x = jax.device_put(x.numpy())
        loss, state = ddpm.learning_from(state, x)
        
        # if loss_ema is None:
        #     loss_ema = loss.item()
        # else:
        #     loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
        loss_ema = loss.item()
        current_ema_decay = ema_obj.ema_update(state.params, step)
        if current_ema_decay is not None:
            ema_decay = current_ema_decay
        data_bar.set_description(f"Step: {step} loss: {loss_ema:.4f} EMA decay: {ema_decay:.4f}")

        if step % 1000 == 0:
            # Sampling
            sampling_key, normal_key = jax.random.split(sampling_key)
            # x_t = jax.random.normal(normal_key, (batch_size, *image_size))
            x_t = jax.random.normal(normal_key, (8, *image_size))

            sampling_bar = tqdm(reversed(range(n_timestep)))
            for t in sampling_bar:
                # x_t = ddpm.p_sample(state, x_t, t)
                x_t = ddpm.p_sample(ema_obj.get_ema_params(), x_t, t)
                sampling_bar.set_description(f"Sampling: {t}")
            xset = jnp.concatenate([x_t[:8], x[:8]], axis=0)
            xset = torch.from_numpy(np.array(xset))
            # filename = os.path.join(sampling_dir, f"ddpm_sample_cifar{i}.png")
            # grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            # save_image(grid, filename)
            jax_utils.save_images(xset, i, sampling_dir)

        
        if step % 10000 == 0:
            if checkpoint_dir is None:
                checkpoint_dir = './checkpoints'
            # saved_state = 
            # state.params_ema = ema_obj.get_ema_params()
            # checkpoints.save_checkpoint(checkpoint_dir, state, step)
            # ema_path = os.path.join(checkpoint_dir, f"ema_{step}.pkl")
            # with open(ema_path , 'wb') as f:
            #     pickle.dump(ema_obj, f)
            # print(f"Saving {step} complete.")
            jax_utils.save_train_state(state, ema_obj.get_ema_params(), checkpoint_dir, step)
        step += 1


def sample_cifar10(
    n_timestep:int = 1000, 
    batch_size:int = 256, 
    checkpoint_dir:Optional[str]=None,
    checkpoint_epoch:Optional[int]=None,
    sampling_dir: str="sampling/save",
    num_sampling:int = 50000
):
    # image_size = (3, 32, 32)
    image_size = (32, 32, 3)
    state, ddpm, _, _ = init_cifar10(n_timestep, checkpoint_dir, checkpoint_epoch)
    # Sampling
    sampling_key = jax.random.PRNGKey(42)
    

    current_num_sampling = 0

    while True:
        sampling_key, normal_key = jax.random.split(sampling_key)
        x_t = jax.random.normal(normal_key, (batch_size, *image_size))

        sampling_bar = tqdm(reversed(range(n_timestep)))
        for t in sampling_bar:
            # x_t = ddpm.p_sample(state, x_t, t)
            x_t = ddpm.p_sample(state.params_ema, x_t, t)
            sampling_bar.set_description(f"Sampling: {t}")
        x_t = jnp.clip((x_t + 1) / 2, 0, 1)
        x_t = torch.from_numpy(np.array(x_t))
        for sampled_idx in range(batch_size):
            # real_sample_filename = os.path.join(sampling_dir, f"generated_{current_num_sampling}.png")
            sample = x_t[sampled_idx]
            # save_image(sample, real_sample_filename)
            jax_utils.save_images(sample, current_num_sampling, sampling_dir)
            current_num_sampling += 1
        
        if current_num_sampling >= num_sampling:
            break


if __name__ =="__main__":
    mode = 0
    total_step_num = 800000
    checkpoint_num = 0
    checkpoint_dir = "checkpoints"
    epoch_num = total_step_num - checkpoint_num
    if mode == 0:
        run_cifar10(n_step=epoch_num, checkpoint_dir=checkpoint_dir, checkpoint_epoch=checkpoint_num)
    else:
        sample_cifar10(checkpoint_dir=checkpoint_dir, checkpoint_epoch=checkpoint_num)
        
