# from framework.DDPM.ddpm import DDPM
from framework.diffusion.diffusion_framework import DiffusionFramework
from framework.LDM import LDM

import jax
import jax.numpy as jnp

from utils.fs_utils import FSUtils
from utils import jax_utils, common_utils
from utils.fid_utils import FIDUtils
from utils.log_utils import WandBLog

from tqdm import tqdm
import os
import wandb

from omegaconf import DictConfig


class UnifyingFramework():
    """
        This framework contains overall methods for training and sampling
    """
    def __init__(self, model_type, config: DictConfig, random_rng) -> None:
        self.config = config # TODO: This code is ugly. It should not need to reuse config obj
        self.current_model_type = model_type.lower()
        self.diffusion_model_type = ['ddpm', 'ddim'] 
        self.random_rng = random_rng
        self.dataset_name = config.dataset.name
        self.do_fid_during_training = config.fid_during_training
        self.set_train_step_process(config)
    
    def set_train_step_process(self, config: DictConfig):
        self.set_utils(config)
        self.set_model(config)
        self.set_step(config)
        self.learning_rate_schedule = jax_utils.get_learning_rate_schedule(config, self.current_model_type)
        self.sample_batch_size = config.sampling_batch
    
    def set_utils(self, config: DictConfig):
        self.fid_utils = FIDUtils(config)
        self.fs_utils = FSUtils(config)
        self.wandblog = WandBLog()
        self.fs_utils.verify_and_create_workspace()

    def set_model(self, config: DictConfig):
        if self.current_model_type in self.diffusion_model_type:
            diffusion_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = DiffusionFramework(config, diffusion_rng, self.fs_utils, self.wandblog)
        elif self.current_model_type == "ldm":
            ldm_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = LDM(config, ldm_rng, self.fs_utils, self.wandblog)
        
    def set_step(self, config: DictConfig):
        if self.current_model_type in self.diffusion_model_type:
            self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='diffusion')
            self.total_step = config['framework']['diffusion']['train']['total_step']
            self.checkpoint_prefix = config.exp.diffusion_prefix
        elif self.current_model_type == "ldm":
            self.train_idx = config['framework']['train_idx']
            if self.train_idx == 1: # AE
                self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='autoencoder')
                self.total_step = config['framework']['autoencoder']['train']['total_step']
                self.checkpoint_prefix = config.exp.autoencoder_prefix
            elif self.train_idx == 2: # Diffusion
                self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='diffusion')
                self.total_step = config['framework']['diffusion']['train']['total_step']
                self.checkpoint_prefix = config.exp.diffusion_prefix

    def sampling(self, num_img, original_data=None):
        sample = self.framework.sampling(num_img, original_data=original_data)
        return sample
    
    def save_model_state(self, state:list):
        if self.current_model_type in self.diffusion_model_type or \
            (self.current_model_type == "ldm" and self.train_idx == 2):
            assert len(state) == 1
            # diffusion_prefix = self.config.get_state_prefix('diffusion')
            diffusion_prefix = self.config.exp.diffusion_prefix
            jax_utils.save_train_state(
                state[0], 
                self.config.exp.checkpoint_dir, 
                self.step, 
                prefix=diffusion_prefix)

        elif self.current_model_type == "ldm" and self.train_idx == 1:
            assert len(state) == 2
            autoencoder_prefix = self.config.exp.autoencoder_prefix
            discriminator_prefix = self.config.exp.discriminator_prefix
            jax_utils.save_train_state(
                state[0], 
                self.config.exp.checkpoint_dir, 
                self.step, 
                prefix=autoencoder_prefix)
            jax_utils.save_train_state(
                state[1], 
                self.config.exp.checkpoint_dir, 
                self.step, 
                prefix=discriminator_prefix)

    def train(self):
        datasets = common_utils.load_dataset_from_tfds()
        datasets_bar = tqdm(datasets, total=self.total_step-self.step)
        in_process_dir = self.config.exp.in_process_dir
        in_process_model_dir_name = "AE" if self.current_model_type == 'ldm' and self.train_idx == 2 else 'diffusion'
        in_process_dir = os.path.join(in_process_dir, in_process_model_dir_name)
        
        for x, _ in datasets_bar:
            x = jax.device_put(x.numpy())
            log = self.framework.fit(x, step=self.step)
            
            if self.current_model_type == "ldm" and self.train_idx == 1:
                loss_ema = log["train/total_loss"]
            else:
                loss_ema = log["diffusion_loss"]

            datasets_bar.set_description("Step: {step} loss: {loss:.4f}  lr*1e4: {lr:.4f}".format(
                step=self.step,
                loss=loss_ema,
                lr=self.learning_rate_schedule(self.step) * (1e4)
            ))

            if self.step % 1000 == 0:
                sample = self.sampling(8, original_data=x[:8])
                xset = jnp.concatenate([sample[:8], x[:8]], axis=0)
                sample_path = self.fs_utils.save_comparison(xset, self.step, in_process_dir)
                log['Sampling'] = wandb.Image(sample_path, caption=f"Step: {self.step}")
                self.wandblog.update_log(log)
                self.wandblog.flush(step=self.step)

            if self.step % 50000 == 0:
                model_state = self.framework.get_model_state()
                self.save_model_state(model_state)

                # Calculate FID score with 1000 samples
                if self.do_fid_during_training and \
                    not (self.current_model_type == "ldm" and self.train_idx == 1):
                    fid_score = self.fid_utils.calculate_fid_in_step(self.step, self.framework, 5000, batch_size=128)
                    if self.fs_utils.get_best_fid() >= fid_score:
                        best_checkpoint_dir = self.config.exp.best_dir
                        jax_utils.save_best_state(model_state, best_checkpoint_dir, self.step, self.checkpoint_prefix)
                    self.wandblog.update_log({"FID score": fid_score})
                    self.wandblog.flush(step=self.step)

            if self.step >= self.total_step:
                if not self.next_step():
                    break
            self.step += 1

    def sampling_and_save(self, total_num, img_size=None):
        if img_size is None:
            img_size = common_utils.get_dataset_size(self.dataset_name)
        current_num = 0
        batch_size = self.sample_batch_size
        while total_num > current_num:
            samples = self.sampling(batch_size, original_data=None)
            self.fs_utils.save_images_to_dir(samples, starting_pos=current_num)
            current_num += batch_size
    
    def reconstruction(self, total_num):
        img_size = common_utils.get_dataset_size(self.dataset_name)
        datasets = common_utils.load_dataset_from_tfds()
        datasets_bar = tqdm(datasets, total=total_num)
        current_num = 0
        for x, _ in datasets_bar:
            x = jax.device_put(x.numpy())
            batch_size = x.shape[0]
            samples = self.sampling(batch_size, img_size, original_data=x)
            self.fs_utils.save_images_to_dir(samples, starting_pos = current_num)
            current_num += batch_size
            if current_num >= total_num:
                break
    
    def next_step(self):
        if self.current_model_type == "ldm" and self.train_idx == 1:
            self.config['framework']['train_idx'] = 2
            self.set_train_step_process(self.config)
            return True
        else:
            return False

