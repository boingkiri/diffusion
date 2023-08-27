import jax
import jax.numpy as jnp

from utils.fs_utils import FSUtils
from utils import jax_utils, common_utils
from utils.fid_utils import FIDUtils
from utils.log_utils import WandBLog
from utils.common_utils import load_class_from_config_for_framework

from tqdm import tqdm
import os
import wandb

from omegaconf import DictConfig


class UnifyingFramework():
    """
        This framework contains overall methods for training and sampling
    """
    def __init__(self, model_type, config: DictConfig, random_rng) -> None:
        self.config = config
        self.current_model_type = model_type.lower()
        self.random_rng = random_rng
        self.dataset_name = config.dataset.name
        self.do_fid_during_training = config.fid_during_training
        self.n_jitted_steps = config.get("n_jitted_steps", 1)
        self.dataset_x_flip = self.config.framework.diffusion.get("augment_rate", None) is None
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
        rng, self.random_rng = jax.random.split(self.random_rng, 2)
        framework_class = load_class_from_config_for_framework(self.framework)
        self.framework = framework_class(config, rng, self.fs_utils, self.wandblog)
        
    def set_step(self, config: DictConfig):
        model_type = config['type']
        if self.current_model_type == "ldm":
            self.train_idx = config['framework']['train_idx']
            if self.train_idx == 1: # AE
                self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='autoencoder')
                self.total_step = config['framework']['autoencoder']['train']['total_step']
            elif self.train_idx == 2: # Diffusion
                self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='diffusion')
                self.total_step = config['framework']['diffusion']['train']['total_step']
        else:
            self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='diffusion')
            self.total_step = config['framework']['diffusion']['train']['total_step']

    def sampling(self, num_img, original_data=None):
        sample = self.framework.sampling(num_img, original_data=original_data)
        sample = jnp.reshape(sample, (num_img, *sample.shape[-3:]))
        return sample
    
    def save_model_state(self, state_dict:dict, checkpoint_dir: str=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.config.exp.checkpoint_dir
        for state_name, state in state_dict.items():
            jax_utils.save_train_state(
                state, 
                checkpoint_dir, 
                self.step, 
                prefix=state_name)

    def train(self):
        datasets = common_utils.load_dataset_from_tfds(n_jitted_steps=self.n_jitted_steps, x_flip=self.dataset_x_flip)
        datasets_bar = tqdm(datasets, total=self.total_step-self.step, initial=self.step)
        in_process_dir = self.config.exp.in_process_dir
        in_process_model_dir_name = "AE" if self.current_model_type == 'ldm' and self.train_idx == 2 else 'diffusion'
        in_process_dir = os.path.join(in_process_dir, in_process_model_dir_name)
        best_fid = self.fs_utils.get_best_fid()
        first_step = True
        
        for x, _ in datasets_bar:
            random_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            log = self.framework.fit(x0=x, rng=random_rng, step=self.step)
            
            if self.current_model_type == "ldm" and self.train_idx == 1:
                loss_ema = log["train/total_loss"]
            else:
                loss_ema = log["total_loss"]

            datasets_bar.set_description("Step: {step} loss: {loss:.4f}  lr*1e4: {lr:.4f}".format(
                step=self.step,
                loss=loss_ema,
                lr=self.learning_rate_schedule(self.step) * (1e4)
            ))

            if self.step % self.config['step']['logging_step'] == 0:
                self.wandblog.flush(step=self.step)

            if self.step % self.config['step']['sampling_step'] == 0:
                batch_data = x[0, 0, :8] # (device_idx, n_jitted_steps, batch_size)
                sample = self.sampling(8, original_data=batch_data)
                xset = jnp.concatenate([sample[:8], batch_data], axis=0)
                sample_path = self.fs_utils.save_comparison(xset, self.step, in_process_dir)
                log['Sampling'] = wandb.Image(sample_path, caption=f"Step: {self.step}")
                self.wandblog.update_log(log)
                self.wandblog.flush(step=self.step)

            if self.step % self.config['step']['save_step'] == 0 and self.step != 0:
                model_state = self.framework.get_model_state()
                if not first_step:
                    self.save_model_state(model_state)

            if self.step % self.config['step']['eval_step'] == 0 and self.step != 0:
                # Calculate FID score with 1000 samples
                if self.do_fid_during_training and \
                    not (self.current_model_type == "ldm" and self.train_idx == 1):
                    fid_score = self.fid_utils.calculate_fid_in_step(
                        self.step, 
                        self.framework, 
                        self.config['fid']['num_sampling_during_training'],
                        batch_size=128)
                    if best_fid >= fid_score:
                        best_checkpoint_dir = self.config.exp.best_dir
                        self.save_model_state(model_state, best_checkpoint_dir)
                        best_fid = fid_score
                    self.wandblog.update_log({"FID score": fid_score})
                    self.wandblog.flush(step=self.step)

            if self.step >= self.total_step:
                if not self.next_step():
                    break
            self.step += self.n_jitted_steps
            first_step = False
            datasets_bar.update(self.n_jitted_steps)
            

    def sampling_and_save(self, total_num, img_size=None):
        if img_size is None:
            img_size = common_utils.get_dataset_size(self.dataset_name)
        current_num = 0
        batch_size = self.sample_batch_size
        while total_num > current_num:
            effective_batch_size = total_num - current_num if current_num + batch_size > total_num else batch_size
            samples = self.sampling(effective_batch_size, original_data=None)
            self.fs_utils.save_images_to_dir(samples, starting_pos=current_num)
            current_num += batch_size
    
    def reconstruction(self, total_num):
        img_size = common_utils.get_dataset_size(self.dataset_name)
        datasets = common_utils.load_dataset_from_tfds(n_jitted_step=self.n_jitted_steps)
        datasets_bar = tqdm(datasets, total=total_num)
        current_num = 0
        for x, _ in datasets_bar:
            batch_size = x.shape[0]
            self.random_rng, rng = jax.random.split(self.random_rng, 2)
            samples = self.sampling(batch_size, rng, img_size, original_data=x)
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

