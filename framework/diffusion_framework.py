from framework.DDPM.ddpm import DDPM
from framework.LDM import LDM

import jax
import jax.numpy as jnp

from utils.fs_utils import FSUtils
from utils import jax_utils, common_utils
from utils.fid_utils import FIDUtils

from tqdm import tqdm

class DiffusionFramework():
    """
        This framework contains overall methods for training and sampling

    """
    def __init__(self, model_type, config, random_rng) -> None:
        self.model_type = model_type.lower()
        self.random_rng = random_rng
        self.set_utils(config)
        self.set_model(config)
        self.set_step(config)
        self.learning_rate_schedule = jax_utils.get_learning_rate_schedule(config, model_type)
    
    def set_utils(self, config):
        self.fid_utils = FIDUtils(config)
        self.fs_utils = FSUtils(config)
        self.fs_utils.verifying_or_create_workspace()

    def set_model(self, config):
        if self.model_type == 'ddpm':
            ddpm_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = DDPM(config, ddpm_rng, self.fs_utils)
        elif self.model_type == "ldm":
            ldm_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = LDM(config, ldm_rng, self.fs_utils)
        
    def set_step(self, config):
        # framework_config = config['framework']
        if self.model_type == "ddpm":
            # self.step = self.fs_utils.get_start_step_from_checkpoint() - 1
            self.step = self.fs_utils.get_start_step_from_checkpoint()
            self.total_step = config['framework']['diffusion']['train']['total_step']
        elif self.model_type == "ldm":
            self.train_idx = config['framework']['train_idx']
            self.step = self.fs_utils.get_start_step_from_checkpoint(idx=self.train_idx)
            if self.train_idx == 1: # AE
                self.total_step = config['framework']['autoencoder']['train']['total_step']
            elif self.train_idx == 2: # Diffusion
                self.total_step = config['framework']['diffusion']['train']['total_step']

    def fit(self, x, cond=None, step=0):
        log = self.framework.fit(x, step=step)
        return log

    def sampling(self, num_img, img_size=None, original_data=None):
        dataset_name = self.fs_utils.get_dataset_name()
        if img_size is None:
            if dataset_name == "cifar10":
                img_size = (32, 32, 3)
        sample = self.framework.sampling(num_img, img_size=img_size, original_data=original_data)
        return sample
    
    def train(self):
        datasets = common_utils.load_dataset_from_tfds()
        datasets_bar = tqdm(datasets, total=self.total_step-self.step)
        
        for x, _ in datasets_bar:
            x = jax.device_put(x.numpy())
            log = self.framework.fit(x, step=self.step)
            
            # loss_ema = loss.item()
            # loss_ema = log['loss']
            if self.model_type == "ldm":
                loss_ema = log["autoencoder"]["train/0_total_loss"]
            datasets_bar.set_description("Step: {step} loss: {loss:.4f}  lr*1e4: {lr:.4f}".format(
                step=self.step,
                loss=loss_ema,
                lr=self.learning_rate_schedule(self.step) * (1e4)
            ))

            if self.step % 1000 == 0:
                # Record various loss in here. 
                sample = self.sampling(8, (32, 32, 3), original_data=x[:8])
                xset = jnp.concatenate([sample[:8], x[:8]], axis=0)
                self.fs_utils.save_comparison(xset, self.step, self.fs_utils.get_in_process_dir())

            if self.step % 50000 == 0:
                model_state = self.framework.get_model_state()

                jax_utils.save_train_state(
                    model_state, 
                    self.fs_utils.get_checkpoint_dir(), 
                    self.step, 
                    prefix=self.fs_utils.get_state_prefix(self.model_type))

                # Calculate FID score with 1000 samples
                fid_score = self.fid_utils.calculate_fid_in_step(self.step, self.framework, 5000, batch_size=128)
                if self.fs_utils.get_best_fid() >= fid_score:
                    best_checkpoint_dir = self.fs_utils.get_best_checkpoint_dir()
                    jax_utils.save_best_state(model_state, best_checkpoint_dir, self.step)
            
            if self.step >= self.total_step:
                break
            self.step += 1
