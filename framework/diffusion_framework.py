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
            self.step = self.fs_utils.get_start_step_from_checkpoint() - 1
            self.total_step = config['framework']['diffusion']['train']['total_step']
        elif self.model_type == "ldm":
            train_idx = config['framework']['train_idx']
            self.step = self.fs_utils.get_start_step_from_checkpoint(idx=train_idx)
            if train_idx == 0: # AE
                self.total_step = config['framework']['autoencoder']['train']['total_step']
            elif train_idx == 1: # Diffusion
                self.total_step = config['framework']['diffusion']['train']['total_step']

    def fit(self, x):
        log = self.framework.fit(x)
        return log

    def sampling(self, num_img):
        dataset_name = self.fs_utils.get_dataset_name()
        if dataset_name == "cifar10":
            img_size = (32, 32, 3)
        sample = self.framework.sampling(num_img, img_size=img_size)
        return sample
    
    def train(self):
        datasets = common_utils.load_dataset_from_tfds()
        datasets_bar = tqdm(datasets, total=self.total_step-self.step)
        
        for x, _ in datasets_bar:
            x = jax.device_put(x.numpy())
            log = self.framework.fit(x)
            
            # loss_ema = loss.item()
            loss_ema = log['loss']
            # current_ema_decay = ema_obj.ema_update(state.params, step)
            # if current_ema_decay is not None:
            #     ema_decay = current_ema_decay
            
            # datasets_bar.set_description("Step: {step} loss: {loss:.4f} EMA decay: {ema_decay:.4f} lr*1e4: {lr:.4f}".format(
            #     step=step,
            #     loss=loss_ema,
            #     ema_decay=ema_decay,
            #     lr=current_learning_rate_schedule(step) * (1e4)
            # ))
            datasets_bar.set_description("Step: {step} loss: {loss:.4f}  lr*1e4: {lr:.4f}".format(
                step=self.step,
                loss=loss_ema,
                lr=self.learning_rate_schedule(self.step) * (1e4)
            ))

            if self.step % 1000 == 0:
                sample = self.sampling(8, (32, 32, 3))
                # self.fs_utils.save_images_to_dir(sample, )
                xset = jnp.concatenate([sample[:8], x[:8]], axis=0)
                # xset = torch.from_numpy(np.array(xset))
                self.fs_utils.save_comparison(xset, self.step, self.fs_utils.get_in_process_dir())

            # if step % 10000 == 0:
            if self.step % 50000 == 0:
                # state = state.replace(params_ema = ema_obj.get_ema_params())
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

    # def save_model(self):
    #     checkpoint_dir = self.fs_utils.get_checkpoint_dir()
    #     state_dict = self.framework.get_model_state() # Dictionary of state
        
    #     for key in state_dict:
    #         state = state_dict[key]
    #         jax_utils.save_train_state(state, checkpoint_dir,)
