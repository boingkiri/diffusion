# from framework.DDPM.ddpm import DDPM
from framework.diffusion.ddpm_framework import DDPMFramework
from framework.diffusion.edm_framework import EDMFramework
from framework.diffusion.consistency_framework import CMFramework
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
        self.config = config
        self.current_model_type = model_type.lower()
        self.diffusion_model_type = ['ddpm', 'ddim', 'edm', 'cm']
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
        if self.current_model_type in ['ddpm', 'ddim']:
            diffusion_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = DDPMFramework(config, diffusion_rng, self.fs_utils, self.wandblog)
        elif self.current_model_type in ['edm']:
            diffusion_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = EDMFramework(config, diffusion_rng, self.fs_utils, self.wandblog)
        elif self.current_model_type in ['cm', 'cm_diffusion']:
            diffusion_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = CMFramework(config, diffusion_rng, self.fs_utils, self.wandblog)
        elif self.current_model_type == "ldm":
            ldm_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = LDM(config, ldm_rng, self.fs_utils, self.wandblog)
        else:
            NotImplementedError("Model Type cannot be identified. Please check model name.")
        
    def set_step(self, config: DictConfig):
        # if self.current_model_type in self.diffusion_model_type:
        #     self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='diffusion')
        #     self.total_step = config['framework']['diffusion']['train']['total_step']
        #     self.checkpoint_prefix = config.exp.diffusion_prefix
        if self.current_model_type == "ldm":
            self.train_idx = config['framework']['train_idx']
            if self.train_idx == 1: # AE
                self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='autoencoder')
                self.total_step = config['framework']['autoencoder']['train']['total_step']
                self.checkpoint_prefix = config.exp.autoencoder_prefix
            elif self.train_idx == 2: # Diffusion
                self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='diffusion')
                self.total_step = config['framework']['diffusion']['train']['total_step']
                self.checkpoint_prefix = config.exp.diffusion_prefix
        else:
            self.step = self.fs_utils.get_start_step_from_checkpoint(model_type='diffusion')
            self.total_step = config['framework']['diffusion']['train']['total_step']
            self.checkpoint_prefix = config.exp.diffusion_prefix

    def sampling(self, num_img, original_data=None, mode=None):
        if mode is None:
            sample = self.framework.sampling(num_img, original_data=original_data)
        else:
            sample = self.framework.sampling(num_img, original_data=original_data, mode=mode)
        sample = jnp.reshape(sample, (num_img, *sample.shape[-3:]))
        return sample
    
    def save_model_state(self, states:dict):
        self.fs_utils.save_model_state(states, self.step, self.checkpoint_prefix)
        # for state_key in states:
        #     jax_utils.save_train_state(
        #         states[state_key],
        #         self.config.exp.checkpoint_dir,
        #         self.step,
        #         prefix=state_key + "_"
        #     )

    def train(self):
        # TODO: The connection_denoiser_type is only used in CM training. need to be fixed.
        STF_flag = self.config["framework"]["diffusion"].get("connection_denoiser_type", None)
        STF_flag = False if STF_flag is None or STF_flag != "STF" else True
        batch_size = self.config["framework"]["diffusion"]["train"]["batch_size"] \
            if not STF_flag else self.config["framework"]["diffusion"]["train"]["STF_reference_batch_size"]
        datasets = common_utils.load_dataset_from_tfds(
            batch_size=batch_size,
            n_jitted_steps=self.n_jitted_steps, x_flip=self.dataset_x_flip,
            stf=STF_flag)
        datasets_bar = tqdm(datasets, total=self.total_step-self.step, initial=self.step)
        in_process_dir = self.config.exp.in_process_dir
        in_process_model_dir_name = "AE" if self.current_model_type == 'ldm' and self.train_idx == 2 else 'diffusion'
        in_process_dir = os.path.join(in_process_dir, in_process_model_dir_name)
        best_fids = self.fs_utils.get_best_fid()
        first_step = True
        
        for x, _ in datasets_bar:
            eval_during_training = self.step % 1000 == 0
            log = self.framework.fit(x, step=self.step, eval_during_training=eval_during_training)

            description_str = "Step: {step} lr*1e4: {lr:.4f} ".format(
                step=self.step,
                lr=self.learning_rate_schedule(self.step)*(1e+4)
            )
            for key in log:
                if key.startswith("train"):
                    represented_key = key.replace("train/", "")
                    description_str += f"{represented_key}: {log[key]:.4f} "
            datasets_bar.set_description(description_str)

            if self.step % 1000 == 0:
                batch_data = x[0, 0, :8] # (device_idx, n_jitted_steps, batch_size)

                # Change of the sample quality is tracked to know how much the CM model is corrupted.
                # Sample generated image for EDM
                if not self.config.framework.diffusion.only_cm_training:
                    sample = self.sampling(8, original_data=batch_data, mode="edm")
                    edm_xset = jnp.concatenate([sample[:8], batch_data], axis=0)
                    sample_image = self.fs_utils.get_pil_from_np(edm_xset)
                    # sample_path = self.fs_utils.save_comparison(edm_xset, self.step, in_process_dir)
                    log['Sampling'] = wandb.Image(sample_image, caption=f"Step: {self.step}")

                # Sample generated image for training CM
                if not self.config.framework.diffusion.CM_freeze:
                    sample = self.sampling(8, original_data=batch_data, mode="cm-training")
                    training_cm_xset = jnp.concatenate([sample[:8], batch_data], axis=0)
                    sample_image = self.fs_utils.get_pil_from_np(training_cm_xset)
                    log['Training CM Sampling'] = wandb.Image(sample_image, caption=f"Step: {self.step}")
                    sample_path = self.fs_utils.save_comparison(training_cm_xset, self.step, in_process_dir) # TMP
                    sample_image.close()

                # Sample generated image for original CM 
                sample = self.sampling(8, original_data=batch_data, mode="cm-not-training")
                edm_xset = jnp.concatenate([sample[:8], batch_data], axis=0)
                sample_image = self.fs_utils.get_pil_from_np(edm_xset)
                log['Original CM Sampling'] = wandb.Image(sample_image, caption=f"Step: {self.step}")
                sample_image.close()
                

                self.wandblog.update_log(log)
                self.wandblog.flush(step=self.step)

            if self.step % 50000 == 0 and self.step != 0:
                model_state = self.framework.get_model_state()
                if not first_step:
                    self.save_model_state(model_state)

                # Calculate FID score with 1000 samples
                # if self.do_fid_during_training and not (self.current_model_type == "ldm" and self.train_idx == 1):
                #     fid_score = self.fid_utils.calculate_fid_in_step(self.step, self.framework, 10000, batch_size=128)
                #     if best_fid >= fid_score:
                #         best_checkpoint_dir = self.config.exp.best_dir
                #         jax_utils.save_best_state(model_state, best_checkpoint_dir, self.step, "diffusion_")

                #         best_fid = fid_score
                #     self.wandblog.update_log({"FID score": fid_score})
                #     self.wandblog.flush(step=self.step)
                
                # TMP: Calculate FID score with 10000 samples for both EDM and CM, 
                # which corresponds to the head and the torso, respectively.

                if self.config.framework.diffusion.only_cm_training:
                    sampling_modes = ['cm-training']
                elif self.config.framework.diffusion.CM_freeze:
                    sampling_modes = ['edm']
                else:
                    sampling_modes = ['edm', 'cm-training']
                if self.do_fid_during_training and not (self.current_model_type == "ldm" and self.train_idx == 1):
                    for mode in sampling_modes:
                        fid_score = self.fid_utils.calculate_fid_in_step(self.framework, 10000, batch_size=128, sampling_mode=mode)
                        self.fid_utils.print_and_save_fid(self.step, fid_score, sampling_mode=mode)
                        if not mode in best_fids or best_fids[mode] >= fid_score:
                            best_checkpoint_dir = self.config.exp.best_dir
                            best_checkpoint_dir = os.path.join(best_checkpoint_dir, mode)
                            os.makedirs(best_checkpoint_dir, exist_ok=True)
                            if mode == "edm":
                                jax_utils.save_best_state(model_state, best_checkpoint_dir, self.step, "head_")
                            else:
                                jax_utils.save_best_state(model_state, best_checkpoint_dir, self.step, "torso_")

                            best_fids[mode] = fid_score
                        if mode == "edm":
                            self.wandblog.update_log({"Head FID score": fid_score})
                        else:
                            self.wandblog.update_log({"Torso FID score": fid_score})
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
            samples = self.sampling(effective_batch_size, original_data=None, mode="cm-training")
            self.fs_utils.save_images_to_dir(samples, starting_pos=current_num)
            current_num += batch_size
    
    def reconstruction(self, total_num):
        img_size = common_utils.get_dataset_size(self.dataset_name)
        datasets = common_utils.load_dataset_from_tfds(n_jitted_step=self.n_jitted_steps)
        datasets_bar = tqdm(datasets, total=total_num)
        current_num = 0
        for x, _ in datasets_bar:
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

