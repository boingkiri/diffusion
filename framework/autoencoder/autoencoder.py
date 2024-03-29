from model.autoencoder import AutoEncoderKL as AEKL
from model.autoencoder import AutoEncoderVQ as AEVQ
from framework.autoencoder.discriminator import LPIPSwithDiscriminator_KL, LPIPSwithDiscriminator_VQ
from utils import jax_utils
from utils.ema.ema_ddpm import DDPMEMA
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
from flax.training import train_state

from typing import TypedDict 
from omegaconf import DictConfig

import os
from functools import partial


def generator_loss_kl(g_params, d_params, autoencoder: nn.Module, discriminator: nn.Module, x, step, rng):
    kl_rng, dropout_rng = jax.random.split(rng, 2)
    rng = {'gaussian': kl_rng, "dropout": dropout_rng}
    x_rec_complete, x_rec, posteriors = autoencoder.apply(
        {"params": g_params}, x=x, train=True, rngs=rng,)
    conv_out_params = g_params['decoder_model']['conv_out']
    posteriors_kl = posteriors.kl()
    loss, log = discriminator.apply(
        {"params": d_params}, 
        inputs=x, 
        reconstructions=x_rec, 
        posteriors_kl=posteriors_kl,
        optimizer_idx=0,
        global_step=step,
        conv_out_params=conv_out_params)
    return loss, (log, x_rec_complete, posteriors_kl)

def discriminator_loss_kl(d_params, discriminator: nn.Module, x, x_rec, step, posteriors_kl):
    loss, log = discriminator.apply(
        {"params": d_params}, 
        inputs=x, 
        reconstructions=x_rec,
        posteriors_kl=posteriors_kl, 
        optimizer_idx=1, 
        global_step=step) # There will be more input
    return loss, log

def generator_loss_vq(g_params, d_params, autoencoder: nn.Module, discriminator: nn.Module, x, step, rng):
    kl_rng, dropout_rng = jax.random.split(rng, 2)
    rng = {'gaussian': kl_rng, "dropout": dropout_rng}
    x_rec_complete, x_rec, codebook_diff, ind = autoencoder.apply(
        {"params": g_params}, x=x, train=True, rngs=rng)
    conv_out_params = g_params['decoder_model']['conv_out']
    loss, log = discriminator.apply(
        {"params": d_params}, 
        inputs=x, 
        reconstructions=x_rec, 
        codebook_loss=codebook_diff,
        optimizer_idx=0,
        global_step=step,
        conv_out_params=conv_out_params,
        predicted_indices=ind
        )
    return loss, (log, x_rec_complete, codebook_diff)

def discriminator_loss_vq(d_params, discriminator: nn.Module, x, x_rec, step, codebook_loss):
    loss, log = discriminator.apply(
        {"params": d_params}, 
        inputs=x, 
        reconstructions=x_rec,
        codebook_loss=codebook_loss,
        optimizer_idx=1, 
        global_step=step) # There will be more input
    return loss, log

def update_grad(state:train_state.TrainState, grads):
    return state.apply_gradients(grads=grads)

def encoder_fn(g_params, autoencoder, x, rng):
    kl_rng, dropout_rng = jax.random.split(rng, 2)
    rng = {'gaussian': kl_rng, "dropout": dropout_rng}
    z = autoencoder.apply({"params": g_params}, x=x, train=False, rngs=rng, method=autoencoder.encoder)
    if type(z) is tuple:
        retval = z[0]
    else:
        retval = z.sample()
    return retval

def decoder_fn(g_params, autoencoder, z):
    x_rec = autoencoder.apply({"params": g_params}, z=z, train=False, method=autoencoder.decoder)
    return x_rec

def reconstruction_fn(g_params, autoencoder, x, rng):
    kl_rng, dropout_rng = jax.random.split(rng, 2)
    rng = {"gaussian": kl_rng, "drooput": dropout_rng}
    return_values = autoencoder.apply({"params": g_params}, x, False, rngs= rng)
    x_rec = return_values[0]
    return x_rec

# Firstly, I implement autoencoder without any regularization such as VQ and KL.
# However, it should be implemented too someday..  
class AutoEncoder():
    def __init__(self, config: DictConfig, rand_rng, fs_obj: FSUtils, wandblog: WandBLog):
        self.autoencoder_framework_config = config.framework.autoencoder
        self.random_rng = rand_rng
        self.wandblog = wandblog
        self.pmap = config.pmap

        self.autoencoder_model_config = config.model.autoencoder
        self.autoencoder_type = self.autoencoder_framework_config['mode']
        if self.autoencoder_type == "KL":
            self.model = AEKL(**self.autoencoder_model_config)
        elif self.autoencoder_type == "VQ":
            self.model = AEVQ(**self.autoencoder_model_config)

        # Autoencoder init
        # g_state_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        # self.g_model_state = jax_utils.create_train_state(config, 'autoencoder', self.model, g_state_rng, None) # Generator
        self.g_model_state = self.init_model_state(config, 'autoencoder')
        
        if config['framework']['train_idx'] == 2 and 'pretrained_ae' in config['framework'].keys():
            checkpoint_dir = os.path.join(config['exp']['exp_dir'], config['framework']['pretrained_ae'])
            checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
            print(f"Checkpoint dir: {checkpoint_dir}")
        else:
            checkpoint_dir = None
        
        self.g_model_state = fs_obj.load_model_state('autoencoder', self.g_model_state, checkpoint_dir)
        if self.autoencoder_type == "KL":
            self.discriminator = LPIPSwithDiscriminator_KL(
                **config['model']['discriminator'], 
                autoencoder=self.model)
        elif self.autoencoder_type == "VQ":
            self.discriminator = LPIPSwithDiscriminator_VQ(
                **config['model']['discriminator'], 
                n_classes=config['model']['autoencoder']['n_embed'],
                autoencoder=self.model)

        # Discriminator init
        # d_state_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        # aux_data = [self.model, self.g_model_state.params]
        # self.d_model_state = jax_utils.create_train_state(config, 'discriminator', self.discriminator, d_state_rng, aux_data) # Discriminator
        self.d_model_state = self.init_model_state(config, 'discriminator')
        self.d_model_state = fs_obj.load_model_state('discriminator', self.d_model_state)       

        if self.autoencoder_type == "KL":
            self.g_loss_fn = jax.jit(
                    partial(
                        jax.value_and_grad(generator_loss_kl, has_aux=True), 
                        autoencoder=self.model,
                        discriminator=self.discriminator
                    )
                )
            self.d_loss_fn = jax.jit(
                    partial(
                        jax.value_and_grad(discriminator_loss_kl, has_aux=True),
                        discriminator=self.discriminator)
                )
        elif self.autoencoder_type == "VQ":
            self.g_loss_fn = jax.jit(
                    partial(
                        jax.value_and_grad(generator_loss_vq, has_aux=True), 
                        autoencoder=self.model,
                        discriminator=self.discriminator
                    )
                )
            self.d_loss_fn = jax.jit(
                    partial(
                        jax.value_and_grad(discriminator_loss_vq, has_aux=True),
                        discriminator=self.discriminator)
                )
        self.update_model = jax.jit(update_grad)


        if self.pmap:
            self.encoder = jax.pmap(partial(encoder_fn, autoencoder=self.model))
            self.decoder = jax.pmap(partial(decoder_fn, autoencoder=self.model))
        else:
            self.encoder = jax.jit(partial(encoder_fn, autoencoder=self.model))
            self.decoder = jax.jit(partial(decoder_fn, autoencoder=self.model))
        
        self.recon_model = jax.jit(partial(reconstruction_fn, autoencoder=self.model))

        # Create ema obj
        ema_config = config['ema']
        self.ema_obj = DDPMEMA(**ema_config)
        if self.pmap:
            self.g_model_state = flax.jax_utils.replicate(self.g_model_state)
            self.d_model_state = flax.jax_utils.replicate(self.d_model_state)
        
    
    def fit(self, x, cond=None, step=0):
        rng, self.random_rng = jax.random.split(self.random_rng, 2)
        if self.autoencoder_type == "KL":
            (_, (g_log, x_rec, posteriors_kl)), grad = self.g_loss_fn(
                self.g_model_state.params, 
                d_params=self.d_model_state.params,
                x=x, step=step, rng=rng)
            self.g_model_state = self.update_model(self.g_model_state, grad)
            (_, d_log), grad = self.d_loss_fn(
                self.d_model_state.params,
                x=x, x_rec=x_rec, posteriors_kl=posteriors_kl, step=step
            )
        elif self.autoencoder_type == "VQ":
            (_, (g_log, x_rec, codebook_diff)), grad = self.g_loss_fn(
                self.g_model_state.params, 
                d_params=self.d_model_state.params,
                x=x, step=step, rng=rng)
            self.g_model_state = self.update_model(self.g_model_state, grad)
            (_, d_log), grad = self.d_loss_fn(
                self.d_model_state.params,
                x=x, x_rec=x_rec, codebook_loss=codebook_diff, step=step
            )
        self.d_model_state = self.update_model(self.d_model_state, grad)

        # Update EMA parameters
        self.ema_obj.ema_update(self.g_model_state, step)

        split = "train"
        log_list = [g_log, d_log]
        log = self.get_log(log_list, split=split)
        self.wandblog.update_log(log)
        return log
    
    def get_log(self, log_list, split="train"):
        if self.autoencoder_type == "KL":
            g_log, d_log = log_list[0], log_list[1]
            log = {
                f"{split}/total_loss": g_log[f"{split}/total_loss"], 
                f"{split}/kl_loss": g_log[f"{split}/kl_loss"], 
                f"{split}/nll_loss": g_log[f"{split}/nll_loss"],
                f"{split}/d_weight": g_log[f"{split}/d_weight"],
                f"{split}/disc_factor": g_log[f"{split}/disc_factor"],
                f"{split}/g_loss": g_log[f"{split}/g_loss"],
                f"{split}/disc_loss": d_log[f"{split}/disc_loss"],
                f"{split}/logits_real": d_log[f"{split}/logits_real"],
                f"{split}/logits_fake": d_log[f"{split}/logits_fake"]
            }
        elif self.autoencoder_type == "VQ":
            g_log, d_log = log_list[0], log_list[1]
            log = {
                f"{split}/total_loss": g_log[f"{split}/total_loss"], 
                f"{split}/quant_loss": g_log[f"{split}/quant_loss"], 
                f"{split}/nll_loss": g_log[f"{split}/nll_loss"],
                f"{split}/d_weight": g_log[f"{split}/d_weight"],
                f"{split}/disc_factor": g_log[f"{split}/disc_factor"],
                f"{split}/g_loss": g_log[f"{split}/g_loss"],
                f"{split}/perplexity": g_log[f"{split}/perplexity"], ## NEW!
                f"{split}/cluster_usage": g_log[f"{split}/cluster_usage"], ## NEW!
                f"{split}/disc_loss": d_log[f"{split}/disc_loss"],
                f"{split}/logits_real": d_log[f"{split}/logits_real"],
                f"{split}/logits_fake": d_log[f"{split}/logits_fake"]
            }
        return log

    def sampling(self, num_image, img_size=(32, 32, 3)):
        NotImplementedError("AutoEncoder cannot generate samples by itself. Use LDM framework.")
    
    def init_model_state(self, config: DictConfig, model_type):
        rng, param_rng, dropout_rng, self.random_rng = jax.random.split(self.random_rng, 4)
        input_format = jnp.ones([1, *config.dataset.data_size])
        if model_type == "autoencoder":
            rng, gaussian_rng = jax.random.split(rng, 2)
            rng_dict = {"params": param_rng, 'dropout': dropout_rng, 'gaussian': gaussian_rng}
            params = self.model.init(rng_dict, x=input_format, train=False)['params']
        elif model_type == "discriminator":
            kl_rng, rng = jax.random.split(rng, 2)
            input_format3 = jnp.ones([1, 32, 32, 3]) 
            conv_out_params = self.g_model_state.params['decoder_model']['conv_out']
            def kl_model_init():
                reconstructions, posteriors = self.model.apply(
                    {"params": self.g_model_state.params},
                    x=input_format,
                    train=False,
                    rngs={'gaussian': kl_rng},
                    method=self.model.forward_before_conv_out
                )
                rng_dict = {"params": param_rng, 'dropout': dropout_rng}
                posteriors_kl = posteriors.kl()
                return self.discriminator.init(rng_dict, inputs=input_format3, reconstructions=reconstructions, 
                                posteriors_kl=posteriors_kl, optimizer_idx=0, global_step=0, conv_out_params=conv_out_params)
            def vq_model_init():
                reconstructions, quantization_diff, ind = self.model.apply(
                    {"params": self.g_model_state.params},
                    x=input_format,
                    train=False,
                    rngs={'gaussian': kl_rng},
                    method=self.model.forward_before_conv_out
                )
                rng_dict = {"params": param_rng, 'dropout': dropout_rng}
                return self.discriminator.init(rng_dict, inputs=input_format3, reconstructions=reconstructions, 
                                    codebook_loss=quantization_diff, optimizer_idx=0, global_step=0, 
                                    conv_out_params=conv_out_params, predicted_indices=ind)
            if config.framework.autoencoder.mode == 'KL':
                experiment_fn_jit = kl_model_init
            elif config.framework.autoencoder.mode == 'VQ':
                experiment_fn_jit = vq_model_init  
        params = experiment_fn_jit()['params']
        apply_fn = self.model.apply if model_type == "autoencoder" else self.discriminator.apply

        return jax_utils.create_train_state(config, model_type, apply_fn, params)

    def reconstruction(self, original_data):
        rng, self.random_rng = jax.random.split(self.random_rng, 2)
        x_rec = self.recon_model(self.g_model_state.params, x=original_data, rng=rng)
        return x_rec
    
    def encoder_forward(self, x):
        encoder_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        if self.pmap:
            encoder_rng = jax.random.split(encoder_rng, jax.local_device_count())
        e_x = self.encoder(g_params=self.g_model_state.params, x=x, rng=encoder_rng)
        return e_x
    
    def decoder_forward(self, z):
        d_e_x = self.decoder(g_params=self.g_model_state.params, z=z)
        return d_e_x
    
    def get_model_state(self) -> TypedDict:
        if self.pmap:
            return [
                flax.jax_utils.unreplicate(self.g_model_state),
                flax.jax_utils.unreplicate(self.d_model_state)
            ]
        return [self.g_model_state, self.d_model_state]

        