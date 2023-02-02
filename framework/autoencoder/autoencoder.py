from model.autoencoder import AutoEncoderKL as AEKL
from model.autoencoder import AutoEncoderVQ as AEVQ
from framework.autoencoder.discriminator import LPIPSwithDiscriminator_KL, LPIPSwithDiscriminator_VQ
from utils import jax_utils
from utils.ema import EMA
from utils.fs_utils import FSUtils
from utils.log_utils import WandBLog

import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import TypedDict 


def generator_loss_kl(g_params, d_params, autoencoder: nn.Module, discriminator: nn.Module, x, step, rng):
    # losses_fn = jax.jit(jax.value_and_grad(nll_and_d_loss, has_aux=True))
    kl_rng, dropout_rng = jax.random.split(rng, 2)
    rng = {'gaussian': kl_rng, "dropout": dropout_rng}
    # x_rec, posteriors = autoencoder.apply({"params": g_params}, x=x, train=True, rngs=rng)
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
    # losses_fn = jax.jit(jax.value_and_grad(nll_and_d_loss, has_aux=True))
    kl_rng, dropout_rng = jax.random.split(rng, 2)
    rng = {'gaussian': kl_rng, "dropout": dropout_rng}
    # x_rec, codebook_diff, ind = autoencoder.apply({"params": g_params}, x=x, train=True, rngs=rng)
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
    # return z
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
    def __init__(self, config, rand_rng, fs_obj: FSUtils, wandblog: WandBLog):
    # def setup(self):
        self.framework_config = config['framework']['autoencoder']
        self.random_rng = rand_rng
        self.wandblog = wandblog

        # self.model_config = config['model']['autoencoder']
        self.mode = self.framework_config['mode']
        self.autoencoder_type = config['framework']['autoencoder']['mode']
        if self.autoencoder_type == "KL":
            self.model = AEKL(**config['model']['autoencoder'])
        elif self.autoencoder_type == "VQ":
            self.model = AEVQ(**config['model']['autoencoder'])

        # Autoencoder init
        g_state_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        self.g_model_state = jax_utils.create_train_state(config, 'autoencoder', self.model, g_state_rng) # Generator

        try: 
            checkpoint_dir = config['framework']['pretrained_ae']
        except KeyError:
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
        d_state_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        aux_data = [self.model, self.g_model_state.params]
        self.d_model_state = jax_utils.create_train_state(config, 'discriminator', self.discriminator, d_state_rng, aux_data) # Discriminator
        self.d_model_state = fs_obj.load_model_state('discriminator', self.d_model_state)       

        if self.autoencoder_type == "KL":
            self.g_loss_fn = jax.jit(
                    functools.partial(
                        jax.value_and_grad(generator_loss_kl, has_aux=True), 
                        autoencoder=self.model,
                        discriminator=self.discriminator
                    )
                )
            self.d_loss_fn = jax.jit(
                    functools.partial(
                        jax.value_and_grad(discriminator_loss_kl, has_aux=True),
                        discriminator=self.discriminator)
                )
        elif self.autoencoder_type == "VQ":
            self.g_loss_fn = jax.jit(
                    functools.partial(
                        jax.value_and_grad(generator_loss_vq, has_aux=True), 
                        autoencoder=self.model,
                        discriminator=self.discriminator
                    )
                )
            self.d_loss_fn = jax.jit(
                    functools.partial(
                        jax.value_and_grad(discriminator_loss_vq, has_aux=True),
                        discriminator=self.discriminator)
                )
        self.update_model = jax.jit(update_grad)
        self.encoder = jax.jit(functools.partial(encoder_fn, autoencoder=self.model))
        self.decoder = jax.jit(functools.partial(decoder_fn, autoencoder=self.model))
        
        self.recon_model = jax.jit(functools.partial(reconstruction_fn, autoencoder=self.model))

        # Create ema obj
        ema_config = config['ema']
        self.ema_obj = EMA(self.g_model_state.params, **ema_config)
    
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
        self.ema_obj.ema_update(self.g_model_state.params, step)

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
                ####
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
                ####
                f"{split}/disc_loss": d_log[f"{split}/disc_loss"],
                f"{split}/logits_real": d_log[f"{split}/logits_real"],
                f"{split}/logits_fake": d_log[f"{split}/logits_fake"]
            }
        return log

    def sampling(self, num_image, img_size=(32, 32, 3)):
        NotImplementedError("AutoEncoder cannot generate samples by itself. Use LDM framework.")

    def reconstruction(self, original_data):
        rng, self.random_rng = jax.random.split(self.random_rng, 2)
        x_rec = self.recon_model(self.g_model_state.params, x=original_data, rng=rng)
        return x_rec
        
    def encoder_forward(self, x):
        encoder_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        e_x = self.encoder(g_params=self.g_model_state.params, x=x, rng=encoder_rng)
        return e_x
        
    def decoder_forward(self, z):
        d_e_x = self.decoder(g_params=self.g_model_state.params, z=z)
        return d_e_x
    
    def set_ema_params_to_state(self):
        self.g_model_state = self.g_model_state.replace(params_ema=self.ema_obj.get_ema_params())

    def get_model_state(self) -> TypedDict:
        self.set_ema_params_to_state()
        return [self.g_model_state, self.d_model_state]

        