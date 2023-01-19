from model.autoencoder import AutoEncoderKL as AEKL
from framework.autoencoder.discriminator import LPIPSwithDiscriminator_KL
from utils import jax_utils
from utils.ema import EMA

import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import TypedDict 


def generator_loss(g_params, d_params, autoencoder: nn.Module, discriminator: nn.Module, x, step, kl_rng):
    # losses_fn = jax.jit(jax.value_and_grad(nll_and_d_loss, has_aux=True))
    rng = {'gaussian': kl_rng}
    x_rec, posteriors = autoencoder.apply({"params": g_params}, x=x, train=True, rngs=rng)
    posteriors_kl = posteriors.kl()
    loss, log = discriminator.apply(
        {"params": d_params}, 
        inputs=x, 
        reconstructions=x_rec, 
        posteriors_kl=posteriors_kl,
        optimizer_idx=0,
        global_step=step,
        g_params=g_params)
    # return loss, (log, x_rec, posterior, last_layer)
    return loss, (log, x_rec, posteriors_kl)

def discriminator_loss(d_params, discriminator: nn.Module, x, x_rec, step, posteriors_kl):
    loss, log = discriminator.apply(
        {"params": d_params}, 
        inputs=x, 
        reconstructions=x_rec,
        posteriors_kl=posteriors_kl, 
        optimizer_idx=1, 
        global_step=step) # There will be more input
    return loss, log

def update_grad(state:train_state.TrainState, grads):
    return state.apply_gradients(grads=grads)

def encoder_fn(g_params, autoencoder, x, train, kl_rng):
    rng = {'gaussian': kl_rng}
    z = autoencoder.apply({"params": g_params}, x, train, rngs=rng, method=autoencoder.encoder)
    return z

def decoder_fn(g_params, autoencoder, z, train):
    x_rec = autoencoder.apply({"params": g_params}, z, train, method=autoencoder.decoder)
    return x_rec

# Firstly, I implement autoencoder without any regularization such as VQ and KL.
# However, it should be implemented too someday..  
class AutoEncoderKL():
    def __init__(self, config, rand_rng):
    # def setup(self):
        self.framework_config = config['framework']['autoencoder']
        self.random_rng = rand_rng

        # self.model_config = config['model']['autoencoder']
        self.mode = self.framework_config['mode']
        self.model = AEKL(**config['model']['autoencoder'])

        self.discriminator = LPIPSwithDiscriminator_KL(disc_start=0)

        # model state contains two parameters: autoencoder(generator) and discriminator
        g_state_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        self.g_model_state = jax_utils.create_train_state(config, 'autoencoder', self.model, g_state_rng) # Generator
        # self.d_model_state = jax_utils.create_train_state(config, 'discriminator', self.discriminator, d_state_rng) # Discriminator
        
        d_state_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        aux_data = [self.model, self.g_model_state.params]
        self.d_model_state = jax_utils.create_train_state(config, 'discriminator', self.discriminator, d_state_rng, aux_data) # Discriminator
        # self.g_model_ema = EMA(self.g_model_state.params)

        self.g_loss_fn = jax.jit(
                functools.partial(
                    jax.value_and_grad(generator_loss, has_aux=True), 
                    autoencoder=self.model,
                    discriminator=self.discriminator
                )
            )
        
        self.d_loss_fn = jax.jit(
                functools.partial(
                    jax.value_and_grad(discriminator_loss, has_aux=True),
                    discriminator=self.discriminator)
            )
        self.update_model = jax.jit(update_grad) # TODO: Can this function be adapted to both generator and discriminator? 
        # self.encoder = jax.jit(nn.apply(encoder_fn, autoencoder=self.model))
        # self.decoder = jax.jit(nn.apply(decoder_fn, autoencoder=self.model))
        self.encoder = jax.jit(functools.partial(encoder_fn, autoencoder=self.model))
        self.decoder = jax.jit(functools.partial(decoder_fn, autoencoder=self.model))
    
    def fit(self, x, cond=None, step=0):
        kl_rng, self.random_rng = jax.random.split(self.random_rng, 2)
        # (g_loss, (g_log, x_rec, posterior, last_layer)), grad = self.g_loss_fn(
        (g_loss, (g_log, x_rec, posteriors_kl)), grad = self.g_loss_fn(
            self.g_model_state.params, 
            d_params=self.d_model_state.params,
            x=x, step=step, kl_rng=kl_rng)
        self.g_model_state = self.update_model(self.g_model_state, grad)
        (d_loss, d_log), grad = self.d_loss_fn(
            self.d_model_state.params,
            x=x, x_rec=x_rec, posteriors_kl=posteriors_kl, step=step
        )
        self.d_model_state = self.update_model(self.d_model_state, grad)

        log = {
            "autoencoder": g_log,
            "discriminator": d_log
        }
        return log
        # return g_log
    
    def sampling(self, num_image, img_size=(32, 32, 3)):
        # if img_size == None:
            # img_size = (32, 32, 3) # CIFAR10
        NotImplementedError("AutoEncoder cannot generate samples by itself. Use LDM framework.")        
        
    def encoder_forward(self, x):
        e_x = self.encoder(g_params=self.g_model_state, x=x, train=True)
        return e_x
        
    def decoder_forward(self, z):
        d_e_x = self.decoder(g_params=self.g_model_state, z=z, train=True)
        return d_e_x
    
    # def get_last_layer(self):
    #     return self.g_model_state.params['']

    def set_ema_params_to_state(self):
        self.model_state = self.model_state.replace(params_ema=self.ema_obj.get_ema_params())

    def get_model_state(self) -> TypedDict:
        self.set_ema_params_to_state()
        # return {"DDPM": self.model_state}
        return self.model_state

        