from model.autoencoder import AutoEncoder as AEModel
from framework.autoencoder.discriminator import LPIPSwithDiscriminator_KL
# from framework.default_diffusion import DefaultModel
from utils import jax_utils

import functools

import jax
import flax.linen as nn
from flax.training import train_state

def generator_loss(g_params, d_params, autoencoder: nn.Module, discriminator: nn.Module, x, train):
    x_rec = autoencoder.apply({"params": g_params}, x=x, train=train)
    loss, log = discriminator.apply({"params": d_params}, x_rec, x, optimizer_idx=0) # There will be more input
    return loss, x_rec, log

def discriminator_loss(d_params, discriminator: nn.Module, x, x_rec, train=True):
    loss, log = discriminator.apply({"params": d_params}, x_rec, x, optimizer_idx=1) # There will be more input
    return loss, log

def update_grad(state:train_state.TrainState, grad):
    return state.apply_gradients(state, grad)

def encoder_fn(g_params, autoencoder: AEModel, x, train):
    z = autoencoder.apply({"params": g_params}, x, train, method=autoencoder.encoder)
    z = autoencoder.encoder(x, train)
    return z

def decoder_fn(g_params, autoencoder: AEModel, z, train):
    x_rec = autoencoder.apply({"params": g_params}, z, train, method=autoencoder.decoder)
    return x_rec

# Firstly, I implement autoencoder without any regularization such as VQ and KL.
# However, it should be implemented too someday..  
class AutoEncoder():
    def __init__(self, config, rand_rng):
        self.framework_config = config['framework']
        self.random_rng = rand_rng

        self.model_config = config['model']
        self.mode = self.framework_config['autoencoder']['type']
        self.model = AEModel(**self.model_config['autoencoder'])

        # Discriminator setting
        if self.mode == "KL":
            self.discriminator = LPIPSwithDiscriminator_KL()
        
        # model state contains two parameters: autoencoder(generator) and discriminator
        g_state_rng, d_state_rng, self.random_rng = jax.random.split(self.random_rng, 3)
        self.g_model_state = jax_utils.create_train_state(config, self.model, g_state_rng) # Generator
        self.d_model_state = jax_utils.create_train_state(config, self.discriminator, d_state_rng) # Discriminator
            
        self.g_loss_fn = jax.jit(
            functools.partial(
                jax.value_and_grad(generator_loss, has_aux=True), 
                autoencoder=self.model,
                discriminator=self.discriminator))
        
        self.d_loss_fn = jax.jit(
            functools.partial(
                jax.value_and_grad(discriminator_loss, has_aux=True),
                discriminator=self.discriminator))
        self.update_model = jax.jit(update_grad) # TODO: Can this function be adapted to both generator and discriminator? 
        self.encoder = jax.jit(nn.apply(encoder_fn, autoencoder=self.model))
        self.decoder = jax.jit(nn.apply(decoder_fn, autoencoder=self.model))
    
    def fit(self, x, cond=None):
        (g_loss, x_rec, g_log), grad = self.g_loss_fn(
            g_params=self.model.params, 
            d_params=self.discriminator.params,
            x=x, train=True)
        self.g_model_state = self.update_model(self.g_model_state, grad)
        (d_loss, d_log), grad = self.d_loss_fn(
            d_params=self.discriminator.params,
            x_rec=x_rec, x=x, train=True
        )
        self.d_model_state = self.update_model(self.d_model_state, grad)
        return g_log, d_log
    
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
