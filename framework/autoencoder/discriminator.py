import lpips_jax

import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial

def hinge_d_loss(logits_real, logits_fake):
    loss_real = jnp.mean(nn.relu(1. - logits_real))
    loss_fake = jnp.mean(nn.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        jnp.mean(nn.softplus(-logits_real)) +
        jnp.mean(nn.softplus(logits_fake))
    )
    return d_loss

class LPIPSwithDiscriminator_KL(nn.Module):
    disc_start: int
    logvar_init: float=0.0
    kl_weight: float=1.0
    pixelloss_weight: float=1.0
    disc_num_layers: int=3
    disc_in_channels: int=3
    disc_factor: float=1.0
    disc_weight: float=1.0
    perceptual_weight: float=1.0
    use_actnorm: bool=False
    disc_conditional: bool=False
    disc_ndf: int=64
    disc_loss: str='hinge'

    def setup(self):
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(replicate=False, net='vgg16')
        # self.discriminator = nn.scan(
        #     NLayerDiscriminator, 
        #     variable_broadcast="params",
        #     split_rngs={"params": False}
        # )(
        #     input_nc=self.disc_in_channels,
        #     ndf=self.disc_ndf,
        #     n_layers=self.disc_num_layers,
        #     use_actnorm=self.use_actnorm,
        # )
        self.discriminator = NLayerDiscriminator(
            input_nc=self.disc_in_channels,
            ndf=self.disc_ndf,
            n_layers=self.disc_num_layers,
            use_actnorm=self.use_actnorm,
        )
        # self.discriminator = 
        self.logvar_dense = nn.Dense(1, use_bias=False, kernel_init=nn.initializers.zeros)
        if self.disc_loss == "hinge":
            self.disc_loss_fn = hinge_d_loss
        elif self.disc_loss == "vanilla":
            self.disc_loss_fn = vanilla_d_loss


        def nll_loss_fn(inputs, reconstructions, g_params=None):
            rec_loss = jnp.absolute(inputs - reconstructions)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss += self.perceptual_weight * p_loss
            dummy_input = jnp.array([1.])
            logvar = self.logvar_dense(dummy_input)
            nll_loss = rec_loss / jnp.exp(logvar) + logvar
            nll_loss = jnp.sum(nll_loss) / nll_loss.shape[0]
            return nll_loss
        
        def generator_d_loss_fn(reconstructions, cond=None, g_params=None, discriminator_model=self.discriminator):
            if cond is None:
                assert not self.disc_conditional
                d_inputs = reconstructions
            else:
                assert self.disc_conditional
                d_inputs = jnp.concatenate((reconstructions, cond), axis=-1)
            d_loss = discriminator_model(d_inputs)
            d_loss_mean = -jnp.mean(d_loss)
            return d_loss_mean
        
        # self, input, reconstructions, *g_params*
        self.nll_loss_and_grad = jax.value_and_grad(nll_loss_fn, argnums=2)
        # self.nll_loss = nll_loss_fn
        # self, reconstructions, cond, *g_params*
        self.generator_d_loss_and_grad = jax.value_and_grad(generator_d_loss_fn, argnums=2)
        # self.generator_d_loss = generator_d_loss_fn

    
    def calculate_adaptive_weight(self, nll_grads, g_grads):
        nll_grads = nll_grads['decoder_model']['conv_out']['kernel']
        g_grads = g_grads['decoder_model']['conv_out']['kernel']
        d_weight = jnp.linalg.norm(nll_grads) / (jnp.linalg.norm(g_grads) + 1e-4)
        d_weight = jnp.clip(d_weight, 0.0, 1e4)
        d_weight = d_weight * self.disc_weight
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        weight = jnp.where(global_step < threshold, value, weight)
        return weight
    
    def discriminator_d_loss(self, inputs, reconstructions, cond=None):
        if cond is None:
            assert not self.disc_conditional
            d_real_inputs = inputs
            d_fake_inputs = reconstructions
        else:
            assert self.disc_conditional
            d_real_inputs = jnp.concatenate((inputs, cond), axis=-1)
            d_fake_inputs = jnp.concatenate((reconstructions, cond), axis=-1)
        d_real_loss = self.discriminator(d_real_inputs)
        d_fake_loss = self.discriminator(d_fake_inputs)
        d_loss_mean = self.disc_loss_fn(d_real_loss, d_fake_loss)
        return d_loss_mean, d_real_loss, d_fake_loss

    def generator_loss(self, nll_loss, gan_loss, nll_grad, g_grad, posteriors, global_step=0, split='train', weights=None):
        weighted_nll_loss = nll_loss

        if weights is not None:
            weighted_nll_loss *= weights
        kl_loss = posteriors.kl()
        kl_loss = jnp.sum(kl_loss) / kl_loss.shape[0]
        if self.disc_factor > 0.0:
            gan_weight = self.calculate_adaptive_weight(nll_grad, g_grad)
        else:
            gan_weight = jnp.array(0.0)
        disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
        loss = weighted_nll_loss + self.kl_weight * kl_loss + gan_weight * disc_factor * gan_loss
        log = {
                "{}/total_loss".format(split): jnp.mean(loss), 
                # "{}/logvar".format(split): logvar,
                "{}/kl_loss".format(split): jnp.mean(kl_loss), 
                "{}/nll_loss".format(split): jnp.mean(nll_loss),
                "{}/d_weight".format(split): gan_weight,
                "{}/disc_factor".format(split): disc_factor,
                "{}/g_loss".format(split): jnp.mean(gan_loss),
            }
        return loss, log
    
    def discriminator_loss(self, inputs, reconstructions, global_step=0, split='train'):
        disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
        disc_loss, d_real_logits, d_fake_logits = self.discriminator_d_loss(inputs, reconstructions)
        disc_loss = disc_loss * disc_factor
        log = {
                "{}/disc_loss".format(split): jnp.mean(disc_loss),
                "{}/logits_real".format(split): jnp.mean(d_real_logits),
                "{}/logits_fake".format(split): jnp.mean(d_fake_logits)
            }
        return disc_loss, log

    def __call__(self, inputs, reconstructions, posteriors_kl, optimizer_idx,
                 global_step, cond=None, split='train', weights=None, g_params=None):
        nll_loss, nll_grad = self.nll_loss_and_grad(inputs, reconstructions, g_params)
        # nll_loss = self.nll_loss(inputs, reconstructions, g_params)
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = nll_loss * weights

        # weighted_nll_loss = jnp.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # nll_loss = jnp.sum(nll_loss) / nll_loss.shape[0]
        # kl_loss = posteriors.kl()
        kl_loss = posteriors_kl
        kl_loss = jnp.sum(kl_loss) / kl_loss.shape[0]

        # GAN update
        def generator_process(discriminator_model):
            g_loss, g_grad = self.generator_d_loss_and_grad(reconstructions, cond, g_params, discriminator_model)
            # g_loss = self.generator_d_loss(reconstructions, cond, g_params)
            if self.disc_factor > 0.0 and optimizer_idx == 0:
                d_weight = self.calculate_adaptive_weight(nll_grad, g_grad)
            else:
                d_weight = jnp.array(1.0)
            disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {
                "{}/total_loss".format(split): jnp.mean(loss), 
                "{}/kl_loss".format(split): jnp.mean(kl_loss), 
                "{}/nll_loss".format(split): jnp.mean(nll_loss),
                "{}/d_weight".format(split): d_weight,
                "{}/disc_factor".format(split): disc_factor,
                "{}/g_loss".format(split): jnp.mean(g_loss),
                # "{}/logvar".format(split): self.logvar_dense.variable(),
                # "{}/rec_loss".format(split): jnp.mean(rec_loss),
                "{}/disc_loss".format(split): 0.0,
                "{}/logits_real".format(split): 0.0,
                "{}/logits_fake".format(split): 0.0
            }
            return loss, log
        
        def discriminator_process(discriminator_model):
            if cond is None:
                d_inputs = inputs
                d_reconstructions = reconstructions
            else:
                d_inputs = jnp.concatenate((inputs, cond), axis=-1)
                d_reconstructions = jnp.concatenate((reconstructions, cond), axis=-1)

            # logits_real = self.discriminator(d_inputs)
            # logits_fake = self.discriminator(d_reconstructions)
            logits_real = discriminator_model(d_inputs)
            logits_fake = discriminator_model(d_reconstructions)

            disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)

            log = {
                "{}/total_loss".format(split): 0.0, 
                "{}/kl_loss".format(split): 0.0, 
                "{}/nll_loss".format(split): 0.0,
                "{}/d_weight".format(split): 0.0,
                "{}/disc_factor".format(split): 0.0,
                "{}/g_loss".format(split): 0.0,
                ####
                "{}/disc_loss".format(split): jnp.mean(d_loss),
                "{}/logits_real".format(split): jnp.mean(logits_real),
                "{}/logits_fake".format(split): jnp.mean(logits_fake)
            }
            return d_loss, log
        
        loss, log = nn.cond(optimizer_idx == 0, generator_process, discriminator_process, self.discriminator)
        return loss, log


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    input_nc: int = 3
    ndf: int = 64
    n_layers: int = 3
    # Actnorm is used in taming github, I will use just batchNorm instead
    use_actnorm: bool = False
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        kw = 4
        padw = 1
        # norm_layer = nn.BatchNorm
        x = nn.Conv(
            self.ndf, 
            (kw, kw), 
            strides=2, 
            padding=padw, 
            kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.leaky_relu(x, 0.2)
        for n in range(1, self.n_layers):
            nf_mult = min(2 ** n, 8)

            x = nn.Conv(
                self.ndf * nf_mult, 
                (kw, kw), 
                strides=2, 
                padding=padw, 
                use_bias=self.use_bias,
                kernel_init=nn.initializers.normal(0.02))(x)
            # x = nn.BatchNorm(use_running_average=not train)(x)
            # Instead of using Batchnorm, I use Layernorm for implementation simplicity
            x = nn.GroupNorm(num_groups=32)(x)
            x = nn.leaky_relu(x, 0.2)
        
        nf_mult = min(2 ** self.n_layers, 8)
        x = nn.Conv(
            self.ndf * nf_mult, 
            (kw, kw), 
            strides=2, 
            padding=padw, 
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(0.02))(x)
        # x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.leaky_relu(x, 0.2)

        x = nn.Conv(
            1, (kw, kw), strides=1, padding=padw,
            )(x)
        return x




