import lpips_jax

import jax
import jax.numpy as jnp
import flax.linen as nn


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
    def __init__(
        self, 
        disc_start, 
        # codebook_weight=1.0, 
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss='hinge'
    ) -> None:

        self.logvar_init = logvar_init
        self.kl_weight = kl_weight

        # self.codebook_weight = codebook_weight
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        self.perceptual_loss = lpips_jax.LPIPSEvaluator(net='vgg16')
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        )

        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            grad_fn = jax.grad(last_layer)
        else:
            grad_fn = jax.grad(self.last_layer[0])
        nll_grads = grad_fn(nll_loss)[0]
        g_grads = grad_fn(g_loss)[0]

        d_weight = jnp.linalg.norm(nll_grads) / (jnp.linalg.norm(g_grads) + 1e-4)
        d_weight = jnp.clip(d_weight, 0.0, 1e4)
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def __call__(self, inputs, reconstructions, posteriors, optimizer_idx,
                 global_step, last_layer=None, cond=None, split='train',
                 weights=None):
        rec_loss = jnp.absolute(inputs - reconstructions)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss += self.perceptual_weight * p_loss
        dummy_input = 1.
        logvar = nn.Dense(1, use_bias=False, kernel_init=nn.initializers.zeros)(dummy_input)
        nll_loss = rec_loss / jnp.exp(logvar) + logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = nll_loss * weights

        weighted_nll_loss = jnp.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = jnp.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = jnp.sum(kl_loss) / kl_loss.shape[0]

        # GAN update
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                d_inputs = reconstructions
            else:
                assert self.disc_conditional
                d_inputs = jnp.concatenate((reconstructions, cond), axis=-1)
            logits_fake = self.discriminator(d_inputs)
            g_loss = -jnp.mean(logits_fake)

            if self.disc_factor > 0.0:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            else:
                d_weight = jnp.array(0.0)
            disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {
                "{}/total_loss".format(split): jnp.mean(loss), 
                "{}/logvar".format(split): logvar,
                "{}/kl_loss".format(split): jnp.mean(kl_loss), 
                "{}/nll_loss".format(split): jnp.mean(nll_loss),
                "{}/rec_loss".format(split): jnp.mean(rec_loss),
                "{}/d_weight".format(split): d_weight,
                "{}/disc_factor".format(split): disc_factor,
                "{}/g_loss".format(split): jnp.mean(g_loss),
            }
            return loss, log

        else:
            if cond is None:
                d_inputs = inputs
                d_reconstructions = reconstructions
            else:
                d_inputs = jnp.concatenate((inputs, cond), axis=-1)
                d_reconstructions = jnp.concatenate((reconstructions, cond), axis=-1)

            logits_real = self.discriminator(d_inputs)
            logits_fake = self.discriminator(d_reconstructions)

            disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): jnp.mean(d_loss),
                "{}/logits_real".format(split): jnp.mean(logits_real),
                "{}/logits_fake".format(split): jnp.mean(logits_fake)
            }
            return d_loss, log


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    input_nc = 3
    ndf = 64
    n_layers = 3
    # Actnorm is used in taming github, I will use just batchNorm instead
    use_actnorm = False
    use_bias = use_actnorm

    @nn.compact
    def call(self, x):
        kw = 4
        padw = 1
        norm_layer = nn.BatchNorm
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
                bias=self.use_bias,
                kernel_init=nn.initializers.normal(0.02))(x)
            x = norm_layer()(x)
            x = nn.leaky_relu(x, 0.2)
        
        nf_mult = min(2 ** self.n_layers, 8)
        x = nn.Conv(
            self.ndf * nf_mult, 
            (kw, kw), 
            strides=2, 
            padding=padw, 
            bias=self.use_bias,
            kernel_init=nn.initializers.normal(0.02))(x)
        x = norm_layer()(x)
        x = nn.leaky_relu(x, 0.2)

        x = nn.Conv(
            1, (kw, kw), strides=1, padding=padw,
            )(x)
        return x




