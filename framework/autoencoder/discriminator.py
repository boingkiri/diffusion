import lpips_jax

import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial

def hinge_d_loss(logits_real, logits_fake, weights=1.0):
    # loss_real = jnp.mean(nn.relu(1. - logits_real))
    # loss_fake = jnp.mean(nn.relu(1. + logits_fake))
    loss_real = jnp.mean(nn.relu(1. - logits_real), axis=[1, 2, 3])
    loss_fake = jnp.mean(nn.relu(1. - logits_fake), axis=[1, 2, 3])
    loss_real = jnp.sum(weights * loss_real) / weights
    loss_fake = jnp.sum(weights * loss_fake) / weights
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        jnp.mean(nn.softplus(-logits_real)) +
        jnp.mean(nn.softplus(logits_fake))
    )
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    weight = jnp.where(global_step < threshold, value, weight)
    return weight

def l1(x, y):
    return jnp.absolute(x - y)

def l2(x, y):
    return jnp.power((x - y), 2)

def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = jax.nn.one_hot(predicted_indices, n_embed)
    encodings = encodings.astype(float).reshape(-1, n_embed)
    avg_probs = jnp.mean(encodings, 0)
    perplexity = jnp.exp(-(avg_probs * jnp.log(avg_probs + 1e-10)).sum())
    cluster_use = jnp.sum(avg_probs > 0)
    return perplexity, cluster_use

class LPIPSwitchDiscriminator(nn.Module):
    disc_start: int
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
    pixel_loss: str='l1'

    def setup(self):
        self.perceptual_loss = lpips_jax.LPIPSEvaluator(replicate=False, net='vgg16')
        self.discriminator = NLayerDiscriminator(
            input_nc=self.disc_in_channels,
            ndf=self.disc_ndf,
            n_layers=self.disc_num_layers,
            use_actnorm=self.use_actnorm,
        )
        
        if self.disc_loss == "hinge":
            self.disc_loss_fn = hinge_d_loss
        elif self.disc_loss == "vanilla":
            self.disc_loss_fn = vanilla_d_loss
        else:
            NotImplementedError("discriminator loss function should be one of the 'hinge' and 'vanila'")
        
        if self.pixel_loss == "l1":
            self.pixel_loss_fn = l1
        elif self.pixel_loss == "l2":
            self.pixel_loss_fn = l2
        else:
            NotImplementedError("pixel loss function should be one of the 'l1' and 'l2'")

        def nll_loss_fn(inputs, reconstructions, g_params=None):
            rec_loss = self.pixel_loss_fn(inputs, reconstructions)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss += self.perceptual_weight * p_loss
            nll_loss = rec_loss
            # nll_loss = jnp.sum(nll_loss) / nll_loss.shape[0]
            nll_loss = jnp.mean(nll_loss)
            return nll_loss
        
        def generator_d_loss_fn(reconstructions, cond=None, g_params=None, discriminator_model=self.discriminator):
            if cond is None:
                assert not self.disc_conditional
                d_inputs = reconstructions
            else:
                assert self.disc_conditional
                d_inputs = jnp.concatenate((reconstructions, cond), axis=-1)
            d_loss = discriminator_model(d_inputs)
            d_loss_mean = -jnp.mean(d_loss) # Discriminator: 0 - fake, 1 - true 
            return d_loss_mean
        
        # self, input, reconstructions, *g_params*
        self.nll_loss_and_grad = jax.value_and_grad(nll_loss_fn, argnums=2)
        # self, reconstructions, cond, *g_params*
        self.generator_d_loss_and_grad = jax.value_and_grad(generator_d_loss_fn, argnums=2)

    
    def calculate_adaptive_weight(self, nll_grads, g_grads):
        nll_grads = nll_grads['decoder_model']['conv_out']['kernel']
        g_grads = g_grads['decoder_model']['conv_out']['kernel']
        d_weight = jnp.linalg.norm(nll_grads) / (jnp.linalg.norm(g_grads) + 1e-4)
        d_weight = jnp.clip(d_weight, 0.0, 1e4)
        d_weight = d_weight * self.disc_weight
        return d_weight

    def regularization_loss(self, loss):
        NotImplementedError("LPIPSwitchDiscriminator should implement as KL or VQ.")

    def __call__(self, inputs, reconstructions, regularization_loss, optimizer_idx,
                 global_step, cond=None, split='train', weights=None, g_params=None):
        nll_loss, nll_grad = self.nll_loss_and_grad(inputs, reconstructions, g_params)
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = nll_loss * weights

        # GAN update
        def generator_process(discriminator_model):
            g_loss, g_grad = self.generator_d_loss_and_grad(reconstructions, cond, g_params, discriminator_model)
            if self.disc_factor > 0.0 and optimizer_idx == 0:
                d_weight = self.calculate_adaptive_weight(nll_grad, g_grad)
            else:
                d_weight = jnp.array(1.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            loss = weighted_nll_loss + self.regularization_loss(regularization_loss) + d_weight * disc_factor * g_loss
            log = {
                "{}/total_loss".format(split): jnp.mean(loss), 
                "{}/regularization_loss".format(split): jnp.mean(regularization_loss),  # kl_loss or vq_loss
                "{}/nll_loss".format(split): jnp.mean(nll_loss),
                "{}/d_weight".format(split): d_weight,
                "{}/disc_factor".format(split): disc_factor,
                "{}/g_loss".format(split): jnp.mean(g_loss),
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

            logits_real = discriminator_model(d_inputs)
            logits_fake = discriminator_model(d_reconstructions)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)
            log = {
                "{}/total_loss".format(split): 0.0, 
                "{}/regularization_loss".format(split): 0.0, 
                "{}/nll_loss".format(split): 0.0,
                "{}/d_weight".format(split): 0.0,
                "{}/disc_factor".format(split): 0.0,
                "{}/g_loss".format(split): 0.0,
                "{}/disc_loss".format(split): jnp.mean(d_loss),
                "{}/logits_real".format(split): jnp.mean(logits_real),
                "{}/logits_fake".format(split): jnp.mean(logits_fake)
            }
            return d_loss, log
        
        loss, log = nn.cond(optimizer_idx == 0, generator_process, discriminator_process, self.discriminator)
        return loss, log



class LPIPSwithDiscriminator_KL(LPIPSwitchDiscriminator):
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
        super().setup()
        self.logvar_dense = nn.Dense(1, use_bias=False, kernel_init=nn.initializers.zeros)

        def nll_loss_fn(inputs, reconstructions, g_params=None):
            # rec_loss = jnp.absolute(inputs - reconstructions)
            rec_loss = self.pixel_loss_fn(inputs, reconstructions)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss += self.perceptual_weight * p_loss
            dummy_input = jnp.array([1.])
            logvar = self.logvar_dense(dummy_input)
            nll_loss = rec_loss / jnp.exp(logvar) + logvar
            nll_loss = jnp.sum(nll_loss) / nll_loss.shape[0]
            return nll_loss
        
        self.nll_loss_and_grad = jax.value_and_grad(nll_loss_fn, argnums=2)
    
    def regularization_loss(self, loss):
        # NotImplementedError("LPIPSwitchDiscriminator should implement as KL or VQ.")
        return jnp.mean(loss) * self.kl_weight

    # In this case, regularization loss is kl divergence of posterior.
    def __call__(self, inputs, reconstructions, posteriors_kl, optimizer_idx,
                 global_step, cond=None, split='train', weights=None, g_params=None):
        loss, log = super().__call__(inputs, reconstructions, posteriors_kl, optimizer_idx, global_step, 
                            cond, split, weights, g_params)
        log[f'{split}/kl_loss'] = log[f'{split}/regularization_loss']
        log.pop(f'{split}/regularization_loss', None)
        return loss, log
        

class LPIPSwithDiscriminator_VQ(LPIPSwitchDiscriminator):
    disc_start: int
    codebook_weight: float=1.0
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
    n_classes: int= None

    def setup(self):
        super().setup()
    
    def regularization_loss(self, loss):
        return jnp.mean(loss) * self.codebook_weight

    # In this case, regularization loss is codebook loss.
    def __call__(self, inputs, reconstructions, codebook_loss, optimizer_idx,
                 global_step, cond=None, split='train', weights=None, g_params=None, predicted_indices=None):
        loss, log = super().__call__(inputs, reconstructions, codebook_loss, optimizer_idx, global_step, 
                            cond, split, weights, g_params)
        log[f'{split}/quant_loss'] = log[f'{split}/regularization_loss']
        log.pop(f'{split}/regularization_loss', None)
        if predicted_indices is not None:
            assert self.n_classes is not None
            perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
            log[f'{split}/perplexity'] = perplexity
            log[f'{split}/cluster_usage'] = cluster_usage
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




