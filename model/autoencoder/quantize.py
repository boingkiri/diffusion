import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    Original github: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
    This model is ported from pytorch to jax, flax.

    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    n_e: int
    e_dim: int
    beta: float
    remap: str = None
    unknown_index: str = "random"
    sane_index_shape: bool = False
    legacy: bool = True

    def setup(self):
        def zero_centered_uniform(key, shape, dtype=jnp.float_):
            scale = 1.0 / self.n_e
            data = jax.random.uniform(key, shape, minval=-scale, maxval=scale)
            return data
        self.embedding = nn.Embed(self.n_e, self.e_dim, embedding_init=zero_centered_uniform)

    def __call__(self, z): # z: b h w c
        z_flatten = jnp.reshape(z, (-1, self.e_dim)) # b*h*w*c/3, 3
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        embedding = self.embedding.embedding
        d = jnp.sum(z_flatten ** 2, axis=1, keepdims=True) + \
            jnp.sum(embedding ** 2, axis=1) - \
            2 * jnp.einsum('bd, dn -> bn', z_flatten, rearrange(embedding, "n d -> d n"))

        min_encoding_indices = jnp.argmin(d, axis=1)
        z_q = self.embedding(min_encoding_indices)
        z_q = jnp.reshape(z_q, z.shape)

        perplexity = None
        min_encodings = None
        loss = jnp.mean((jax.lax.stop_gradient(z_q) - z) ** 2) + \
                jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)
        
        # Preserve gradients
        z_q = z + jax.lax.stop_gradient(z_q - z)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
