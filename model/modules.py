import jax
import jax.numpy as jnp
import flax.linen as nn

class TimeEmbedding(nn.Module):
    emb_dim: int

    @nn.compact
    def __call__(self, time):
        inv_freq = 1.0 / (
            10000
            ** (jnp.arange(0, self.emb_dim, 2, dtype=float) / self.emb_dim)
        )
        time = jnp.expand_dims(time, -1)
        pos_enc_a = jnp.sin(jnp.repeat(time, self.emb_dim // 2, axis=-1) * inv_freq)
        pos_enc_b = jnp.cos(jnp.repeat(time, self.emb_dim // 2, axis=-1) * inv_freq)
        pos_enc = jnp.concatenate([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc

        # return emb

class TimeEmbed(nn.Module):
    n_channels: int
    hidden_size: int
    activation_type: str = "swish"
    
    @nn.compact
    def __call__(self, t):
        t = TimeEmbedding(self.n_channels)(t)
        t = nn.Dense(self.hidden_size)(t)
        # t = nn.swish(t)
        t = getattr(nn, self.activation_type)(t)
        t = nn.Dense(self.hidden_size)(t)
        return t
    
class ResidualBlock(nn.Module):
    out_channels: int
    n_groups: int = 32
    # n_groups: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, t, train):
        h = nn.GroupNorm(self.n_groups)(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3))(h)

        # Add time embedding value
        if t is not None:
            t = nn.swish(t)
            t_emb = nn.Dense(self.out_channels)(t)
            h += t_emb[:, None, None, :]

        h = nn.GroupNorm(self.n_groups)(h)
        h = nn.swish(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not train)(h)
        h = nn.Conv(self.out_channels, (3, 3))(h)

        if x.shape != h.shape:
            short = nn.Conv(self.out_channels, (1, 1))(x)
            # short = nn.Conv(self.out_channels, (3, 3))(x)
        else:
            short = x
        return h + short

class AttentionDense(nn.Module):
    # This block is not containing skip connection.
    n_channels: int
    n_heads: int = 1
    n_groups: int = 32
    dropout_rate: float = 0.
    
    @nn.compact
    def __call__(self, x): # x: b x y c
        scale = self.n_channels ** -0.5

        batch_size, num_patch, n_channels = x.shape
        head_channels = n_channels // self.n_heads

        # Projection
        qkv = nn.Dense(self.n_heads * head_channels * 3, use_bias=False)(x) # qkv: b x y h*c*3
        qkv = qkv.reshape(batch_size, -1, self.n_heads, 3 * head_channels) # b (x y) h c*3

        # Split as query, key, value
        q, k, v = jnp.split(qkv, 3, axis=-1) # q,k,v = b (x y) h c

        # Scale dot product 
        atten = jnp.einsum('bihd,bjhd->bijh', q, k) * scale

        # Softmax
        atten = nn.softmax(atten, axis=2)

        # Multiply by value
        res = jnp.einsum('bijh,bjhd->bihd', atten, v)
        res = res.reshape(batch_size, num_patch, self.n_heads * head_channels)
        res = nn.Dense(self.n_channels)(res)

        return res


class AttentionConv(nn.Module):
    # This block is not containing skip connection.
    n_channels: int
    n_heads: int = 1
    n_groups: int = 32
    dropout_rate: float = 0.
    
    @nn.compact
    def __call__(self, x): # x: b x y c
        scale = self.n_channels ** -0.5

        batch_size, height, width, n_channels = x.shape
        head_channels = n_channels // self.n_heads
        # Projection
        qkv = nn.Conv(self.n_heads * head_channels * 3, (1, 1), use_bias=False)(x) # qkv: b x y h*c*3
        qkv = qkv.reshape(batch_size, -1, self.n_heads, 3 * head_channels) # b (x y) h c*3

        # Split as query, key, value
        q, k, v = jnp.split(qkv, 3, axis=-1) # q,k,v = b (x y) h c

        # Scale dot product 
        atten = jnp.einsum('bihd,bjhd->bijh', q, k) * scale

        # Softmax
        atten = nn.softmax(atten, axis=2)

        # Multiply by value
        res = jnp.einsum('bijh,bjhd->bihd', atten, v)
        res = res.reshape(batch_size, height, width, self.n_heads * head_channels)
        res = nn.Conv(self.n_channels, (1, 1))(res)

        return res


class AttentionBlock(nn.Module):
    n_channels: int
    n_heads: int = 1
    n_groups: int = 32
    attention_type: str = "conv"
    
    @nn.compact
    def __call__(self, x): # x: b x y c
        # Projection
        x_skip = x
        x = nn.GroupNorm(self.n_groups)(x)
        if self.attention_type == "conv":
            res = AttentionConv(self.n_channels, self.n_heads, self.n_groups)(x)
        elif self.attention_type == "dense":
            res = AttentionDense(self.n_channels, self.n_heads, self.n_groups)(x)

        # skip connection
        res += x_skip

        return res

class UnetDown(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    n_groups: int
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels, n_groups=self.n_groups)(x)
        return x

class UnetUp(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    n_groups: int
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels, n_groups=self.n_groups)(x)
        return x

class UnetMiddle(nn.Module):
    n_channels: int
    dropout_rate: float
    n_groups: int
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        x = AttentionBlock(self.n_channels)(x)
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        return x


class Upsample(nn.Module):
    n_channels: int
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        scale = 2
        x = jax.image.resize(
            x, 
            shape=(B, H * scale, W * scale, C),
            method="nearest")
        x = nn.Conv(self.n_channels, (3, 3))(x)
        return x

class Downsample(nn.Module):
    n_channels: int
    @nn.compact
    def __call__(self, x):
        # B, H, W, C = x.shape
        # x = jnp.reshape(x, (B, H // 2, W // 2, C * 4))
        # x = nn.Conv(self.n_channels, (3, 3), (1, 1))(x)
        x = nn.Conv(self.n_channels, (3, 3), strides=2)(x)
        return x


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
            2 * jnp.einsum('bd, dn -> bn', z_flatten, jnp.transpose(embedding))
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

## For DiT
class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    dropout_prob: float
    
    def token_drop(self, labels, force_drop_ids=None):
        """
        Token drop function for classifier free guidence (although it is not used for now.)
        """
        if force_drop_ids is None:
            rng_val = self.make_rng('CFG') # TODO: the models should contain additional PRNG
            drop_ids = jax.random.uniform(rng_val, labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embed_classes = self.num_classes + 1 if self.dropout_prob > 0 else self.num_classes
        embeddings = nn.Embed(embed_classes, self.hidden_size)(labels)
        return embeddings

class FeedForwardMLP(nn.Module):
    hidden_features: int
    out_features: int
    dropout_rate: float
    act_layer: str = "gelu"
    
    @nn.compact
    def __call__(self, x, train):
        x = nn.Dense(self.hidden_features)(x)
        x = getattr(nn, self.act_layer)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.out_features)(x)
        x = getattr(nn, self.act_layer)(x)
        return x

class PatchEmbed(nn.Module):
    img_size: int
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 128
    norm_layer: bool = False
    flatten: bool = True
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        assert H == W
        assert H == self.img_size
        x = nn.Conv(
            self.embed_dim, 
            (self.patch_size, self.patch_size), 
            self.patch_size,
            use_bias=self.bias)(x)
        if self.flatten:
            x = jnp.reshape(x, (B, -1, self.embed_dim))
        
        if self.norm_layer:
            x = nn.LayerNorm()(x)
        return x
            