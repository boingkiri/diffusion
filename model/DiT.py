import jax.numpy as jnp
import flax.linen as nn


from model.modules import *

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    dropout_rate: float
    mlp_ratio: float = 4.0
    n_groups: int = 32
    
    @nn.compact
    def __call__(self, x, c, train):
        adaLN_modulation = nn.Sequential(
            [
                nn.silu,
                nn.Dense(6 * self.hidden_size)
            ]
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            tuple(map(
                lambda ary: jnp.expand_dims(ary, axis=1), 
                jnp.split(adaLN_modulation(c), indices_or_sections=6, axis=-1)))
        
        # First Step
        x_skip = x
        x = nn.LayerNorm()(x)
        x = modulate(x, shift_msa, scale_msa)
        x = AttentionDense(self.hidden_size, self.num_heads, self.n_groups)(x)
        x = x_skip + gate_msa * x
        
        # Second Step
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        x_skip = x
        x = nn.LayerNorm()(x)
        x = modulate(x, shift_mlp, scale_mlp)
        x = FeedForwardMLP(mlp_hidden_dim, self.hidden_size, self.dropout_rate)(x, train)
        x = x_skip + gate_mlp * x

        return x

class DiTFinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, c):
        adaLN_modulation = nn.Sequential(
            [
                nn.silu,
                nn.Dense(2 * self.hidden_size)
            ]
        )
        shift, scale = \
            tuple(map(
                lambda ary: jnp.expand_dims(ary, axis=1), 
                jnp.split(adaLN_modulation(c), indices_or_sections=2, axis=-1)
                ))
        x = modulate(nn.LayerNorm()(x), shift, scale)
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels)(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone
    """
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    dropout_rate: float = 0.0
    learn_sigma: bool = True
    def setup(self):
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = TimeEmbed(self.hidden_size // 4, self.hidden_size)
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size, self.class_dropout_prob)

        num_patch = int(self.input_size // self.patch_size)
        self.pos_embed = get_2d_sincos_pos_embed(self.hidden_size, num_patch)

        self.blocks = [DiTBlock(
                self.hidden_size, 
                self.num_heads, 
                mlp_ratio=self.mlp_ratio, 
                dropout_rate=self.dropout_rate) 
                for _ in range(self.depth)]
        self.final_layer = DiTFinalLayer(self.hidden_size, self.patch_size, self.out_channels)
        # TODO: NEED MODULE INITIALIZATION PART (XAVIOR .. ETC)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size ** 2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, w * p, c))
        return imgs
    
    # def __call__(self, x, t, y, train):
    def __call__(self, x, t, train):
        """
        Forward pass of DiT
        x: (N, H, W, C) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (NOT YET IMPLEMENTED)
        """
        x = self.x_embedder(x) + self.pos_embed # (N, T, D) where T = H * W / (patch_size ** 2)
        t = self.t_embedder(t)                  # (N, D)
        # y = self.y_embedder(y, train)           # (N, D)
        # c  = t + y
        c = t
        for layer in self.blocks:
            x = layer(x, c, train)
        # x, c, train = self.blocks(x, c, train)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1 + grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_h, grid_w)
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = jnp.concatenate([jnp.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    embed_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) # (H*W, D/2)
    embed_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) # (H*W, D/2)
    emb = jnp.concatenate([embed_h, embed_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M, )
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega # (D/2, )

    pos = pos.reshape(-1)
    out = jnp.einsum('m, d -> md', pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb