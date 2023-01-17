import jax
import jax.numpy as jnp

class DiagonalGaussianDistribution():
    def __init__(self, parameters, rand_rng, deterministic=False):
        self.parameters = parameters # jnp array
        self.rand_rng = rand_rng
        self.mean, self.logvar = jnp.split(parameters, 2, dim=-1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)
    
    def sample(self):
        sample_rng, self.rand_rng = jax.random.split(self.rand_rng, 2)
        x = self.mean + self.std * jax.random.normal(sample_rng, self.mean.shape)
        return x
    
    def kl(self, other=None):
        if self.deterministic:
            return jnp.array([0.])
        elif other is None:
            return 0.5 * jnp.sum(
                jnp.power(self.mean, 2) + self.var - 1.0 - self.logvar, 
                axis=[1, 2, 3]
            )
        else:
            return 0.5 * jnp.sum(
                jnp.power(self.mean - other.mean, 2) / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar, 
                axis=[1, 2, 3]
            )
    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return jnp.array([0.])
        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(
            logtwopi + self.logvar + jnp.power(sample - self.mean, 2) / self.var,
            dim=dims
        )
    
    def mode(self):
        return self.mean
