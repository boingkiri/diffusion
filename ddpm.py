from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp

from flax import linen as nn

from flax.training import train_state

class DDPM():
    def __init__(self, eps_model: nn.Module, n_steps: int):
        self.eps_model = eps_model

        self.n_steps = n_steps

        self.rand_key = jax.random.PRNGKey(42)
        self.beta = jnp.linspace(0.0001, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha, axis=0)
        self.sqrt_alpha = jnp.sqrt(self.alpha)
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1 - self.alpha_bar)


        def loss_fn(params, perturbed_data, time, real, dropout_key):
            pred_noise = self.eps_model.apply(
                {'params': params}, x=perturbed_data, t=time, train=True, rngs={'dropout': dropout_key})
            loss = jnp.mean((pred_noise - real) ** 2)
            return loss
        
        def update_grad(state:train_state.TrainState, grad):
            return state.apply_gradients(grads=grad)
        
        def p_sample_jit(params, perturbed_data, time, dropout_key, normal_key):
            pred_noise = self.eps_model.apply(
                {'params': params}, x=perturbed_data, t=time, train=False, rngs={'dropout': dropout_key})

            beta = jnp.take(self.beta, time)
            sqrt_one_minus_alpha_bar = jnp.take(self.sqrt_one_minus_alpha_bar, time)
            eps_coef = beta / sqrt_one_minus_alpha_bar
            eps_coef = eps_coef[:, None, None, None]

            # alpha = jnp.take(self.alpha, time)
            sqrt_alpha = jnp.take(self.sqrt_alpha, time)
            sqrt_alpha = sqrt_alpha[:, None, None, None]
            

            mean = (perturbed_data - eps_coef * pred_noise) / sqrt_alpha

            # var = jnp.take(self.sigma2, time, -1)
            # var = jnp.take(self.sigma2, time)
            var = beta[:, None, None, None]
            eps = jax.random.normal(normal_key, perturbed_data.shape)

            return mean + (var ** 0.5) * eps
        
        self.loss_fn = jax.jit(loss_fn)
        self.grad_fn = jax.jit(jax.value_and_grad(self.loss_fn))
        self.update_grad = jax.jit(update_grad)
        self.p_sample_jit = jax.jit(p_sample_jit)

    def q_xt_x0(self, x0, t):
        mean_coeff = jnp.take(self.sqrt_alpha_bar, t)
        mean = mean_coeff[:, None, None, None] * x0

        std = jnp.take(self.sqrt_one_minus_alpha_bar, t)
        std = std[:, None, None, None]
        return mean, std
    
    def q_sample(self, x0, t, eps):
        # Sample from q(x_t|x_0)

        if eps is None:
            # TODO: need to check
            key, normal_key = jax.random.split(self.rand_key)
            self.rand_key = key
            eps = jax.random.normal(normal_key, x0.shape)
        mean, std = self.q_xt_x0(x0, t)
        return mean + std * eps, eps
    
    def p_sample(self, param, xt, t):
        # Sample from p_theta(x_{t-1}|x_t)
        t = jnp.array(t)
        t = jnp.repeat(t, xt.shape[0])

        key, normal_key, dropout_key = jax.random.split(self.rand_key, 3)
        self.rand_key = key
        return self.p_sample_jit(param, xt, t, normal_key, dropout_key)

    def learning_from(self, state, x0):
        batch_size = x0.shape[0]
        key, int_key, normal_key, dropout_key = jax.random.split(self.rand_key, 4)
        self.rand_key = key

        noise = jax.random.normal(normal_key, x0.shape)
        t = jax.random.randint(int_key, (batch_size, ), 0, self.n_steps)
        xt, noise = self.q_sample(x0, t, eps=noise)
        
        loss, grad = self.grad_fn(state.params, xt, t, noise, dropout_key)
        new_state = self.update_grad(state, grad)
        return loss, new_state

        
    


