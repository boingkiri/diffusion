import sys
sys.path.append("..")

import jax
import jax.numpy as jnp
import jax.random as random

import hydra
import os
from omegaconf import OmegaConf, DictConfig
from functools import partial

from framework.unifying_framework import UnifyingFramework 
from framework.diffusion.consistency_framework import CMFramework 
from utils.fid_utils import FIDUtils

def stochastic_iterative_sampler(
    rng,
    num_sampling,
    distiller: CMFramework,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)

    params = distiller.model_state.params_ema
    
    # Sampling x from sampler
    input_shape = (jax.local_device_count(), num_sampling // jax.local_device_count(), 32, 32, 3)
    rng, sampling_key = jax.random.split(rng, 2)
    x = jax.random.normal(sampling_key, input_shape) * t_max

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho

        # x0 = distiller(x, t * s_in)
        rng, p_sample_key = jax.random.split(rng, 2)
        p_sample_key = jax.random.split(sampling_key, jax.local_device_count())

        x0 = distiller.p_sample_jit(params, x, p_sample_key, t)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = jnp.clip(next_t, t_min, t_max)

        rng, normal_rng = jax.random.split(rng, 2)
        x = x0 + jax.random.normal(normal_rng, x.shape) * jnp.sqrt(next_t**2 - t_min**2)

    return x

def get_fid(rng, sampling_fn, fid_obj: FIDUtils, p, begin=(0,), end=(17, ), sampling_dir="."):
    total_size = 50000
    batch_size = 128
    current_sampling_num = 0

    ts = begin + (p,) + end
    tmp_dir = os.path.join(sampling_dir, f"{p}")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    while current_sampling_num < total_size:
        effective_sampling = batch_size \
            if (total_size - current_sampling_num) // batch_size != 0 \
            else total_size - current_sampling_num

        rng, sampling_rng = jax.random.split(rng, 2)
        x0 = sampling_fn(rng=sampling_rng, num_sampling=effective_sampling, ts=ts)
        x0 = x0.reshape(-1, *x0.shape[-3:])
        fid_obj.fs_utils.save_images_to_dir(x0, tmp_dir, current_sampling_num)
        current_sampling_num += effective_sampling
    
    fid = fid_obj.calculate_fid(tmp_dir)
    return fid

def ternery_search(rng, sampling_fn, fid_obj, before, after):
    right = after[0]
    left = before[-1]
    while right - left >= 3:
        m1 = int(left + (right - left) / 3.0)
        m2 = int(right - (right - left) / 3.0)
        f1 = get_fid(rng, sampling_fn, fid_obj, m1, before, after)

        # logger.log(f"fid at m1 = {m1} is {f1}, IS is {is1}")
        print(f"fid at m1 = {m1} is {f1}")
        f2 = get_fid(rng, sampling_fn, fid_obj, m2, before, after)

        print(f"fid at m2 = {m2} is {f2}")

        if f1 < f2:
            right = m2
        else:
            left = m1

        print(f"new interval is [{left}, {right}]")

    if right == left:
        p = right
    elif right - left == 1:
        f1 = get_fid(rng, sampling_fn, fid_obj, left, before, after)
        f2 = get_fid(rng, sampling_fn, fid_obj, right, before, after)
        p = m1 if f1 < f2 else m2
    elif right - left == 2:
        mid = left + 1
        f1 = get_fid(rng, sampling_fn, fid_obj, left, before, after)
        f2 = get_fid(rng, sampling_fn, fid_obj, right, before, after)
        fmid = get_fid(rng, sampling_fn, fid_obj, mid, before, after)

        print(f"fmid at mid = {mid} is {fmid}")

        if fmid < f1 and fmid < f2:
            p = mid
        elif f1 < f2:
            p = m1
        else:
            p = m2

    return p
    

@hydra.main(config_path="../configs", config_name="config")
def find_optimal_p(config: DictConfig):
    rng = random.PRNGKey(config.rand_seed)
    model_type = config.type

    # Assume that we only use CIFAR10
    sigma_max = config.framework.diffusion.sigma_max
    sigma_min = config.framework.diffusion.sigma_min
    config.exp.exp_dir = os.path.join("..", config.exp.exp_dir)
    config.framework.diffusion.is_distillation = False

    rho = config.framework.diffusion.rho
    steps = 18
    begin = (0,)
    end = (steps - 1,)
    
    print("-------------------Config Setting---------------------")
    print(OmegaConf.to_yaml(config))
    print("------------------------------------------------------")
    diffusion_framework = UnifyingFramework(model_type, config, rng)
    fid_utils = FIDUtils(config)
    sampler = partial(stochastic_iterative_sampler, # need rng, num_sampling, ts
            distiller=diffusion_framework.framework,
            sigmas=None,
            generator=None,
            progress=False,
            callback=None,
            t_min=sigma_min,
            t_max=sigma_max,
            rho=rho,
            steps=steps)
    
    # Start ternary search
    optimal_p = ternery_search(rng, sampler, fid_utils, begin, end)
    optimal_fid = get_fid(rng, sampler, fid_utils, optimal_p, begin, end)
    print(f"Optimal fid at {optimal_p} = {optimal_fid}")
    


if __name__=="__main__":
    find_optimal_p()