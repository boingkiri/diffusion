# Configuration setting with hydra

This file explains the overall options to control the experiment for diffusion training and sampling.

Last update is Aug 19, 2023, authored by Jeongjun Lee, KAIST AI.

## Prerequisite

* Highly recommand to get familiar with [hydra](https://hydra.cc/docs/intro)


## TL;DR

Experiment setting can be changed by this file with Hydra.

You can reproduce the famous diffusion experiment with pre-defined configuration in `examples` directory.

> $ python main.py --config examples/ddpm

The experiment setting correspondings to the conducted experiments can be found in "experiments_name/{experiments_name}/config.yaml".

 

## Structure

The detailed options are following

### framework
> 
> Set training and sampling method with ready-to-use predefined options 
> 
> * Available option
>   * ldm 
>   * diffusion/ddim
>   * diffusion/ddpm, 
>   * diffusion/improved_ddpm,
>   * diffusion/edm, 
>   * diffusion/cm, 
>   * diffusion/cm_diffusion
>
> You can also modify some options in the framework

### model
> 
> Set model architecture,
> 
> * Available option
>   * autoencoder/autoencoder_kl, 
>   * discriminator/discriminator_kl, \
>   * diffusion/dit, 
>   * diffusion/unet, 
>   * diffusion/unetpp, 
>   * diffusion/unetpp_cm

### ema
>  Set ema method,
>  * Available option: ema, ema_edm

### dataset:
>  Set dataset,
>  Available option: cifar10

### exp:
>  Set experiment setting,
>  Available option: exp_setting, exp_setting_cm
