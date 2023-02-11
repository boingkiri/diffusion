# Diffusion implementation

This repository is for practicing implementation of diffusion model with JAX family framework. The code is worked in environment of TPU v3-8.

# How to use
You can manipulate the overall hyperparameter in yml files in conig directory.
'ddpm.yml' file effects to make a model and to train it with DDPM method. 
'ldm_kl.yml' file and 'ldm_vq.yml' file do the same thing but it is operated with Latent Diffusion methods. The two files contain same information to construct overall training, but the regularization term in autoencoder training. 'ldm_kl.yml' makes encoder's latent space similar to diagonal gaussian distribution, and 'ldm_vq.yml' maps encoder's latent space to the vector quantized embedding value. For more specific information, please read [latent diffusion model paper](https://arxiv.org/abs/2112.10752)

This documentation 
If you want to train the model, `ddpm_training.sh`, for DDPM manners, or `ldm_kl_training.sh`, `ldm_vq_training.sh` for Latent Diffusion Model manners, will take care of all of the training processing. 
Or, if you want to do sampling with trained model, `sampling.sh` will be your choice.


# Citation
~~~
  @article{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models},
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      journal={arXiv preprint arxiv:2006.11239}
  }
~~~
~~~
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
~~~
~~~
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
~~~
