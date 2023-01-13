# Diffusion implementation

This repository is for practicing implementation of diffusion model with JAX family framework. The code is worked in environment of TPU v3-8.

# How to use
You can manipulate the overall hyperparameter in `conig.yml` file.

If you want to train the model, `training.sh` will take care of all of the training processing. Or, if you want to do sampling with trained model, `sampling.sh` will be your choice.


# Citation

  @article{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models},
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      journal={arXiv preprint arxiv:2006.11239}
  }

  @misc{https://doi.org/10.48550/arxiv.2112.10752,
    doi = {10.48550/ARXIV.2112.10752},
    url = {https://arxiv.org/abs/2112.10752},
    author = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn},
    keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {High-Resolution Image Synthesis with Latent Diffusion Models},
    publisher = {arXiv},
    year = {2021},
    copyright = {arXiv.org perpetual, non-exclusive license}
  }

