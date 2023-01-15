from typing import TypedDict

class DefaultDiffusion():
    def __init__(self, config, random_rng) -> None:
        pass

    def fit(self, x, cond=None):
        # pass
        NotImplementedError("fit: Diffusion model should implement this method.")

    def get_model_state(self) -> TypedDict:
        # pass
        NotImplementedError("get_model_state: Diffusion model should implement this method.")