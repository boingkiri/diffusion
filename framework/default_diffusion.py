from typing import TypedDict

from abc import *
from omegaconf import DictConfig

from model.modelContainer import ModelContainer

class DefaultModel(metaclass=ABCMeta):

    model_container: ModelContainer

    @abstractmethod
    def fit(self, x, cond=None):
        pass
    
    @abstractmethod
    def sampling(self, num_image, img_size=None):
        pass

    @abstractmethod
    def get_model_state(self) -> dict:
        pass
