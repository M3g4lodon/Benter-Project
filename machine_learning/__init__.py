from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class WinningModel(metaclass=ABCMeta):
    @abstractmethod
    def get_n_horses_model(self, n_horses: int):
        pass

    @abstractmethod
    def predict(self, x: np.array):
        """A common method to predict probabilities on given races (with n_horses)"""
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, trainable: bool) -> "WinningModel":
        pass


class ModelNotCreatedOnce(Exception):
    pass