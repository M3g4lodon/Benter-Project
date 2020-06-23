import os
import re
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

import joblib
import numpy as np

from constants import SAVED_MODELS_DIR

# TODO add MLP


class AbstractWinningModel(metaclass=ABCMeta):

    _NotFittedModelError: Optional[Exception] = None

    def __init__(self):
        assert self._NotFittedModelError is not None
        self.n_horses_models = {}

    def get_n_horses_model(self, n_horses: int):
        if n_horses not in self.n_horses_models:
            self.n_horses_models[n_horses] = self._create_n_horses_model(
                n_horses=n_horses
            )

        return self.n_horses_models[n_horses]

    @abstractmethod
    def _create_n_horses_model(self, n_horses: int):
        pass

    @abstractmethod
    def predict(self, x: np.array):
        """A common method to predict probabilities on given races (with n_horses)"""

    @abstractmethod
    def save_model(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def load_model(cls) -> "AbstractWinningModel":
        pass


class SequentialMixin:
    def predict(self, x: np.array):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            return model.predict(x=x)
        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError


class FlattenMixin:
    def predict(self, x: np.array):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            return model.predict_proba(
                X=np.reshape(
                    a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F"
                )
            )
        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError


class JoblibPicklerMixin:
    def save_model(self) -> None:
        if self.__class__.__name__ not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.__class__.__name__))
        for n_horse, n_horse_model in self.n_horses_models.items():
            joblib.dump(
                n_horse_model,
                os.path.join(
                    SAVED_MODELS_DIR,
                    self.__class__.__name__,
                    f"{self.__class__.__name__}_{n_horse}.pickle",
                ),
            )

    @classmethod
    def load_model(cls):
        model = cls()
        assert cls.__name__ in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(os.path.join(SAVED_MODELS_DIR, cls.__name__)):
            if filename.startswith(cls.__name__):
                match = re.match(fr"{cls.__name__}_(\d*)\.pickle", filename)
                if not match:
                    continue

                n_horse = int(match.group(1))
                model.n_horses_models[n_horse] = joblib.load(
                    os.path.join(SAVED_MODELS_DIR, cls.__name__, filename)
                )
        return model


class ModelNotCreatedOnceError(Exception):
    pass
