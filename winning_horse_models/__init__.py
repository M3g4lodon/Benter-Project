import os
import re
from abc import ABC
from abc import abstractmethod
from typing import Optional

import joblib
import numpy as np

from constants import SAVED_MODELS_DIR
from constants import Sources


# TODO add MLP


class AbstractWinningModel(ABC):

    _NotFittedModelError: Optional[Exception] = None

    def __init__(self, source: Sources):
        assert self._NotFittedModelError is not None
        self.source = source
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
    def fit(self, x: np.array, y: np.array, **kwargs):
        """A common method to fit the model on given races (with n_horses)"""

    @abstractmethod
    def predict(self, x: np.array, **kwargs):
        """A common method to predict probabilities on given races (with n_horses)"""

    @abstractmethod
    def save_model(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def load_model(cls) -> "AbstractWinningModel":
        pass


class SequentialMixin:
    def __init__(self, n_features: Optional[int], **kwargs):
        assert self._NotFittedModelError is not None
        self.n_horses_models = {}
        self.n_features = n_features

    def predict(self, x: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            return model.predict(x=x, **kwargs)
        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError

    def fit(self, x: np.array, y: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])

        return model.fit(x=x, y=y, **kwargs)

    def evaluate(self, x: np.array, y: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            return model.evaluate(x=x, y=y, **kwargs)

        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError


class FlattenMixin:
    def fit(self, x: np.array, y: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])

        return model.fit(
            x=np.reshape(
                a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F"
            ),
            y=y,
            **kwargs,
        )

    def predict(self, x: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            return model.predict_proba(
                X=np.reshape(
                    a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F"
                ),
                **kwargs,
            )
        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError

    # TODO extend to other sklearn api methods, with feature selection


class JoblibPicklerMixin:
    def save_model(self, prefix: Optional[str] = None) -> None:
        prefix = prefix or ""
        if self.__class__.__name__ not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.__class__.__name__))
        for n_horse, n_horse_model in self.n_horses_models.items():
            joblib.dump(
                n_horse_model,
                os.path.join(
                    SAVED_MODELS_DIR,
                    self.__class__.__name__,
                    f"{prefix}{self.__class__.__name__}_{n_horse}.pickle",
                ),
            )

    @classmethod
    def load_model(cls, prefix: Optional[str] = None):
        prefix = prefix or ""
        model = cls()
        assert cls.__name__ in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(os.path.join(SAVED_MODELS_DIR, cls.__name__)):
            if filename.startswith(cls.__name__):
                match = re.match(fr"{prefix}{cls.__name__}_(\d*)\.pickle", filename)
                if not match:
                    continue

                n_horse = int(match.group(1))
                model.n_horses_models[n_horse] = joblib.load(
                    os.path.join(SAVED_MODELS_DIR, cls.__name__, filename)
                )
        return model


class ModelNotCreatedOnceError(Exception):
    pass


class OddsCombinedWinningModel:
    def __init__(self, alpha: float, beta: float, winning_model: AbstractWinningModel):
        self._alpha = alpha
        self._beta = beta
        self._winning_model = winning_model

    def predict(self, x: np.array, odds: np.array, **kwargs):
        if np.all(np.isnan(odds)):
            return np.full_like(np.expand_dims(odds, axis=0), np.nan)
        winning_model_pred = self._winning_model.predict(x=x)
        pari_mutual_proba = 1 / odds
        combined_pred = np.exp(
            self._alpha * np.log(winning_model_pred)
            + self._beta * np.log(pari_mutual_proba)
        )
        denominator = np.exp(
            self._alpha * np.log(winning_model_pred)
            + self._beta * np.log(pari_mutual_proba)
        )
        denominator[np.isnan(denominator)] = 0
        denominator = denominator.sum()
        assert not np.isnan(denominator)
        combined_pred /= denominator
        return combined_pred
