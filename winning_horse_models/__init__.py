import os
import re
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

import joblib
import numpy as np

from constants import SAVED_MODELS_DIR
from constants import SOURCE_PMU
from utils import preprocess

N_FEATURES = preprocess.get_n_preprocessed_feature_columns(source=SOURCE_PMU)

# TODO add MLP


class AbstractWinningModel(ABC):

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
    def __init__(self, selected_features: Optional[List[int]] = None):
        assert self._NotFittedModelError is not None
        self.n_horses_models = {}
        if selected_features is not None:
            assert len(selected_features) <= N_FEATURES
            assert all(0 <= index < N_FEATURES for index in selected_features)
        self.n_features = (
            N_FEATURES if selected_features is None else len(selected_features)
        )
        self.selected_features = selected_features

    def predict(self, x: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            if self.selected_features is not None:
                return model.predict(x=x[:, :, self.selected_features], **kwargs)
            return model.predict(x=x, **kwargs)
        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError

    def fit(self, x: np.array, y: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        if self.selected_features is None:
            return model.fit(x=x, y=y, **kwargs)

        if "validation_data" in kwargs:
            x_val, y_val = kwargs["validation_data"]
            x_val = x_val[:, :, self.selected_features]
            kwargs["validation_data"] = (x_val, y_val)
        return model.fit(x=x[:, :, self.selected_features], y=y, **kwargs)

    def evaluate(self, x: np.array, y: np.array, **kwargs):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            if self.selected_features is not None:
                return model.evaluate(x=x[:, :, self.selected_features], y=y, **kwargs)
            return model.evaluate(x=x, y=y, **kwargs)

        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError


class FlattenMixin:
    # TODO add flatten x for fit
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
