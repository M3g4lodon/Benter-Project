import os
import re

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from constants import SAVED_MODELS_DIR
from winning_horse_models import AbstractWinningModel
from winning_horse_models import ModelNotCreatedOnceError


class KNNModel(AbstractWinningModel):
    def __init__(self):
        self.n_horses_models = {}

    def get_n_horses_model(self, n_horses: int, should_be_fitted: bool = False):
        if n_horses not in self.n_horses_models and should_be_fitted:
            raise ModelNotCreatedOnceError
        if n_horses not in self.n_horses_models:
            self.n_horses_models[n_horses] = KNeighborsClassifier()

        model = self.n_horses_models[n_horses]
        if should_be_fitted:
            check_is_fitted(model)
        return model

    def predict(self, x: np.array):
        model = self.get_n_horses_model(n_horses=x.shape[1], should_be_fitted=True)
        return model.predict_proba(
            X=np.reshape(a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F")
        )

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
    def load_model(cls, trainable: bool) -> "KNNModel":
        model = KNNModel()
        assert cls.__name__ in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(SAVED_MODELS_DIR):
            if filename.startswith(cls.__name__):
                match = re.match(fr"{cls.__name__}_(\d*)\.pickle", filename)
                if not match:
                    continue

                n_horse = int(match.group(0))
                model.n_horses_models[n_horse] = joblib.load(
                    os.path.join(SAVED_MODELS_DIR, cls.__name__, filename)
                )
        return model
