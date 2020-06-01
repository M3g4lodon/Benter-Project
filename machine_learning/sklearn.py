import os
import re

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted
import joblib
import numpy as np

from constants import SAVED_MODELS_DIR
from machine_learning import WinningModel, ModelNotCreatedOnce


class KNNModel(WinningModel):
    name = "K_Nearest_Neighbours"

    def __init__(self):
        self.n_horses_models = {}

    def get_n_horses_model(self, n_horses: int, should_be_fitted: bool = False):
        if n_horses not in self.n_horses_models and should_be_fitted:
            raise ModelNotCreatedOnce
        if n_horses not in self.n_horses_models:
            self.n_horses_models[n_horses] = KNeighborsClassifier()

        model = self.n_horses_models[n_horses]
        if should_be_fitted:
            check_is_fitted(model)
        return model

    def predict(self, x: np.array):
        model = self.get_n_horses_model(n_horses=x.shape[1], should_be_fitted=True)
        return model.predict_proba(X=np.reshape(a=x, newshape=(x.shape[0], x.shape[1]*x.shape[2]), order="F"))

    def save_model(self) -> None:
        if self.name not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.name))
        for n_horse, n_horse_model in self.n_horses_models.items():
            joblib.dump(
                n_horse_model,
                os.path.join(SAVED_MODELS_DIR, self.__name__, f"{self.name}_{n_horse}.pickle"),
            )

    @classmethod
    def load_model(cls, trainable: bool) -> "KNNModel":
        model = KNNModel()
        assert cls.name in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(SAVED_MODELS_DIR):
            if filename.startswith(cls.name):
                n_horse = int(re.match(fr"{cls.name}_(\d*)\.pickle", filename).group(0))
                model.n_horses_models[n_horse] = joblib.load(
                    os.path.join(SAVED_MODELS_DIR,cls.__name__,  filename)
                )
        return model
