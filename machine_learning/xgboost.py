import os
import re
from xgboost import XGBClassifier
import numpy as np

from constants import SAVED_MODELS_DIR
from machine_learning import AbstractWinningModel, ModelNotCreatedOnce


def check_if_xgboost_if_fitted(model: XGBClassifier) -> None:
    model.feature_importances_


class XGBoostWinningModel(AbstractWinningModel):

    def __init__(self):
        self.n_horses_models = {}

    def get_n_horses_model(self, n_horses: int, should_be_fitted: bool = False)->XGBClassifier:
        if n_horses not in self.n_horses_models and should_be_fitted:
            raise ModelNotCreatedOnce
        if n_horses not in self.n_horses_models:
            self.n_horses_models[n_horses] = XGBClassifier()

        model = self.n_horses_models[n_horses]
        if should_be_fitted:
            check_if_xgboost_if_fitted(model=model)
        return model

    def predict(self, x: np.array)->np.array:
        model = self.get_n_horses_model(n_horses=x.shape[1], should_be_fitted=True)
        return model.predict_proba(
            X=np.reshape(a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F")
        )

    def save_model(self) -> None:
        if self.__name__ not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.__name__))
        for n_horse, n_horse_model in self.n_horses_models.items():
            n_horse_model.save_model(
                fname=os.path.join(
                    SAVED_MODELS_DIR, self.__name__, f"{self.__name__}_{n_horse}.json"
                )
            )

    @classmethod
    def load_model(cls, trainable: bool) -> "XGBoostWinningModel":
        model = XGBoostWinningModel()
        assert cls.__name__ in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(SAVED_MODELS_DIR):
            if filename.startswith(cls.__name__):
                n_horse = int(re.match(fr"{cls.name}_(\d*)\.json", filename).group(0))
                model.n_horses_models[n_horse] = XGBClassifier().load_model(
                    fname=os.path.join(SAVED_MODELS_DIR, cls.__name__, filename)
                )
        return model
