import os
import re
from typing import Dict

import numpy as np
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from constants import SAVED_MODELS_DIR
from winning_horse_models import AbstractWinningModel
from winning_horse_models import ModelNotCreatedOnceError


class XGBoostWinningModel(AbstractWinningModel):
    def __init__(self):
        self.n_horses_models: Dict[int, XGBClassifier] = {}

    def get_n_horses_model(
        self, n_horses: int, should_be_fitted: bool = False
    ) -> XGBClassifier:
        if n_horses not in self.n_horses_models and should_be_fitted:
            raise ModelNotCreatedOnceError
        if n_horses not in self.n_horses_models:
            self.n_horses_models[n_horses] = XGBClassifier()

        model = self.n_horses_models[n_horses]
        return model

    def predict(self, x: np.array) -> np.array:
        model = self.get_n_horses_model(n_horses=x.shape[1], should_be_fitted=True)
        return model.predict_proba(
            np.reshape(a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F")
        )

    def save_model(self) -> None:
        if self.__class__.__name__ not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.__class__.__name__))
        for n_horse, n_horse_model in self.n_horses_models.items():
            try:
                n_horse_model.save_model(
                    fname=os.path.join(
                        SAVED_MODELS_DIR,
                        self.__class__.__name__,
                        f"{self.__class__.__name__}_{n_horse}.json",
                    )
                )
            except XGBoostError as e:
                print(f"Could not save model for {n_horse} horses: {e}")

    @classmethod
    def load_model(cls, trainable: bool) -> "XGBoostWinningModel":
        model = XGBoostWinningModel()
        assert cls.__name__ in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(os.path.join(SAVED_MODELS_DIR, cls.__name__)):
            if filename.startswith(cls.__name__):
                match = re.match(fr"{cls.__name__}_(\d*)\.json", filename)
                if not match:
                    continue
                n_horse = int(match.group(1))
                model.n_horses_models[n_horse] = XGBClassifier()
                model.n_horses_models[n_horse].load_model(
                    fname=os.path.join(SAVED_MODELS_DIR, cls.__name__, filename)
                )
                assert model.n_horses_models[n_horse] is not None
        return model
