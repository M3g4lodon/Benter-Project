import os
import re
from typing import Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from constants import SAVED_MODELS_DIR
from constants import Sources
from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import ModelNotCreatedOnceError


class XGBoostWinningModel(FlattenMixin, AbstractWinningModel):

    _NotFittedModelError = XGBoostError

    def __init__(
        self, source: Sources, n_features: int, hyperparameters: Optional[dict] = None
    ):
        super().__init__(source=source, n_features=n_features)
        self._hyperparameters = hyperparameters

    def _create_n_horses_model(self, n_horses: int):
        if self._hyperparameters is None:
            return XGBClassifier()
        return XGBClassifier(**self._hyperparameters)

    def predict(self, x: np.array, **kwargs):
        n_races, n_horses, n_features = x.shape
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            prediction = model.predict_proba(
                data=np.reshape(
                    a=x, newshape=(n_races, n_horses * n_features), order="F"
                )
            )
            # Can happen when training data doest not contain all horses position (aka classes)
            # as winner
            if model.n_classes_ != x.shape[1]:
                missing_classes = [
                    i for i in range(n_horses) if i not in model.classes_
                ]
                for idx in missing_classes:
                    prediction = np.hstack(
                        (
                            prediction[:, :idx],
                            np.zeros((n_races, 1)),
                            prediction[:, idx:],
                        )
                    )
            return prediction

        except self._NotFittedModelError:
            raise ModelNotCreatedOnceError

    def save_model(self, prefix: Optional[str] = None) -> None:
        if self.__class__.__name__ not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.__class__.__name__))
        for n_horse, n_horse_model in self.n_horses_models.items():
            try:
                n_horse_model.save_model(
                    fname=os.path.join(
                        SAVED_MODELS_DIR,
                        self.__class__.__name__,
                        f"{prefix}{self.__class__.__name__}_{n_horse}.pickle",
                    )
                )
            except XGBoostError as e:
                print(f"Could not save model for {n_horse} horses: {e}")
            except NotFittedError as e:
                print(f"Could not save model for {n_horse} horses: {e}")

    # todo(mathieu) To be tested
    @classmethod
    def load_model(
        cls, source: Sources, n_features: int, prefix: Optional[str] = None
    ) -> "XGBoostWinningModel":
        model = XGBoostWinningModel(source=source, n_features=n_features)
        assert cls.__name__ in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(os.path.join(SAVED_MODELS_DIR, cls.__name__)):
            if filename.startswith(f"{prefix}{cls.__name__}"):
                match = re.match(fr"{prefix}{cls.__name__}_(\d*)\.pickle", filename)
                if not match:
                    continue
                n_horse = int(match.group(1))
                model.n_horses_models[n_horse] = XGBClassifier()
                model.n_horses_models[n_horse].load_model(
                    fname=os.path.join(SAVED_MODELS_DIR, cls.__name__, filename)
                )
                assert model.n_horses_models[n_horse] is not None
        return model
