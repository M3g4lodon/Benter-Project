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

    # TODO(mathieu) Add n_features as attributes
    _NotFittedModelError = XGBoostError

    def _create_n_horses_model(self, n_horses: int):
        return XGBClassifier()

    def predict(self, x: np.array):
        model = self.get_n_horses_model(n_horses=x.shape[1])
        try:
            return model.predict_proba(
                data=np.reshape(
                    a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F"
                )
            )
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

    @classmethod
    def load_model(
        cls, source: Sources, prefix: Optional[str] = None
    ) -> "XGBoostWinningModel":
        model = XGBoostWinningModel(source=source)
        assert cls.__name__ in os.listdir(SAVED_MODELS_DIR)
        for filename in os.listdir(os.path.join(SAVED_MODELS_DIR, cls.__name__)):
            if filename.startswith(f"{prefix}{cls.__name__}"):
                match = re.match(fr"{prefix}{cls.__name__}_(\d*)\.json", filename)
                if not match:
                    continue
                n_horse = int(match.group(1))
                model.n_horses_models[n_horse] = XGBClassifier()
                model.n_horses_models[n_horse].load_model(
                    fname=os.path.join(SAVED_MODELS_DIR, cls.__name__, filename)
                )
                assert model.n_horses_models[n_horse] is not None
        return model


NotFittedError
