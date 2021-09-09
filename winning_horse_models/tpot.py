from typing import Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from tpot import TPOTClassifier

from constants import Sources
from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import ModelNotCreatedOnceError


class TPOTWinningModel(FlattenMixin, AbstractWinningModel):

    _NotFittedModelError = NotFittedError

    def __init__(
        self, source: Sources, n_features: int, hyperparameters: Optional[dict] = None
    ):
        super().__init__(source=source, n_features=n_features)
        self._hyperparameters = hyperparameters

    def _create_n_horses_model(self, n_horses: int):
        if self._hyperparameters is None:
            return TPOTClassifier()
        return TPOTClassifier(**self._hyperparameters)

    def predict(self, x: np.array, **kwargs):
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
        raise NotImplementedError

    @classmethod
    def load_model(
        cls, source: Sources, n_features: int, prefix: Optional[str] = None
    ) -> "TPOTWinningModel":
        raise NotImplementedError
