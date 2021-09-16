from typing import Optional

from catboost import CatBoostClassifier
from catboost import CatboostError

from constants import Sources
from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import JoblibPicklerMixin


class CatboostWinningModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    _NotFittedModelError = CatboostError

    def __init__(
        self, source: Sources, n_features: int, hyperparameters: Optional[dict] = None
    ):
        super().__init__(source=source, n_features=n_features)
        self._hyperparameters = hyperparameters

    def _create_n_horses_model(self, n_horses: int):
        if self._hyperparameters is None:
            return CatBoostClassifier()
        return CatBoostClassifier(**self._hyperparameters)
