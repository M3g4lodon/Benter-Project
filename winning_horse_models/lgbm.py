from typing import Optional

from lightgbm import LGBMClassifier
from lightgbm.compat import LGBMNotFittedError

from constants import Sources
from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import JoblibPicklerMixin


class LGBMWinningModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    _NotFittedModelError = LGBMNotFittedError

    def __init__(
        self, source: Sources, n_features: int, hyperparameters: Optional[dict] = None
    ):
        super().__init__(source=source, n_features=n_features)
        self._hyperparameters = hyperparameters

    def _create_n_horses_model(self, n_horses: int):
        if self._hyperparameters is None:
            return LGBMClassifier()
        return LGBMClassifier(**self._hyperparameters)
