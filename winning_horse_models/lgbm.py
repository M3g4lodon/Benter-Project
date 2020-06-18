from lightgbm import LGBMClassifier

from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import JoblibPicklerMixin


class LGBMWinningModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return LGBMClassifier()
