from typing import Optional

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from constants import Sources
from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import JoblibPicklerMixin


class _SklearnMixin:
    _NotFittedModelError = NotFittedError
    _ClassifierClass = None

    def __init__(
        self, source: Sources, n_features: int, hyperparameters: Optional[dict] = None
    ):
        super().__init__(source=source, n_features=n_features)
        assert issubclass(self._ClassifierClass, ClassifierMixin)
        self._hyperparameters = hyperparameters

    def _create_n_horses_model(self, n_horses: int):
        if self._hyperparameters is None:
            return self._ClassifierClass()
        return self._ClassifierClass(**self._hyperparameters)


class KNNModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    _ClassifierClass = KNeighborsClassifier


class DecisionTreeModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    _ClassifierClass = DecisionTreeClassifier


class SVCModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    _ClassifierClass = SVC


class RandomForestModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    _ClassifierClass = RandomForestClassifier


class GradientBoostingModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    _ClassifierClass = GradientBoostingClassifier


class GaussianNBModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    _ClassifierClass = GaussianNB


class LDAModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    _ClassifierClass = LinearDiscriminantAnalysis


class SGDModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    _ClassifierClass = SGDClassifier


class LogisticRegressionModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    _ClassifierClass = LogisticRegression


class CatBoostModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    _ClassifierClass = CatBoostClassifier


class LGBMModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    _ClassifierClass = LGBMClassifier
