from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import JoblibPicklerMixin


class _SklearnMixin:
    _NotFittedModelError = NotFittedError


class KNNModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return KNeighborsClassifier()


class DecisionTreeModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    def _create_n_horses_model(self, n_horses: int):
        return DecisionTreeClassifier()


class SVCModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return SVC(kernel="rbf", C=0.025, probability=True)


class RandomForestModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    def _create_n_horses_model(self, n_horses: int):
        return RandomForestClassifier()


class GradientBoostingModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    def _create_n_horses_model(self, n_horses: int):
        return GradientBoostingClassifier()


class GaussianNBModel(
    _SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel
):
    def _create_n_horses_model(self, n_horses: int):
        return GaussianNB()


class LDAModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return LinearDiscriminantAnalysis()


class SGDModel(_SklearnMixin, FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return SGDClassifier(loss="log")
