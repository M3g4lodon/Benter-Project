from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from winning_horse_models import AbstractWinningModel
from winning_horse_models import FlattenMixin
from winning_horse_models import JoblibPicklerMixin


class KNNModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return KNeighborsClassifier()


class DecisionTreeModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return DecisionTreeClassifier()


class SVCModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return SVC(kernel="rbf", C=0.025, probability=True)


class RandomForestModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return RandomForestClassifier()


class GradientBoostingModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return GradientBoostingClassifier()


class GaussianNBModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return GaussianNB()


class LDAModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return LinearDiscriminantAnalysis()


class QDAModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return QuadraticDiscriminantAnalysis()


class SGDModel(FlattenMixin, JoblibPicklerMixin, AbstractWinningModel):
    def _create_n_horses_model(self, n_horses: int):
        return SGDClassifier()
