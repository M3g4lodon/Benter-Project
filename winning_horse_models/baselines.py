import numpy as np

from winning_horse_models import AbstractWinningModel
from winning_horse_models import SequentialMixin


class RandomModel(SequentialMixin, AbstractWinningModel):

    _NotFittedModelError = Exception("This exception should not be triggered")

    def __init__(self):
        super().__init__(n_features=None)

    def _create_n_horses_model(self, n_horses: int):
        class RandomBaseline:
            def __init__(self, n_horses: int):
                self.n_horses = n_horses

            def fit(self, x, y):
                pass

            def predict(self, x):
                n_examples = x.shape[0]
                return np.random.random(size=(n_examples, self.n_horses))

            def fit_predict(self, x, y):
                return self.predict(x=x)

        return RandomBaseline(n_horses=n_horses)

    def save_model(self) -> None:
        pass

    @classmethod
    def load_model(cls) -> "RandomModel":
        return RandomModel()
