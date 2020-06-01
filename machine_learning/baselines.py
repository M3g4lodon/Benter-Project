import numpy as np

from machine_learning import WinningModel


class RandomModel(WinningModel):
    def get_n_horses_model(self, n_horses: int):
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

    def predict(self, x: np.array):
        n_examples = x.shape[0]
        n_horses = x.shape[1]
        return np.random.random(size=(n_examples, n_horses))

    def save_model(self) -> None:
        pass

    @classmethod
    def load_model(cls, trainable: bool) -> "RandomModel":
        return RandomModel()
