import json
import os
from typing import Optional

import numpy as np
import tensorflow as tf

from constants import SAVED_MODELS_DIR
from utils import import_data
from winning_horse_models import AbstractWinningModel
from winning_horse_models import SequentialMixin


class _ShouldNotBeTriggeredException(Exception):
    pass


class LogisticRegressionModel(SequentialMixin, AbstractWinningModel):

    _NotFittedModelError = _ShouldNotBeTriggeredException

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_layer = tf.keras.layers.Dense(1, name="shared_layer")

    def _create_n_horses_model(self, n_horses: int):
        inputs = tf.keras.Input(shape=(n_horses, self.n_features))
        unstacked = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=1))(inputs)
        dense_outputs = [self.shared_layer(x) for x in unstacked]  # our shared layer
        merged = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
        outputs = tf.keras.layers.Reshape(target_shape=(n_horses,))(merged)
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.keras.activations.softmax(x, axis=-1)
        )(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["categorical_accuracy", "categorical_crossentropy"],
        )
        model.build(input_shape=(None, n_horses, self.n_features))
        return model

    def save_model(self, prefix: Optional[str] = None) -> None:
        if self.__class__.__name__ not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.__class__.__name__))
        weights, bias = self.shared_layer.get_weights()
        prefix = prefix or ""
        shared_layer = {
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "config": self.shared_layer.get_config(),
            "n_features": self.n_features,
        }
        with open(
            os.path.join(
                SAVED_MODELS_DIR,
                self.__class__.__name__,
                f"{prefix}shared_weights.json",
            ),
            "w+",
        ) as fp:
            json.dump(obj=shared_layer, fp=fp)

    @classmethod
    def load_model(cls, prefix: Optional[str] = None) -> "LogisticRegressionModel":
        prefix = prefix or ""
        with open(
            os.path.join(
                SAVED_MODELS_DIR, cls.__name__, f"{prefix}shared_weights.json"
            ),
            "r",
        ) as fp:
            shared_layer_json = json.load(fp=fp)
        baseline_weights = [
            np.array(shared_layer_json["weights"]),
            np.array(shared_layer_json["bias"]),
        ]

        shared_layer = tf.keras.layers.Dense(1, name="shared_layer", trainable=True)
        shared_layer.build(input_shape=(len(shared_layer_json["weights"]),))
        shared_layer.set_weights(baseline_weights)

        model = LogisticRegressionModel(n_features=shared_layer_json["n_features"])
        model.shared_layer = shared_layer
        return model


from joblib import Parallel, delayed
from scipy import optimize

#### SCRIPT Don't use it
from constants import Sources


def compute_log_likelihood(intercept_weights):
    intercept, weights = intercept_weights[0], intercept_weights[1:]
    weights = np.array(weights)
    min_horse, max_horse = import_data.get_min_max_horse(source=Sources.PMU)

    # TODO np.vectorize this function
    def _compute_n_horses_log_likelihood(n_horses):
        x, y, _ = import_data.get_races_per_horse_number(
            source=Sources.PMU,
            n_horses=n_horses,
            on_split="train",
            x_format="sequential_per_horse",
            y_format="index_first",
        )
        n_races = y.shape[0]
        if n_races == 0:
            return 0
        winner_horses = np.take_along_axis(
            arr=x, indices=y.astype(int).reshape((-1, 1, 1)), axis=1
        )
        log_likelihood = np.tensordot(weights.T, winner_horses, axes=([0], [2])).sum()
        log_likelihood += n_races * intercept

        divider = np.tensordot(weights.T, x, axes=([0], [2])) + intercept
        divider = np.exp(divider).sum(axis=1)
        log_likelihood -= np.log(divider).sum()
        return log_likelihood

    r = Parallel(n_jobs=6, verbose=10)(
        delayed(_compute_n_horses_log_likelihood)(n_horses)
        for n_horses in range(max(2, min_horse), max_horse + 1)
    )

    return np.sum(r)


def minus_log_likelihood(intercept_weights):
    return -compute_log_likelihood(intercept_weights)


class ClassesFreeMultinomialLogisticRegression(SequentialMixin, AbstractWinningModel):
    def __init__(self):
        super().__init__()
        # Use old
        with open(
            "./saved_models/LogisticRegressionModel/shared_weights.json", "r"
        ) as fp:
            shared_weights = json.load(fp=fp)

        weights0 = np.array(shared_weights["weights"]).reshape(-1)

        intercept0 = shared_weights["bias"][0]
        self.weights = weights0
        self.intercept = intercept0

    def train(self):

        intercept_weight0 = np.array([self.intercept0] + self.weights0.tolist())

        optimized_result = optimize.minimize(
            fun=minus_log_likelihood, x0=intercept_weight0
        )
        # TODO finish it
