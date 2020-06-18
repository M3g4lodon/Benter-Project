import json
import os

import numpy as np
import tensorflow as tf

from constants import SAVED_MODELS_DIR
from constants import SOURCE_PMU
from utils import preprocess
from winning_horse_models import AbstractWinningModel
from winning_horse_models import SequentialMixin

N_FEATURES = preprocess.get_n_preprocessed_feature_columns(source=SOURCE_PMU)


class LogisticRegressionModel(SequentialMixin, AbstractWinningModel):
    def __init__(self):
        super().__init__()
        self.shared_layer = tf.keras.layers.Dense(1, name="shared_layer")

    def _create_n_horses_model(self, n_horses: int):
        inputs = tf.keras.Input(shape=(n_horses, N_FEATURES))
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
        model.build(input_shape=(None, n_horses, N_FEATURES))
        return model

    def save_model(self) -> None:
        if self.__class__.__name__ not in os.listdir(SAVED_MODELS_DIR):
            os.mkdir(os.path.join(SAVED_MODELS_DIR, self.__class__.__name__))
        weights, bias = self.shared_layer.get_weights()

        shared_layer = {
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "config": self.shared_layer.get_config(),
        }
        with open(
            os.path.join(
                SAVED_MODELS_DIR, self.__class__.__name__, "shared_weights.json"
            ),
            "w+",
        ) as fp:
            json.dump(obj=shared_layer, fp=fp)

    @classmethod
    def load_model(cls, trainable: bool) -> "LogisticRegressionModel":
        with open(
            os.path.join(SAVED_MODELS_DIR, cls.__name__, "shared_weights.json"), "r"
        ) as fp:
            shared_layer_json = json.load(fp=fp)
        baseline_weights = [
            np.array(shared_layer_json["weights"]),
            np.array(shared_layer_json["bias"]),
        ]

        shared_layer = tf.keras.layers.Dense(
            1, name="shared_layer", trainable=trainable
        )
        shared_layer.build(input_shape=(N_FEATURES,))
        shared_layer.set_weights(baseline_weights)

        model = LogisticRegressionModel()
        model.shared_layer = shared_layer
        return model
