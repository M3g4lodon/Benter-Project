from typing import Tuple

import numpy as np
from sklearn import metrics

from utils import import_data
from winning_horse_models import AbstractWinningModel


def _one_hot_encode(y: np.array, n_horses: int) -> np.array:
    assert len(y.shape) == 1
    one_hot_encoded_y = np.zeros((y.size, n_horses))
    one_hot_encoded_y[np.arange(y.size), y.astype(np.int)] = 1
    return one_hot_encoded_y


# inspiration:
# - https://www.kaggle.com/yyzz1010/predict-the-winning-horse-100-on-small-test-data
def train_per_n_horses_races(
    source: str, winning_model: AbstractWinningModel, verbose: bool = False
) -> Tuple[AbstractWinningModel, dict]:
    """Train Sklearn-like model on each n_horses-size races.
    Returns trained model and training history"""
    min_horse, max_horse = import_data.get_min_max_horse(source=source)
    training_history = {
        "n_horses_min": min_horse,
        "n_horses_max": max_horse,
        "n_horses_history": {},
    }
    for n_horses in range(max(2, min_horse), max_horse + 1):
        x, y, _ = import_data.get_races_per_horse_number(
            source=source,
            n_horses=n_horses,
            on_split="train",
            x_format="flattened",
            y_format="index_first",
        )
        x_val, y_val, _ = import_data.get_races_per_horse_number(
            source=source,
            n_horses=n_horses,
            on_split="val",
            x_format="flattened",
            y_format="index_first",
        )

        training_history["n_horses_history"][n_horses] = {
            "n_training_races": x.shape[0],
            "n_validation_races": x_val.shape[0],
        }
        if x.shape[0] < MIN_N_EXAMPLES and x_val.size == 0:
            message = f"No training and validation data for {n_horses}"
            training_history["n_horses_history"][n_horses].update({"message": message})

            if verbose:
                print(message)

            continue

        if x_val.size == 0:
            if verbose:
                print(f"\nNo val data for {n_horses}")

        model = winning_model.get_n_horses_model(n_horses=n_horses)
        if x.shape[0] < MIN_N_EXAMPLES:
            message = (
                f"Not enough training examples for {n_horses} "
                f"horses (only {x.shape[0]} races)"
            )
            training_history["n_horses_history"][n_horses].update({"message": message})
            if verbose:
                print(message)
            continue

        model = model.fit(X=x, y=y)

        if verbose:

            loss_per_horse = (
                metrics.log_loss(
                    y_true=_one_hot_encode(y, n_horses=n_horses),
                    y_pred=model.predict_proba(x),
                )
                / n_horses
            )
            accuracy = metrics.accuracy_score(y_true=y, y_pred=model.predict(x))
            if x_val.shape[0] < MIN_N_EXAMPLES:
                val_loss_per_horse = np.nan
                val_accuracy = np.nan

            else:
                val_loss_per_horse = (
                    metrics.log_loss(
                        y_true=_one_hot_encode(y_val, n_horses=n_horses),
                        y_pred=model.predict_proba(x_val),
                    )
                    / n_horses
                )
                val_accuracy = metrics.accuracy_score(
                    y_true=y_val, y_pred=model.predict(x_val)
                )
            training_history["n_horses_history"][n_horses].update(
                {
                    "training_loss_per_horse": loss_per_horse,
                    "validation_loss_per_horse": val_loss_per_horse,
                    "training_accuracy": accuracy,
                    "validation_accuracy": val_accuracy,
                }
            )
            print(
                f"Training for {n_horses} horses ({x.shape[0]} races): loss per horse: "
                f"{loss_per_horse:.3f}, val loss per horse: {val_loss_per_horse:.3f} "
                f"Train Accuracy: {accuracy:.1%}, Val Accuracy: {val_accuracy:.1%}\n"
            )
        assert winning_model.get_n_horses_model(n_horses=n_horses) is model
    return winning_model, training_history


MIN_N_EXAMPLES = 10
