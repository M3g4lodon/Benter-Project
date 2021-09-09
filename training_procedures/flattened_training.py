from typing import List
from typing import Tuple

import numpy as np
from sklearn import metrics

from constants import Sources
from constants import SplitSets
from utils import import_data
from utils import permutations
from winning_horse_models import AbstractWinningModel

MIN_N_EXAMPLES = 10
N_MAX_PERMUTED_RACES = 50000


def _one_hot_encode(y: np.array, n_horses: int) -> np.array:
    assert len(y.shape) == 1
    one_hot_encoded_y = np.zeros((y.size, n_horses))
    one_hot_encoded_y[np.arange(y.size), y.astype(np.int)] = 1
    return one_hot_encoded_y


def _extend_y_pred_with_unseen_classes(
    missing_indexes: List[int], y_pred: np.ndarray
) -> np.ndarray:
    # Some label classes might not be seen by model, and therefore be missing
    for missing_index in sorted(missing_indexes):
        y_pred = np.concatenate(
            [
                y_pred[:, :missing_index],
                np.zeros((y_pred.shape[0], 1)),
                y_pred[:, missing_index:],
            ],
            axis=1,
        )
    return y_pred


def train_on_n_horses_races(
    source: Sources,
    winning_model: AbstractWinningModel,
    n_horses: int,
    verbose: bool = False,
) -> Tuple[AbstractWinningModel, dict]:
    x, y, _ = import_data.get_races_per_horse_number(
        source=source,
        n_horses=n_horses,
        on_split=SplitSets.TRAIN,
        x_format="flattened",
        y_format="index_first",
    )
    x_val, y_val, _ = import_data.get_races_per_horse_number(
        source=source,
        n_horses=n_horses,
        on_split=SplitSets.VAL,
        x_format="flattened",
        y_format="index_first",
    )

    training_history_n_horses = {
        "n_training_races": x.shape[0],
        "n_validation_races": x_val.shape[0],
    }
    if x.shape[0] < MIN_N_EXAMPLES and x_val.size == 0:
        message = f"No training and validation data for {n_horses}"
        training_history_n_horses.update({"message": message})

        if verbose:
            print(message)

        return winning_model, training_history_n_horses

    if x_val.size == 0:
        if verbose:
            print(f"\nNo val data for {n_horses}")

    model = winning_model.get_n_horses_model(n_horses=n_horses)
    if x.shape[0] < MIN_N_EXAMPLES:
        message = (
            f"Not enough training examples for {n_horses} "
            f"horses (only {x.shape[0]} races)"
        )
        training_history_n_horses.update({"message": message})
        if verbose:
            print(message)
        return winning_model, training_history_n_horses

    model = model.fit(x, y)

    if verbose:
        y_pred = model.predict_proba(x)
        y_pred = _extend_y_pred_with_unseen_classes(
            missing_indexes=[i for i in range(n_horses) if i not in model.classes_],
            y_pred=y_pred,
        )

        y_true = _one_hot_encode(y, n_horses=n_horses)
        if np.any(np.isnan(y_pred)):
            print(
                f"Model predict proba with NA values on "
                f"{np.any(np.isnan(y_pred), axis=1).sum()} races"
            )
            y_true = y_true[np.all(~np.isnan(y_pred), axis=1)]
            y_pred = y_pred[np.all(~np.isnan(y_pred), axis=1)]
        loss_per_horse = metrics.log_loss(y_true=y_true, y_pred=y_pred) / n_horses
        accuracy = metrics.accuracy_score(y_true=y, y_pred=model.predict(x))
        if x_val.shape[0] < MIN_N_EXAMPLES:
            val_loss_per_horse = np.nan
            val_accuracy = np.nan

        else:
            y_pred = model.predict_proba(x_val)
            y_pred = _extend_y_pred_with_unseen_classes(
                missing_indexes=[i for i in range(n_horses) if i not in model.classes_],
                y_pred=y_pred,
            )
            y_true = _one_hot_encode(y_val, n_horses=n_horses)
            if np.any(np.isnan(y_pred)):
                print(
                    f"Model predict proba with NA values on "
                    f"{np.any(np.isnan(y_pred), axis=1).sum()} validation races"
                )
                y_true = y_true[np.all(~np.isnan(y_pred), axis=1)]
                y_pred = y_pred[np.all(~np.isnan(y_pred), axis=1)]
            val_loss_per_horse = (
                metrics.log_loss(y_true=y_true, y_pred=y_pred) / n_horses
            )
            val_accuracy = metrics.accuracy_score(
                y_true=y_val, y_pred=model.predict(x_val)
            )
        training_history_n_horses.update(
            {
                "training_loss_per_horse": loss_per_horse,
                "validation_loss_per_horse": val_loss_per_horse,
                "training_accuracy": accuracy,
                "validation_accuracy": val_accuracy,
            }
        )
        if verbose:
            print(
                f"Training for {n_horses} horses ({x.shape[0]} races): loss per horse: "
                f"{loss_per_horse:.3f}, val loss per horse: {val_loss_per_horse:.3f} "
                f"Train Accuracy: {accuracy:.1%}, Val Accuracy: {val_accuracy:.1%}\n"
            )
    assert winning_model.get_n_horses_model(n_horses=n_horses) is model
    return winning_model, training_history_n_horses


# inspiration:
# - https://www.kaggle.com/yyzz1010/predict-the-winning-horse-100-on-small-test-data
def train_per_n_horses_races(
    source: Sources, winning_model: AbstractWinningModel, verbose: bool = False
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
        winning_model, n_horses_training_history = train_on_n_horses_races(
            source=source,
            winning_model=winning_model,
            n_horses=n_horses,
            verbose=verbose,
        )
        training_history["n_horses_history"][n_horses] = n_horses_training_history
    return winning_model, training_history


def train_per_n_horses_races_with_permutations(
    source: Sources,
    winning_model: AbstractWinningModel,
    n_permuted_races: int,
    verbose: bool = False,
) -> Tuple[AbstractWinningModel, dict]:
    """Train Sklearn-like model on each permutated n_horses-size races,
    Returns trained model and training history"""
    assert n_permuted_races <= N_MAX_PERMUTED_RACES

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
            on_split=SplitSets.TRAIN,
            x_format="sequential_per_horse",
            y_format="index_first",
        )
        x_val, y_val, _ = import_data.get_races_per_horse_number(
            source=source,
            n_horses=n_horses,
            on_split=SplitSets.VAL,
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

        n_permutations = n_permuted_races // x.shape[0]
        x_with_permutations = np.concatenate(
            [x]
            + [
                x[:, permutation, :]
                for permutation in permutations.get_n_permutations(
                    sequence=range(x.shape[1]), n=n_permutations
                )
            ]
        )
        x_with_permutations = np.reshape(
            a=x_with_permutations,
            newshape=(
                x_with_permutations.shape[0],
                x_with_permutations.shape[1] * x_with_permutations.shape[2],
            ),
            order="F",
        )
        y_with_permutations = np.concatenate(
            [y]
            + [
                np.vectorize(lambda y_i: permutation.index(y_i))(y)
                for permutation in permutations.get_n_permutations(
                    sequence=range(x.shape[1]), n=n_permutations
                )
            ]
        )
        model = model.fit(x_with_permutations, y_with_permutations)

        if verbose:
            y_pred = model.predict_proba(x_with_permutations)
            y_true = _one_hot_encode(y_with_permutations, n_horses=n_horses)
            if np.any(np.isnan(y_pred)):
                print(
                    f"Model predict proba with NA values on "
                    f"{np.any(np.isnan(y_pred), axis=1).sum()} races"
                )
                y_true = y_true[np.all(~np.isnan(y_pred), axis=1)]
                y_pred = y_pred[np.all(~np.isnan(y_pred), axis=1)]
            loss_per_horse = metrics.log_loss(y_true=y_true, y_pred=y_pred) / n_horses
            accuracy = metrics.accuracy_score(
                y_true=y_with_permutations, y_pred=model.predict(x_with_permutations)
            )
            if x_val.shape[0] < MIN_N_EXAMPLES:
                val_loss_per_horse = np.nan
                val_accuracy = np.nan

            else:
                y_pred = model.predict_proba(x_val)
                y_true = _one_hot_encode(y_val, n_horses=n_horses)
                if np.any(np.isnan(y_pred)):
                    print(
                        f"Model predict proba with NA values on "
                        f"{np.any(np.isnan(y_pred), axis=1).sum()} validation races"
                    )
                    y_true = y_true[np.all(~np.isnan(y_pred), axis=1)]
                    y_pred = y_pred[np.all(~np.isnan(y_pred), axis=1)]
                val_loss_per_horse = (
                    metrics.log_loss(y_true=y_true, y_pred=y_pred) / n_horses
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
                f"Training for {n_horses} horses ({x.shape[0]} races, "
                f"{x_with_permutations.shape[0]} with permutations): loss per horse: "
                f"{loss_per_horse:.3f}, val loss per horse: {val_loss_per_horse:.3f} "
                f"Train Accuracy: {accuracy:.1%}, Val Accuracy: {val_accuracy:.1%}\n"
            )
        assert winning_model.get_n_horses_model(n_horses=n_horses) is model
    return winning_model, training_history
