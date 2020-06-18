from typing import Tuple

import numpy as np

from utils import import_data
from utils import winning_validation
from winning_horse_models import AbstractWinningModel


def train_on_each_horse_with_epochs(
    source: str,
    winning_model: AbstractWinningModel,
    n_epochs: int,
    verbose: bool = False,
) -> Tuple[AbstractWinningModel, dict]:
    """Train deep learning (tf.keras like) model of each n_horses on n_epochs.
    Returns trained models and triaining history"""
    min_horse, max_horse = import_data.get_min_max_horse(source=source)
    training_history = {
        "n_horses_min": min_horse,
        "n_horses_max": max_horse,
        "epochs": {},
        "n_horses": {},
    }
    for epoch in range(n_epochs):
        training_history["epochs"][epoch] = {"n_horses_history": {}}
        if verbose:
            print(f"Epoch {epoch}")
        for n_horses in range(max(2, min_horse), max_horse + 1):
            training_history["epochs"][epoch]["n_horses_history"][n_horses] = {}
            x, y, _ = import_data.get_races_per_horse_number(
                source=source,
                n_horses=n_horses,
                on_split="train",
                x_format="sequential_per_horse",
                y_format="first_position",
            )
            x_val, y_val, _ = import_data.get_races_per_horse_number(
                source=source,
                n_horses=n_horses,
                on_split="val",
                x_format="sequential_per_horse",
                y_format="first_position",
            )
            if epoch == 0:
                training_history["n_horses"][n_horses] = {
                    "n_training_races": x.shape[0],
                    "n_validation_races": x_val.shape[0],
                }
            if x.size == 0 and x_val.size == 0:
                message = f"No training or validation data for {n_horses}"
                training_history["epochs"][epoch].update({"message": message})
                if verbose:
                    print(message)
                continue

            if x_val.size == 0:
                if verbose:
                    print(f"\nNo val data for {n_horses}")

            model = winning_model.get_n_horses_model(n_horses=n_horses)
            if x.size == 0:
                val_loss = model.evaluate(x=x_val, y=y_val)
                val_loss_per_horse = val_loss / n_horses
                message = (
                    f"Evaluation only for {n_horses} horses: loss per horse None, "
                    f"val loss per horse: {val_loss_per_horse:.3f}"
                )
                training_history["epochs"][epoch]["n_horses_history"][n_horses].update(
                    {
                        "message": message,
                        "validation_loss_per_horse": val_loss_per_horse,
                    }
                )

                if verbose:
                    print(message)
                    print()
                continue

            if x_val.size == 0:
                history = model.fit(x=x, y=y, verbose=int(verbose))
            else:
                history = model.fit(
                    x=x, y=y, validation_data=(x_val, y_val), verbose=int(verbose)
                )
            loss_per_horse = history.history["loss"][0] / n_horses
            accuracy = history.history["categorical_accuracy"][0]
            if x_val.size == 0:
                val_loss_per_horse = np.nan
                val_accuracy = np.nan

            else:
                val_loss_per_horse = history.history["val_loss"][0] / n_horses
                val_accuracy = history.history["val_categorical_accuracy"][0]

            training_history["epochs"][epoch]["n_horses_history"][n_horses].update(
                {
                    "training_loss_per_horse": loss_per_horse,
                    "validation_loss_per_horse": val_loss_per_horse,
                    "training_accuracy": accuracy,
                    "validation_accuracy": val_accuracy,
                }
            )
            if verbose:
                print(
                    f"Training for {n_horses} horses ({x.shape[0]} races, "
                    f"val {x_val.shape[0]} races): loss per horse: "
                    f"{loss_per_horse:.3f}, val loss per horse: "
                    f"{val_loss_per_horse:.3f} "
                    f"Train Accuracy: {accuracy:.1%}, "
                    f"Val Accuracy: {val_accuracy:.1%}\n"
                )

        if verbose:
            print("=" * 80)
            print()

    return winning_model, training_history


def train_on_all_races(
    source: str,
    winning_model: AbstractWinningModel,
    n_epochs: int,
    verbose: bool = False,
) -> Tuple[AbstractWinningModel, dict]:
    """Only for deeplearning models (tf.keras like)
    Train on several epochs on all races, in the order of happening"""
    assert n_epochs

    training_history = {"epochs": {}}

    for epoch in range(n_epochs):
        training_history["epochs"][epoch] = {"history": []}
        if verbose:
            print(f"Epoch {epoch}")
        for x_race, y_race, _ in import_data.get_dataset_races(
            source=source,
            on_split="train",
            x_format="sequential_per_horse",
            y_format="first_position",
        ):
            n_horses = x_race.shape[0]
            model = winning_model.get_n_horses_model(n_horses=n_horses)
            history = model.fit(x=x_race, y=y_race, verbose=int(verbose))
            training_history["epochs"][epoch]["history"].append(history)

        training_history["epochs"][epoch][
            "val_performance"
        ] = winning_validation.compute_validation_error(
            source=source,
            k=1,
            winning_model=winning_model,
            validation_method=winning_validation.exact_top_k,
        )

    return winning_model, training_history
