from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

import winning_validation.errors
from constants import Sources
from constants import SplitSets
from utils import import_data
from utils import permutations
from utils import preprocess
from winning_horse_models import AbstractWinningModel


def train_on_each_horse_with_epochs(
    source: Sources,
    winning_model: AbstractWinningModel,
    n_epochs: int,
    selected_features_index: Optional[List[int]] = None,
    extra_features_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    verbose: bool = False,
) -> Tuple[AbstractWinningModel, dict]:
    """Train deep learning (tf.keras like) model of each n_horses on n_epochs.
    Returns trained models and training history"""
    if selected_features_index is not None:
        assert len(set(selected_features_index)) == len(selected_features_index)
        assert min(selected_features_index) >= 0
        assert preprocess.get_n_preprocessed_feature_columns(source=source) > max(
            selected_features_index
        )
    else:
        selected_features_index = list(
            range(preprocess.get_n_preprocessed_feature_columns(source=source))
        )
    min_horse, max_horse = import_data.get_min_max_horse(source=source)
    training_history = {
        "n_horses_min": min_horse,
        "n_horses_max": max_horse,
        "epochs": {},
        "n_horses": {},
    }

    features_index = selected_features_index + ([-1] if extra_features_func else [])
    for epoch in range(n_epochs):
        training_history["epochs"][epoch] = {"n_horses_history": {}}
        if verbose:
            print(f"Epoch {epoch}")
        for n_horses in range(max(2, min_horse), max_horse + 1):
            training_history["epochs"][epoch]["n_horses_history"][n_horses] = {}
            x, y, _ = import_data.get_races_per_horse_number(
                source=source,
                n_horses=n_horses,
                on_split=SplitSets.TRAIN,
                x_format="sequential_per_horse",
                y_format="first_position",
                extra_features_func=extra_features_func,
            )
            if selected_features_index and x.size != 0:
                x = x[:, :, features_index]

            x_val, y_val, _ = import_data.get_races_per_horse_number(
                source=source,
                n_horses=n_horses,
                on_split=SplitSets.VAL,
                x_format="sequential_per_horse",
                y_format="first_position",
                extra_features_func=extra_features_func,
            )
            if selected_features_index and x_val.size != 0:
                x_val = x_val[:, :, features_index]

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

            if x.size == 0:
                val_loss = winning_model.evaluate(x=x_val, y=y_val)[0]
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
                history = winning_model.fit(x=x, y=y, verbose=int(verbose))
            else:
                history = winning_model.fit(
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


def pretrain_on_each_subraces(
    source: Sources,
    winning_model: AbstractWinningModel,
    n_permutations: int,
    verbose: bool = False,
) -> Tuple[AbstractWinningModel, dict]:
    min_horse, max_horse = import_data.get_min_max_horse(source=source)
    pretraining_history = {
        "n_horses_min": min_horse,
        "n_horses_max": max_horse,
        "n_horses_history": {},
    }
    for n_horses in range(max(2, min_horse), max_horse + 1):
        if verbose:
            print(f"Pretraining for {n_horses} horses")
        pretraining_history["n_horses_history"][n_horses] = {}

        x_val, y_val, _ = import_data.get_races_per_horse_number(
            source=source,
            n_horses=n_horses,
            on_split=SplitSets.VAL,
            x_format="sequential_per_horse",
            y_format="first_position",
        )

        cut_permuted_xs = []
        cut_permuted_ys = []

        for cut_n_horses in range(n_horses + 1, max_horse):
            if verbose:
                print(
                    f"Computing Subraces of {n_horses} horses "
                    f"with {cut_n_horses}-horse races"
                )
            x, rank, _ = import_data.get_races_per_horse_number(
                source=source,
                n_horses=cut_n_horses,
                on_split=SplitSets.TRAIN,
                x_format="sequential_per_horse",
                y_format="rank",
            )
            if x.size == 0:
                if verbose:
                    print(f"No {cut_n_horses}-horse races found")
                continue
            cut_permuted_xs.append(
                np.concatenate(
                    [
                        x[:, cut_permutation, :]
                        for cut_permutation in permutations.get_n_cut_permutations(
                            sequence=range(x.shape[1]), n=n_permutations, r=n_horses
                        )
                    ]
                )
            )

            permuted_rank = np.concatenate(
                [
                    rank[:, cut_permutation]
                    for cut_permutation in permutations.get_n_cut_permutations(
                        sequence=range(x.shape[1]), n=n_permutations, r=n_horses
                    )
                ]
            )
            first_positions = np.apply_along_axis(
                func1d=lambda rank_race: rank_race == rank_race.min(),
                axis=1,
                arr=permuted_rank,
            )

            cut_permuted_ys.append(np.asarray(first_positions).astype(np.float32))

        if not cut_permuted_xs:
            message = f"For {n_horses}, no subrace was found"
            pretraining_history["n_horses_history"][n_horses] = {"message": message}
            if verbose:
                print(message)

            continue

        cut_permuted_x: np.array = np.concatenate(cut_permuted_xs)
        cut_permuted_y: np.array = np.concatenate(cut_permuted_ys)

        pretraining_history["n_horses_history"][n_horses] = {
            "n_training_races": cut_permuted_x.shape[0],
            "n_validation_races": x_val.shape[0],
        }
        if cut_permuted_x.size == 0 and x_val.size == 0:
            message = f"No training or validation data for {n_horses}"
            pretraining_history.update({"message": message})
            if verbose:
                print(message)
            continue

        if x_val.size == 0:
            if verbose:
                print(f"\nNo val data for {n_horses}")

        model = winning_model.get_n_horses_model(n_horses=n_horses)
        if cut_permuted_x.size == 0:
            val_loss = model.evaluate(x=x_val, y=y_val)
            val_loss_per_horse = val_loss / n_horses
            message = (
                f"Evaluation only for {n_horses} horses: loss per horse None, "
                f"val loss per horse: {val_loss_per_horse:.3f}"
            )
            pretraining_history["n_horses_history"][n_horses].update(
                {"message": message, "validation_loss_per_horse": val_loss_per_horse}
            )

            if verbose:
                print(message)
                print()
            continue

        if x_val.size == 0:
            history = model.fit(
                x=cut_permuted_x, y=cut_permuted_y, verbose=int(verbose)
            )
        else:
            history = model.fit(
                x=cut_permuted_x,
                y=cut_permuted_y,
                validation_data=(x_val, y_val),
                verbose=int(verbose),
            )
        loss_per_horse = history.history["loss"][0] / n_horses
        accuracy = history.history["categorical_accuracy"][0]
        if x_val.size == 0:
            val_loss_per_horse = np.nan
            val_accuracy = np.nan

        else:
            val_loss_per_horse = history.history["val_loss"][0] / n_horses
            val_accuracy = history.history["val_categorical_accuracy"][0]

        pretraining_history["n_horses_history"][n_horses].update(
            {
                "training_loss_per_horse": loss_per_horse,
                "validation_loss_per_horse": val_loss_per_horse,
                "training_accuracy": accuracy,
                "validation_accuracy": val_accuracy,
            }
        )
        if verbose:
            print(
                f"Training for {n_horses} horses ({cut_permuted_x.shape[0]} races, "
                f"val {x_val.shape[0]} races): loss per horse: "
                f"{loss_per_horse:.3f}, val loss per horse: "
                f"{val_loss_per_horse:.3f} "
                f"Train Accuracy: {accuracy:.1%}, "
                f"Val Accuracy: {val_accuracy:.1%}\n"
            )

    if verbose:
        print("=" * 80)
        print()

    return winning_model, pretraining_history


# TODO test this
def train_on_all_races(
    source: Sources,
    winning_model: AbstractWinningModel,
    n_epochs: int,
    verbose: bool = False,
) -> Tuple[AbstractWinningModel, dict]:
    """Only for deeplearning models (tf.keras like)
    Train on several epochs on all races, in the order of happening"""
    assert n_epochs

    training_history: dict = {"epochs": {}}

    for epoch in range(n_epochs):
        training_history["epochs"][epoch] = {"history": []}
        if verbose:
            print(f"Epoch {epoch}")
        for x_race, y_race, _ in import_data.iter_dataset_races(
            source=source,
            on_split=SplitSets.TRAIN,
            x_format="sequential_per_horse",
            y_format="first_position",
        ):
            n_horses = x_race.shape[0]
            model = winning_model.get_n_horses_model(n_horses=n_horses)
            history = model.fit(x=x_race, y=y_race, verbose=int(verbose))
            training_history["epochs"][epoch]["history"].append(history)

        training_history["epochs"][epoch][
            "val_performance"
        ] = winning_validation.errors.compute_validation_error(
            source=source,
            k=1,
            winning_model=winning_model,
            validation_method=winning_validation.errors.exact_top_k,
        )

    return winning_model, training_history
