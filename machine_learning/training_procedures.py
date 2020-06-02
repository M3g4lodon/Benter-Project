import numpy as np
from sklearn import metrics

from machine_learning import AbstractWinningModel
from utils import import_data


MIN_N_EXAMPLES = 10
# TODO compare training on all races, specialise on each n_horse races, pre train on combinaison,


def _one_hot_encode(y, n_horses:int):
    assert len(y.shape) == 1
    one_hot_encoded_y = np.zeros((y.size, n_horses))

    one_hot_encoded_y[np.arange(y.size), y] = 1
    return one_hot_encoded_y


# inspiration:
# - https://www.kaggle.com/yyzz1010/predict-the-winning-horse-100-on-small-test-data
def train_per_horse(
    source: str, winning_model: AbstractWinningModel, verbose: bool = False
) -> AbstractWinningModel:
    """Train Sklearn-like model on each n_horses-size races."""
    min_horse, max_horse = import_data.get_min_max_horse(source=source)

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
        if x.shape[0] < MIN_N_EXAMPLES and x_val.size == 0:
            if verbose:
                print(f"No training and validation data for {n_horses}")
            continue

        if x_val.size == 0:
            if verbose:
                print(f"\nNo val data for {n_horses}")

        model = winning_model.get_n_horses_model(n_horses=n_horses)
        if x.shape[0] < MIN_N_EXAMPLES:

            if verbose:
                print(
                    f"Not enough training examples for {n_horses} horses (only {x.shape[0]} races)"
                )
            continue

        model = model.fit(X=x, y=y)

        if verbose:

            loss_per_horse = (
                metrics.log_loss(
                    y_true=_one_hot_encode(y, n_horses=n_horses), y_pred=model.predict_proba(X=x)
                )
                / n_horses
            )
            accuracy = metrics.accuracy_score(y_true=y, y_pred=model.predict(X=x))
            if x_val.shape[0] < MIN_N_EXAMPLES:
                val_loss_per_horse = np.nan
                val_accuracy = np.nan

            else:
                val_loss_per_horse = (
                    metrics.log_loss(
                        y_true=_one_hot_encode(y_val, n_horses=n_horses),
                        y_pred=model.predict_proba(X=x_val),
                    )
                    / n_horses
                )
                val_accuracy = metrics.accuracy_score(
                    y_true=y_val, y_pred=model.predict(X=x_val)
                )
            print(
                f"Training for {n_horses} horses ({x.shape[0]} races): loss per horse: "
                f"{loss_per_horse:.3f}, val loss per horse: {val_loss_per_horse:.3f} "
                f"Train Accuracy: {accuracy:.1%}, Val Accuracy: {val_accuracy:.1%}\n"
            )
        assert winning_model.n_horses_models[n_horses] is model
    return winning_model


def train_on_each_horse_with_epochs(
    source: str, winning_model: AbstractWinningModel, n_epochs: int, verbose: bool = False
) -> AbstractWinningModel:
    """Train deep learning (tf.keras like) model of each n_horses on n_epochs"""
    min_horse, max_horse = import_data.get_min_max_horse(source=source)
    for epoch in range(n_epochs):
        if verbose:
            print(f"Epoch {epoch}")
        for n_horses in range(max(2, min_horse), max_horse + 1):
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
            if x.size == 0 and x_val.size == 0:
                if verbose:
                    print(f"No training or validation data for {n_horses}")
                continue

            if x_val.size == 0:
                if verbose:
                    print(f"\nNo val data for {n_horses}")

            model = winning_model.get_n_horses_model(n_horses=n_horses)
            if x.size == 0:
                val_loss = model.evaluate(x=x_val, y=y_val)
                val_loss_per_horse = val_loss / n_horses

                if verbose:
                    print(
                        f"Evaluation only for {n_horses} horses: loss per horse None, "
                        f"val loss per horse: {val_loss_per_horse:.3f}\n"
                    )
                continue

            if x_val.size == 0:
                history = model.fit(x=x, y=y, verbose=int(verbose))
            else:
                history = model.fit(
                    x=x, y=y, validation_data=(x_val, y_val), verbose=int(verbose)
                )

            if verbose:
                loss_per_horse = history.history["loss"][0] / n_horses
                accuracy = history.history["categorical_accuracy"][0]
                if x_val.size == 0:
                    val_loss_per_horse = np.nan
                    val_accuracy = np.nan

                else:
                    val_loss_per_horse = history.history["val_loss"][0] / n_horses
                    val_accuracy = history.history["val_categorical_accuracy"][0]
                print(
                    f"Training for {n_horses} horses ({x.shape[0]} races, val {x_val.shape[0]} races): loss per horse: "
                    f"{loss_per_horse:.3f}, val loss per horse: {val_loss_per_horse:.3f} "
                    f"Train Accuracy: {accuracy:.1%}, Val Accuracy: {val_accuracy:.1%}\n"
                )

        if verbose:
            print("=" * 80)
            print()

    return winning_model


def train_on_all_races(
    source: str, winning_model: AbstractWinningModel, n_epochs: int, verbose: bool = False
) -> AbstractWinningModel:
    """Only for deeplearning models (tf.keras like)"""
    pass
