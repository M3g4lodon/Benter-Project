from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import stats

from constants import Sources
from constants import SplitSets
from utils import import_data
from utils import preprocess
from winning_horse_models import AbstractWinningModel
from winning_horse_models import ModelNotCreatedOnceError
from winning_horse_models.baselines import RandomModel


def _true_ordered_topk(rank, k: int):
    return np.argwhere(rank.argsort().argsort() <= (k - 1)).flatten()


def exact_top_k(rank_race, rank_hat, k: int):
    return np.all(
        rank_hat[_true_ordered_topk(rank=rank_race, k=k)]
        == rank_race[_true_ordered_topk(rank=rank_race, k=k)]
    )


def same_top_k(rank_race, rank_hat, k: int):
    return rank_hat[_true_ordered_topk(rank=rank_race, k=k)].max() <= k


def precision_at_k(rank_race, rank_hat, k: int):
    return np.mean(rank_hat[_true_ordered_topk(rank=rank_race, k=k)] <= k)


def kappa_cohen_like(rank_race, rank_hat, k: Optional[int] = None):
    n_horses = rank_race.shape[0]
    return (
        (rank_hat[_true_ordered_topk(rank=rank_race, k=1)][0] == 1) - 1 / n_horses
    ) / (1 - 1 / n_horses)


def compute_validation_error(
    source: Sources,
    k: int,
    validation_method: Callable,
    winning_model: AbstractWinningModel,
    selected_features_index: Optional[List[int]] = None,
    extra_features_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    verbose: bool = False,
) -> dict:
    assert k > 0
    if selected_features_index is None:
        selected_features_index = list(range(winning_model.n_features))
    assert len(set(selected_features_index)) == len(selected_features_index)
    assert 0 <= min(selected_features_index)
    assert preprocess.get_n_preprocessed_feature_columns(source=source) > max(
        selected_features_index
    )
    features_index = selected_features_index + ([-1] if extra_features_func else [])
    min_horse, max_horse = import_data.get_min_max_horse(source=source)
    res = {
        "source": source,
        "k": k,
        "validation_method": validation_method.__name__,
        "winning_model": winning_model.__class__.__name__,
        "n_horses_validations": {},
        "min_horse": min_horse,
        "max_horse": max_horse,
    }
    if k > max_horse:
        message = f"k={k} is above the maximum number of horse per race ({max_horse}"
        if verbose:
            print(message)
        res["message"] = message
        return res

    np.random.seed(42)

    for n_horses in range(max(k, min_horse), max_horse + 1):

        x_race, rank_race, race_dfs = import_data.get_races_per_horse_number(
            source=source,
            n_horses=n_horses,
            on_split=SplitSets.VAL,
            x_format="sequential_per_horse",
            y_format="rank",
            extra_features_func=extra_features_func,
        )

        if selected_features_index and x_race.size != 0:
            x_race = x_race[:, :, features_index]

        if x_race.size == 0:
            res["n_horses_validations"][n_horses] = {"n_val_races": 0}
            continue
        try:
            y_hat = winning_model.predict(x=x_race)
        except ModelNotCreatedOnceError:
            res["n_horses_validations"][n_horses] = {"message": "Model was not created"}
            continue

        topk_arr = np.array(
            [
                validation_method(rank_race=rank_r, rank_hat=rank_h, k=k)
                for rank_r, rank_h in zip(
                    rank_race,
                    np.apply_along_axis(
                        lambda x: stats.rankdata(a=-x, method="min"), arr=y_hat, axis=1
                    ),
                )
            ]
        )
        random_topk_arr = np.array(
            [
                validation_method(rank_r, rank_h, k=k)
                for rank_r, rank_h in zip(
                    rank_race,
                    np.apply_along_axis(
                        lambda x: stats.rankdata(a=-x, method="min"),
                        arr=RandomModel().predict(x=x_race),
                        axis=1,
                    ),
                )
            ]
        )
        odds_race = np.stack(
            arrays=[race_df["odds"].values for race_df in race_dfs], axis=0
        )
        odds_topk_arr = np.array(
            [
                validation_method(rank_r, rank_h, k=k)
                for rank_r, rank_h in zip(
                    rank_race,
                    np.apply_along_axis(
                        lambda x: stats.rankdata(a=x, method="min"),
                        arr=odds_race,
                        axis=1,
                    ),
                )
            ]
        )

        if validation_method == exact_top_k:
            validation_name = f"top {k} in right order"
        elif validation_method == same_top_k:
            validation_name = f"top {k} w/o order"
        elif validation_method == kappa_cohen_like:
            validation_name = f"Kappa Cohen like (optional k:{k})"
        else:
            assert validation_method == precision_at_k
            validation_name = f"precision at rank {k}"

        res["n_horses_validations"][n_horses] = {
            "n_val_races": x_race.shape[0],
            "validation_method_message": validation_name,
            "validation_values": topk_arr,
            "random_validation_values": random_topk_arr,
            "odds_validation_values": odds_topk_arr,
            "race_ids": [race_df["race_id"].iloc[0] for race_df in race_dfs],
        }
        if verbose:
            print(
                f"For races w/ {n_horses} horses, "
                f"{x_race.shape[0]} races in val, "
                f"{validation_name}: {np.mean(topk_arr):.3%} "
                f"(Random: {np.mean(random_topk_arr):.3%}, "
                f"Odds {np.mean(odds_topk_arr):.3%})"
            )
    return res


def compute_overall_average(
    validation_errors: dict,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:

    validation_values = []
    random_values = []
    odds_values = []
    for validation in validation_errors["n_horses_validations"].values():
        if "validation_values" in validation:
            validation_values.append(validation["validation_values"])
            random_values.append(validation["random_validation_values"])
            odds_values.append(validation["odds_validation_values"])
    if not validation_values:
        return None, None, None
    validation_values = np.concatenate(validation_values)
    random_values = np.concatenate(random_values)
    odds_values = np.concatenate(odds_values)

    return np.mean(validation_values), np.mean(random_values), np.mean(odds_values)
