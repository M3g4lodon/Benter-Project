from typing import Callable
from typing import Optional

import numpy as np
from scipy.stats import stats

from utils import import_data
from winning_horse_models import AbstractWinningModel
from winning_horse_models import ModelNotCreatedOnceError
from winning_horse_models.baselines import RandomModel


# TODO error on unseen horses
# TODO more metrics here https://gist.github.com/bwhite/3726239
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


# Predicted Proba of actual race results
def compute_rank_proba(rank_race: np.array, k: int, proba_distribution: np.array):

    if proba_distribution.size == 0:
        return np.nan
    assert proba_distribution.shape[0] >= k
    assert proba_distribution.shape == rank_race.shape

    remaining_horses_indices = list(range(proba_distribution.shape[0]))
    predicted_proba = 1.0
    for rank in range(1, k + 1):
        rank_pos = np.argwhere(rank_race == rank)

        sub_rank_proba = proba_distribution[remaining_horses_indices]
        sub_rank_proba = sub_rank_proba / sub_rank_proba.sum()

        probas = sub_rank_proba[np.argwhere(remaining_horses_indices == rank_pos)[:, 1]]
        predicted_proba *= np.prod(probas)
        remaining_horses_indices = [
            r for r in remaining_horses_indices if r not in rank_pos
        ]
        if not remaining_horses_indices:
            break

    return predicted_proba


def compute_validation_error(
    source: str,
    k: int,
    validation_method: Callable,
    winning_model: AbstractWinningModel,
) -> dict:
    assert k > 0

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
        print(message)
        res["message"] = message
        return res

    np.random.seed(42)

    for n_horses in range(max(k, min_horse), max_horse + 1):

        x_race, rank_race, odds_race = import_data.get_races_per_horse_number(
            y_format="rank",
            x_format="sequential_per_horse",
            n_horses=n_horses,
            on_split="val",
            source=source,
        )
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
        else:
            assert validation_method == precision_at_k
            validation_name = f"precision at rank {k}"

        res["n_horses_validations"][n_horses] = {
            "n_val_races": x_race.shape[0],
            "validation_method_message": validation_name,
            "validation_values": topk_arr,
            "random_validation_values": random_topk_arr,
            "odds_validation_values": odds_topk_arr,
        }
        print(
            f"For races w/ {n_horses} horses, "
            f"{x_race.shape[0]} races in val, "
            f"{validation_name}: {np.mean(topk_arr):.3%} "
            f"(Random: {np.mean(random_topk_arr):.3%}, "
            f"Odds {np.mean(odds_topk_arr):.3%})"
        )
    return res


def compute_predicted_proba_on_actual_races(
    k: int, source: str, same_races_support: bool, winning_model: AbstractWinningModel
) -> dict:
    assert k > 1

    min_horse, max_horse = import_data.get_min_max_horse(source=source)

    res = {
        "source": source,
        "k": k,
        "same_races_support": same_races_support,
        "winning_model": winning_model.__class__.__name__,
        "min_horse": min_horse,
        "max_horse": max_horse,
        "n_horses_predicted_probas": {},
    }

    if k > max_horse:
        message = f"k={k} is above the maximum number of horse per race ({max_horse}"
        print(message)
        res["message"] = message
        return res

    np.random.seed(42)

    for n_horses in range(max(k, min_horse), max_horse + 1):

        x_race, rank_race, odds_race = import_data.get_races_per_horse_number(
            y_format="rank",
            x_format="sequential_per_horse",
            n_horses=n_horses,
            on_split="val",
            source=source,
        )

        if x_race.size == 0:
            continue

        model_prediction = winning_model.predict(x=x_race)
        random_prediction = RandomModel().predict(x=x_race)
        race_odds_notna_index = np.logical_not(np.isnan(odds_race)).all(axis=1)
        pari_mutual_proba = (1 / odds_race)[race_odds_notna_index]
        rank_race_ = rank_race

        if same_races_support:
            model_prediction = model_prediction[race_odds_notna_index]
            random_prediction = random_prediction[race_odds_notna_index]
            rank_race_ = rank_race_[race_odds_notna_index]

        predicted_probas = np.array(
            [
                compute_rank_proba(proba_distribution=proba_dist, rank_race=rank_r, k=k)
                for proba_dist, rank_r in zip(model_prediction, rank_race_)
            ]
        )
        random_predicted_probas = np.array(
            [
                compute_rank_proba(proba_distribution=proba_dist, rank_race=rank_r, k=k)
                for proba_dist, rank_r in zip(random_prediction, rank_race)
            ]
        )

        if not race_odds_notna_index.all() and not same_races_support:
            print(
                f"[Be careful, only {race_odds_notna_index.sum()} races "
                f"({race_odds_notna_index.mean():.2%}) are kept "
                f"in odds analysis for {n_horses} horses, ({x_race.shape[0]} "
                f"races in total)]"
            )
        if same_races_support and not race_odds_notna_index.all():
            print(
                f"Comparing on same races w/ {n_horses} horses with odds "
                f"{np.sum(race_odds_notna_index)} "
                f"races ({x_race.shape[0]} races in total)"
            )
        if race_odds_notna_index.all():
            print(
                f"Comparing on all races w/ {n_horses} horses "
                f"({x_race.shape[0]} races in total)"
            )
        odds_predicted_probas = np.array(
            [
                compute_rank_proba(proba_distribution=proba_dist, rank_race=rank_r, k=k)
                for proba_dist, rank_r in zip(pari_mutual_proba, rank_race_)
            ]
        )

        res["n_horses_predicted_probas"][n_horses] = {
            "predicted_probabilities": predicted_probas,
            "random_probabilities": random_predicted_probas,
            "odds_probabilities": odds_predicted_probas,
            "race_odds_notna_index": race_odds_notna_index,
        }
        print(
            f"Mean Predicted probas of actual race result: "
            f"{predicted_probas.mean():.3%} "
            f"(Random: {random_predicted_probas.mean():.3%}, "
            f"Odds: {odds_predicted_probas.mean():.3%})"
        )
        print()

    return res
