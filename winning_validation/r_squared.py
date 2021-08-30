from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from constants import Sources
from utils import import_data
from winning_horse_models import AbstractWinningModel
from winning_horse_models.baselines import RandomModel


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


def compute_predicted_proba_on_actual_races(
    k: int,
    source: Sources,
    same_races_support: bool,
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

    features_index = selected_features_index + ([-1] if extra_features_func else [])

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
        if verbose:
            print(message)
        res["message"] = message
        return res

    np.random.seed(42)

    for n_horses in range(max(k, min_horse), max_horse + 1):

        x_race, rank_race, race_dfs = import_data.get_races_per_horse_number(
            source=source,
            n_horses=n_horses,
            on_split="val",
            x_format="sequential_per_horse",
            y_format="rank",
            extra_features_func=extra_features_func,
        )

        if selected_features_index and x_race.size != 0:
            x_race = x_race[:, :, features_index]

        if x_race.size == 0:
            continue
        odds_race = np.stack(
            arrays=[race_df["odds"].values for race_df in race_dfs], axis=0
        )

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
            message = (
                f"[Be careful, only {race_odds_notna_index.sum()} races "
                f"({race_odds_notna_index.mean():.2%}) are kept "
                f"in odds analysis for {n_horses} horses, ({x_race.shape[0]} "
                f"races in total)]"
            )
        elif not race_odds_notna_index.all() and same_races_support:
            message = (
                f"Comparing on same races w/ {n_horses} horses with odds "
                f"{np.sum(race_odds_notna_index)} "
                f"races ({x_race.shape[0]} races in total)"
            )

        else:
            message = (
                f"Comparing on all races w/ {n_horses} horses "
                f"({x_race.shape[0]} races in total)"
            )
        assert message
        if verbose:
            print(message)
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
            "race_ids": [race_df["race_id"].iloc[0] for race_df in race_dfs],
            "message": message,
        }
        if verbose:
            print(
                f"Mean Predicted probas of actual race result: "
                f"{predicted_probas.mean():.3%} "
                f"(Random: {random_predicted_probas.mean():.3%}, "
                f"Odds: {odds_predicted_probas.mean():.3%})"
            )
            print()

    return res


# McFadden's pseudo-R squared
def compute_mcfadden_r_squared(
    source: Sources,
    winning_model: AbstractWinningModel,
    selected_features_index: Optional[List[int]] = None,
    extra_features_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, dict]]:
    proba_at_winning_horse = compute_predicted_proba_on_actual_races(
        k=1,
        source=source,
        same_races_support=True,
        winning_model=winning_model,
        selected_features_index=selected_features_index,
        extra_features_func=extra_features_func,
        verbose=verbose,
    )

    res: Dict[str, Union[float, dict]] = {"n_horses_r_squared": {}}
    for n_horses, proba in proba_at_winning_horse["n_horses_predicted_probas"].items():
        n_races = (proba["odds_probabilities"] != 0).sum()
        n_rejected_races = (proba["odds_probabilities"] == 0).sum()
        assert np.all(proba["predicted_probabilities"] > 0)
        assert np.all(proba["random_probabilities"] > 0)

        model_proba = proba["predicted_probabilities"][proba["odds_probabilities"] > 0]
        model_r_squared = 1 - np.sum(np.log(model_proba)) / (
            -n_races * np.log(n_horses)
        )

        odds_proba = proba["odds_probabilities"][proba["odds_probabilities"] > 0]
        odds_r_squared = 1 - np.sum(np.log(odds_proba)) / (-n_races * np.log(n_horses))

        random_proba = proba["random_probabilities"][proba["odds_probabilities"] > 0]
        random_r_squared = 1 - np.sum(np.log(random_proba)) / (
            -n_races * np.log(n_horses)
        )  # Should be close to zero

        assert n_rejected_races < 5

        if verbose:

            print(
                f"On {n_races} races with {n_horses} horses,"
                f"R² of winning model: {model_r_squared:.2f}, "
                f"R² of odds: {odds_r_squared:.2f}, "
                f"[R² of random model: {random_r_squared:.2f} (should be closed to 0)]"
            )
        res["n_horses_r_squared"][n_horses] = {
            "model_r_squared": model_r_squared,
            "odds_r_squared": odds_r_squared,
            "random_r_squared": random_r_squared,
            "n_races": n_races,
            "n_rejected_races": n_rejected_races,
        }
    n_races_per_n_horses = [
        n_horse_metrics["n_races"]
        for n_horse_metrics in res["n_horses_r_squared"].values()
    ]

    average_model_r_squared = np.average(
        a=[
            n_horse_metrics["model_r_squared"]
            for n_horse_metrics in res["n_horses_r_squared"].values()
        ],
        weights=n_races_per_n_horses,
    )
    res["average_model_r_squared"] = average_model_r_squared

    average_odds_r_squared = np.average(
        a=[
            n_horse_metrics["odds_r_squared"]
            for n_horse_metrics in res["n_horses_r_squared"].values()
        ],
        weights=n_races_per_n_horses,
    )
    res["average_odds_r_squared"] = average_odds_r_squared
    average_random_r_squared = np.average(
        a=[
            n_horse_metrics["random_r_squared"]
            for n_horse_metrics in res["n_horses_r_squared"].values()
        ],
        weights=n_races_per_n_horses,
    )
    res["average_random_r_squared"] = average_random_r_squared

    if verbose:
        print(
            f"Average Winning Model R²: {average_model_r_squared:.3f}, "
            f"Average Odds R²: {average_odds_r_squared:.3f} "
            f"(Average Random R²: {average_random_r_squared:.3f})"
        )

    combined_model_r_squared = np.log(
        np.concatenate(
            [
                proba["predicted_probabilities"][proba["odds_probabilities"] > 0]
                for n_horses, proba in proba_at_winning_horse[
                    "n_horses_predicted_probas"
                ].items()
            ]
        )
    ).sum()
    combined_model_r_squared /= -np.sum(
        [
            (proba["odds_probabilities"] != 0).sum() * np.log(n_horses)
            for n_horses, proba in proba_at_winning_horse[
                "n_horses_predicted_probas"
            ].items()
        ]
    )
    combined_model_r_squared = 1 - combined_model_r_squared
    res["combined_model_r_squared"] = combined_model_r_squared

    combined_odds_r_squared = np.log(
        np.concatenate(
            [
                proba["odds_probabilities"][proba["odds_probabilities"] > 0]
                for n_horses, proba in proba_at_winning_horse[
                    "n_horses_predicted_probas"
                ].items()
            ]
        )
    ).sum()
    combined_odds_r_squared /= -np.sum(
        [
            (proba["odds_probabilities"] != 0).sum() * np.log(n_horses)
            for n_horses, proba in proba_at_winning_horse[
                "n_horses_predicted_probas"
            ].items()
        ]
    )
    combined_odds_r_squared = 1 - combined_odds_r_squared
    res["combined_odds_r_squared"] = combined_odds_r_squared

    combined_random_r_squared = np.log(
        np.concatenate(
            [
                proba["random_probabilities"][proba["odds_probabilities"] > 0]
                for n_horses, proba in proba_at_winning_horse[
                    "n_horses_predicted_probas"
                ].items()
            ]
        )
    ).sum()
    combined_random_r_squared /= -np.sum(
        [
            (proba["odds_probabilities"] != 0).sum() * np.log(n_horses)
            for n_horses, proba in proba_at_winning_horse[
                "n_horses_predicted_probas"
            ].items()
        ]
    )
    combined_random_r_squared = 1 - combined_random_r_squared
    res["combined_random_r_squared"] = combined_random_r_squared

    if verbose:
        print(
            f"Combined Winning Model R²: {combined_model_r_squared:.3f}, "
            f"Combined Odds R²: {combined_odds_r_squared:.3f} "
            f"(Combined Random R²: {combined_random_r_squared:.3f}) "
        )

    delta_model_odds = combined_model_r_squared - combined_odds_r_squared
    res["delta_model_odds"] = delta_model_odds

    if verbose:
        print(f"Delta Model vs Odds: {delta_model_odds:.3f}")
    return res
