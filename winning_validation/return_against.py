from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from constants import PMU_MINIMUM_BET_SIZE
from constants import Sources
from constants import SplitSets
from constants import XFormats
from constants import YFormats
from utils import expected_return
from utils import import_data
from winning_horse_models import AbstractWinningModel
from winning_horse_models import ModelNotCreatedOnceError
from winning_horse_models import N_FEATURES
from winning_horse_models.baselines import RandomModel


def compute_race_return_against(
    rank_race: np.array, predicted_probabilities: np.array, base_probabilities: np.array
):
    assert rank_race.shape == predicted_probabilities.shape == base_probabilities.shape

    return_against = np.where(
        rank_race == 1,
        (predicted_probabilities - base_probabilities) / base_probabilities,
        (base_probabilities - predicted_probabilities),
    )
    return_against = np.sum(return_against, axis=1)

    return return_against


def get_race_rectified_pari_mutual_probabilities(race_previous_stakes: np.array):
    pari_mutual_odds = expected_return.get_race_odds(
        track_take=0.0,
        previous_stakes=race_previous_stakes,
        race_bet=PMU_MINIMUM_BET_SIZE * np.ones_like(race_previous_stakes),
    )  # To avoid infinite return
    return PMU_MINIMUM_BET_SIZE / pari_mutual_odds


def get_rectified_pari_mutual_probabilities(previous_stakes: np.array):
    return np.apply_along_axis(
        func1d=get_race_rectified_pari_mutual_probabilities, axis=1, arr=previous_stakes
    )


def compute_return_against_odds(
    source: Sources,
    same_races_support: bool,
    winning_model: AbstractWinningModel,
    selected_features_index: Optional[List[int]] = None,
    extra_features_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    preprocessing: bool = True,
    verbose: bool = False,
) -> dict:
    assert len(set(selected_features_index)) == len(selected_features_index)
    assert min(selected_features_index) >= 0
    assert N_FEATURES > max(selected_features_index)

    features_index = selected_features_index + ([-1] if extra_features_func else [])
    min_horse, max_horse = import_data.get_min_max_horse(source=source)

    res = {
        "source": source,
        "same_races_support": same_races_support,
        "winning_model": winning_model.__class__.__name__,
        "min_horse": min_horse,
        "max_horse": max_horse,
        "n_horses_return_against_odds": {},
    }

    np.random.seed(42)

    for n_horses in range(min_horse, max_horse + 1):

        x_race, rank_race, race_dfs = import_data.get_races_per_horse_number(
            source=source,
            n_horses=n_horses,
            on_split=SplitSets.VAL,
            x_format=XFormats.SEQUENTIAL,
            y_format=YFormats.RANK,
            extra_features_func=extra_features_func,
            preprocessing=preprocessing,
        )

        if selected_features_index and x_race.size != 0:
            x_race = x_race[:, :, features_index]

        if x_race.size == 0:
            continue
        previous_stakes = np.stack(
            arrays=[race_df["totalEnjeu"].values for race_df in race_dfs], axis=0
        )
        try:
            model_prediction = winning_model.predict(x=x_race)
        except ModelNotCreatedOnceError:
            res["n_horses_return_against_odds"][n_horses] = {
                "message": f"Model was not created for {n_horses}"
            }
            continue

        random_prediction = RandomModel().predict(x=x_race)
        race_ps_notna_index = np.logical_not(np.isnan(previous_stakes)).all(axis=1)
        rectified_pari_mutual_probas = get_rectified_pari_mutual_probabilities(
            previous_stakes=np.nan_to_num(previous_stakes)  # Put nan to zero
        )
        rank_race_ = rank_race

        if same_races_support:
            model_prediction = model_prediction[race_ps_notna_index]
            random_prediction = random_prediction[race_ps_notna_index]
            rectified_pari_mutual_probas = rectified_pari_mutual_probas[
                race_ps_notna_index
            ]
            rank_race_ = rank_race_[race_ps_notna_index]

        predicted_return_against_odds = compute_race_return_against(
            rank_race=rank_race_,
            predicted_probabilities=model_prediction,
            base_probabilities=rectified_pari_mutual_probas,
        )

        random_predicted_return_agnst_odds = compute_race_return_against(
            rank_race=rank_race_,
            predicted_probabilities=random_prediction,
            base_probabilities=rectified_pari_mutual_probas,
        )

        if not race_ps_notna_index.all() and not same_races_support:
            message = (
                f"[Be careful, only {race_ps_notna_index.sum()} races "
                f"({race_ps_notna_index.mean():.2%}) are kept "
                f"in odds analysis for {n_horses} horses, ({x_race.shape[0]} "
                f"races in total)]"
            )
        elif not race_ps_notna_index.all() and same_races_support:
            message = (
                f"Comparing on same races w/ {n_horses} horses with odds "
                f"{np.sum(race_ps_notna_index)} "
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

        res["n_horses_return_against_odds"][n_horses] = {
            "predicted_return_against_odds": predicted_return_against_odds,
            "random_predicted_return_against_odds": random_predicted_return_agnst_odds,
            "race_ps_notna_index": race_ps_notna_index,
            "race_ids": [race_df["race_id"].iloc[0] for race_df in race_dfs],
            "message": message,
        }
        if verbose:
            print(
                f"Mean Return against odds: "
                f"{predicted_return_against_odds.mean():.3f} "
                f"(std: {predicted_return_against_odds.std():.2f}) "
                f"Random: {random_predicted_return_agnst_odds.mean():.3f} "
                f"(std: {random_predicted_return_agnst_odds.std():.2f})"
            )
            print()

    return res


def compute_overall_average(
    return_against_odds: dict,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:

    predicted_returns = []
    random_returns = []
    for return_againsts in return_against_odds["n_horses_return_against_odds"].values():
        if "predicted_return_against_odds" in return_againsts:
            predicted_returns.append(return_againsts["predicted_return_against_odds"])
            random_returns.append(
                return_againsts["random_predicted_return_against_odds"]
            )
    if not predicted_returns:
        return None, None, None, None
    predicted_returns = np.concatenate(predicted_returns)
    random_returns = np.concatenate(random_returns)

    return (
        np.mean(predicted_returns),
        np.mean(random_returns),
        np.median(predicted_returns),
        np.median(random_returns),
    )
