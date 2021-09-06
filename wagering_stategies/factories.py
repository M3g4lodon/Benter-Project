import functools
from typing import Callable
from typing import Optional

import numpy as np

from constants import PMU_MINIMUM_BET_SIZE
from utils.expected_return import get_race_expected_return
from winning_horse_models import AbstractWinningModel


def _betting_on_best_exp_return_thresholded_winning_proba_expected_returns(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    _minimum_winning_probabilities: float,
    _expected_return_threshold: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns bettings putting all capital fraction on the best expected return on
    winning probabilities above a given threshold (if it is above a threshold on
    expected return and one)"""
    assert (previous_stakes is not None) or (odds is not None)
    if previous_stakes is None:
        odds_race = odds
    else:
        odds_race = (1 / previous_stakes) / np.sum(1 / previous_stakes)
    n_horses = x_race.shape[0]

    y_hat_race = winning_model.predict(
        x=np.expand_dims(x_race, axis=0), odds=odds_race
    )[0, :]
    y_hat_race = np.where(
        y_hat_race > _minimum_winning_probabilities, y_hat_race, np.zeros((n_horses,))
    )

    expected_return_race = (
        get_race_expected_return(
            y_hat_race=y_hat_race,
            track_take=track_take,
            previous_stakes=previous_stakes,
            odds=odds,
            race_bet=PMU_MINIMUM_BET_SIZE * np.ones((n_horses,)),
        )
        / PMU_MINIMUM_BET_SIZE
    )
    max_expected_return = expected_return_race.max()

    if max_expected_return <= _expected_return_threshold:
        return np.zeros_like(y_hat_race)

    betting = expected_return_race == max_expected_return
    assert betting.sum()
    betting = betting / betting.sum()
    betting = betting * capital_fraction
    return betting


def betting_on_best_expected_return_thresholded_expected_return_factory(
    expected_return_threshold: float,
) -> Callable:

    return functools.partial(
        _betting_on_best_exp_return_thresholded_winning_proba_expected_returns,
        _minimum_winning_probabilities=0.0,
        _expected_return_threshold=expected_return_threshold,
    )


def betting_on_best_expected_return_thresholded_winning_probabilities_factory(
    minimum_winning_probabilities: float,
) -> Callable:
    return functools.partial(
        _betting_on_best_exp_return_thresholded_winning_proba_expected_returns,
        _minimum_winning_probabilities=minimum_winning_probabilities,
        _expected_return_threshold=0.0,
    )


def betting_on_best_exp_return_thresholded_winning_proba_expected_returns_factory(
    minimum_winning_probabilities: float, expected_return_threshold: float
) -> Callable:
    return functools.partial(
        _betting_on_best_exp_return_thresholded_winning_proba_expected_returns,
        _minimum_winning_probabilities=minimum_winning_probabilities,
        _expected_return_threshold=expected_return_threshold,
    )
