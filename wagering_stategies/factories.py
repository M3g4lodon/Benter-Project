import functools
from typing import Callable

import numpy as np

from constants import PMU_MINIMUM_BET_SIZE
from utils.expected_return import get_race_expected_return
from winning_horse_models import AbstractWinningModel


def _betting_on_best_expected_return_thresholded_expected_return(
    x_race: np.array,
    previous_stakes: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    _expected_return_threshold: float,
) -> np.array:
    """Returns bettings putting all capital fraction on the best expected return
    (if it is above threshold on expected return)"""
    n_horses = x_race.shape[0]

    y_hat_race = winning_model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

    expected_return_race = (
        get_race_expected_return(
            y_hat_race=y_hat_race,
            track_take=track_take,
            previous_stakes=previous_stakes,
            race_bet=PMU_MINIMUM_BET_SIZE * np.ones((n_horses,)),
        )
        / PMU_MINIMUM_BET_SIZE
    )
    if expected_return_race.max() <= 1.0:
        return np.zeros((n_horses,))
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
        _betting_on_best_expected_return_thresholded_expected_return,
        _expected_return_threshold=expected_return_threshold,
    )


def _betting_on_best_expected_return_thresholded_winning_probabilities(
    x_race: np.array,
    previous_stakes: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    _minimum_winning_probabilities: float,
) -> np.array:
    """Returns bettings putting all capital fraction on the best expected return on winning probabilities
    above a given threshold (if it is above one)"""
    n_horses = x_race.shape[0]

    y_hat_race = winning_model.predict(x=np.expand_dims(x_race, axis=0))[0, :]
    y_hat_race = np.where(
        y_hat_race > _minimum_winning_probabilities, y_hat_race, np.zeros((n_horses,))
    )

    expected_return_race = (
        get_race_expected_return(
            y_hat_race=y_hat_race,
            track_take=track_take,
            previous_stakes=previous_stakes,
            race_bet=PMU_MINIMUM_BET_SIZE * np.ones((n_horses,)),
        )
        / PMU_MINIMUM_BET_SIZE
    )
    max_expected_return = expected_return_race.max()
    if max_expected_return <= 0.0:
        return np.zeros((n_horses,))

    betting = expected_return_race == max_expected_return
    assert betting.sum(), betting
    betting = betting / betting.sum()
    betting = betting * capital_fraction
    return betting


def betting_on_best_expected_return_thresholded_winning_probabilities_factory(
    minimum_winning_probabilities: float,
) -> Callable:
    return functools.partial(
        _betting_on_best_expected_return_thresholded_winning_probabilities,
        _minimum_winning_probabilities=minimum_winning_probabilities,
    )


def _betting_on_best_expected_return_thresholded_winning_probabilities_expected_returns(
    x_race: np.array,
    previous_stakes: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    _minimum_winning_probabilities: float,
    _expected_return_threshold: float,
) -> np.array:
    """Returns bettings putting all capital fraction on the best expected return on winning probabilities
    above a given threshold (if it is above a threshold on expected return and one)"""
    n_horses = x_race.shape[0]

    y_hat_race = winning_model.predict(x=np.expand_dims(x_race, axis=0))[0, :]
    y_hat_race = np.where(
        y_hat_race > _minimum_winning_probabilities, y_hat_race, np.zeros((n_horses,))
    )

    expected_return_race = (
        get_race_expected_return(
            y_hat_race=y_hat_race,
            track_take=track_take,
            previous_stakes=previous_stakes,
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


def betting_on_best_expected_return_thresholded_winning_probabilities_expected_returns_factory(
    minimum_winning_probabilities: float, expected_return_threshold: float
) -> Callable:
    return functools.partial(
        _betting_on_best_expected_return_thresholded_winning_probabilities_expected_returns,
        _minimum_winning_probabilities=minimum_winning_probabilities,
        _expected_return_threshold=expected_return_threshold,
    )
