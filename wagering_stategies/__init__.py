from typing import Optional

import numpy as np

from constants import PMU_MINIMUM_BET_SIZE
from utils.expected_return import get_race_expected_return
from winning_horse_models import AbstractWinningModel


def race_betting_proportional_positive_return(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns bettings proportional to computed positive returns"""
    assert (previous_stakes is not None) or (odds is not None)
    n_horses = x_race.shape[0]
    if previous_stakes is None:
        odds_race = odds
    else:
        odds_race = (1 / previous_stakes) / np.sum(1 / previous_stakes)
    y_hat_race = winning_model.predict(
        x=np.expand_dims(x_race, axis=0), odds=odds_race
    )[0, :]

    if np.all(np.isnan(y_hat_race)):
        return y_hat_race
    expected_return_race = (
        get_race_expected_return(
            y_hat_race=y_hat_race,
            track_take=track_take,
            previous_stakes=previous_stakes,
            race_bet=PMU_MINIMUM_BET_SIZE * np.ones((n_horses,)),
            odds=odds,
        )
        / PMU_MINIMUM_BET_SIZE
    )
    positives_returns = np.where(
        expected_return_race > 1,
        expected_return_race,
        np.zeros_like(expected_return_race),
    )
    betting_race = positives_returns / np.sum(positives_returns, keepdims=True)
    betting_race[np.isnan(betting_race)] = 0
    betting_race = betting_race * capital_fraction

    return betting_race


def race_betting_best_expected_return(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns bettings putting all capital fraction on the best expected return
    (if it is positive)"""
    assert (previous_stakes is not None) or (odds is not None)
    n_horses = x_race.shape[0]
    if previous_stakes is None:
        odds_race = odds
    else:
        odds_race = (1 / previous_stakes) / np.sum(1 / previous_stakes)

    y_hat_race = winning_model.predict(
        x=np.expand_dims(x_race, axis=0), odds=odds_race
    )[0, :]
    if np.all(np.isnan(y_hat_race)):
        return y_hat_race
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
    max_expected_return = np.nanmax(expected_return_race)
    if max_expected_return <= 0.0:
        return np.zeros((n_horses,))

    betting = expected_return_race == max_expected_return
    assert betting.sum(), (betting.sum(), betting, max_expected_return, y_hat_race)
    betting = betting / betting.sum()
    betting = betting * capital_fraction
    return betting


def race_bettings_kelly(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: Optional[float],
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns Kelly criterion to compute optimal betting
    (capital_fraction can be omitted here since Kelly criterion already computes the
     optimal capital fraction to risk)

     References:
         - https://en.wikipedia.org/wiki/Kelly_criterion#Multiple_outcomes
    """
    assert (previous_stakes is not None) or (odds is not None)

    # Race odds, with little adjustment when no bets has been put on a horse
    if previous_stakes is not None:
        odds_race = np.where(
            previous_stakes > 0,
            previous_stakes.sum() / previous_stakes,
            (previous_stakes.sum() + PMU_MINIMUM_BET_SIZE)
            / PMU_MINIMUM_BET_SIZE
            * np.ones_like(previous_stakes),
        )
    else:
        odds_race = odds
    y_hat_race = winning_model.predict(
        x=np.expand_dims(x_race, axis=0), odds=odds_race
    )[0, :]
    if np.all(np.isnan(y_hat_race)):
        return y_hat_race
    expected_return_race = y_hat_race * odds_race * (1 - track_take)

    S = []
    R_S = 1
    order_index = (-expected_return_race).argsort()

    for kth in order_index:
        if expected_return_race[kth] < R_S:
            break
        S.append(kth)
        R_S = 1 - np.sum(y_hat_race[S]) / (
            1 - np.sum(1 / (odds_race * (1 - track_take)))
        )

    if not S:
        return np.zeros_like(y_hat_race)

    return np.clip(
        (expected_return_race - R_S) / (1 - track_take) / odds_race, a_min=0, a_max=None
    )


def race_betting_proportional_winning_proba(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns bettings proportional to winning proba given by winning_model"""
    assert (previous_stakes is not None) or (odds is not None)
    if previous_stakes is None:
        odds_race = odds
    else:
        odds_race = (1 / previous_stakes) / np.sum(1 / previous_stakes)
    y_hat_race = winning_model.predict(
        x=np.expand_dims(x_race, axis=0), odds=odds_race
    )[0, :]
    if np.all(np.isnan(y_hat_race)):
        return y_hat_race
    assert np.isclose(y_hat_race.sum(), 1.0)
    betting = y_hat_race * capital_fraction

    return betting


def race_betting_best_winning_proba(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns bettings with all capital_fraction on the best horse
    according to winning_model"""
    assert (previous_stakes is not None) or (odds is not None)
    if previous_stakes is None:
        odds_race = odds
    else:
        odds_race = 1 / previous_stakes
    y_hat_race = winning_model.predict(
        x=np.expand_dims(x_race, axis=0), odds=odds_race
    )[0, :]
    if np.all(np.isnan(y_hat_race)):
        return y_hat_race
    assert np.isclose(y_hat_race.sum(), 1.0)
    betting = y_hat_race == np.nanmax(y_hat_race)
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_betting_best_winning_proba_not_max_pari_mutual_proba(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns the best winning proba horse that is not the horse
    with the most bets on"""
    assert (previous_stakes is not None) or (odds is not None)
    if previous_stakes is None:
        odds_race = odds
    else:
        odds_race = 1 / previous_stakes
    y_hat_race = winning_model.predict(
        x=np.expand_dims(x_race, axis=0), odds=odds_race
    )[0, :]
    if np.all(np.isnan(y_hat_race)):
        return y_hat_race
    assert np.isclose(y_hat_race.sum(), 1.0)
    if previous_stakes is not None:
        betting = np.logical_and(
            y_hat_race == np.nanmax(y_hat_race),
            previous_stakes != np.nanmax(previous_stakes),
        )
    else:
        assert odds is not None
        betting = np.logical_and(
            y_hat_race == np.nanmax(y_hat_race), (1 / odds) != np.nanmax(1 / odds)
        )
    if np.sum(betting) == 0:
        betting = y_hat_race == np.sort(y_hat_race)[-2]
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_random_one_horse(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """(Baseline)Returns uniformly draws of betting"""
    assert (previous_stakes is not None) or (odds is not None)

    n_horses = x_race.shape[0]
    betting = np.random.rand(n_horses)
    betting = betting == np.max(betting)
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_random_all_horses(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """(Baseline) Returns uniform bettings on all horses"""
    assert (previous_stakes is not None) or (odds is not None)

    n_horses = x_race.shape[0]
    betting = np.ones((n_horses,)) / n_horses
    betting = betting * capital_fraction
    return betting


def race_rickiest_horse(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns bettings with capital_fraction put on rickiest horse according
    to pari-mutual odds"""
    assert (previous_stakes is not None) or (odds is not None)

    if previous_stakes is not None:
        betting = previous_stakes == np.nanmin(previous_stakes)
    else:
        betting = odds == np.nanmax(odds)
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_least_risky_horse(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Returns bettings with capital_fraction put on least ricky horse according
    to pari-mutual odds"""
    assert (previous_stakes is not None) or (odds is not None)

    if previous_stakes is not None:
        betting = previous_stakes == np.nanmax(previous_stakes)
    else:
        betting = odds == np.nanmin(odds)
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_proportional_odds(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Return bettings proportional to pari-mutual odds"""
    assert (previous_stakes is not None) or (odds is not None)

    if previous_stakes is not None:
        betting = (1 / previous_stakes) / np.sum(1 / previous_stakes)
    else:
        betting = odds / np.sum(odds)
    betting = betting * capital_fraction
    return betting


def race_proportional_pari_mutual_proba(
    x_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    """Return bettings proportional to inverse of pari-mutual odds
    (pari-mutual probabilities)"""
    assert (previous_stakes is not None) or (odds is not None)

    if previous_stakes is not None:
        betting = previous_stakes / np.sum(previous_stakes)
    else:
        assert odds is not None
        betting = (1 / odds) / (np.sum(1 / odds))
    betting = betting * capital_fraction
    return betting
