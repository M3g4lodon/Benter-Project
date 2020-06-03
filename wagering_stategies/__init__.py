from typing import Optional

import numpy as np

from winning_horse_models import AbstractWinningModel


def race_betting_proportional_positive_return(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Returns bettings proportional to computed positive returns"""
    n_horses = x_race.shape[0]

    model = winning_model.get_n_horses_model(n_horses=n_horses)
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

    expected_return_race = y_hat_race * odds_race * (1 - track_take)
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
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Returns bettings putting all capital fraction on the best expected return (if it is positive)"""
    n_horses = x_race.shape[0]

    model = winning_model.get_n_horses_model(n_horses=n_horses)
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

    expected_return_race = y_hat_race * odds_race * (1 - track_take)
    if expected_return_race.max() <= 1.0:
        return np.zeros((n_horses,))
    max_expected_return = expected_return_race.max()

    if max_expected_return <= 0:
        return np.zeros_like(y_hat_race)

    betting = expected_return_race == max_expected_return
    assert betting.sum()
    betting = betting / betting.sum()
    betting = betting * capital_fraction
    return betting


def race_bettings_kelly(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: Optional[float],
) -> np.array:
    """Returns Kelly criterion to compute optimal betting
    (capital_fraction can be omitted here since Kelly criterion already computes the
     optimal capital fraction to risk)"""
    n_horses = x_race.shape[0]

    model = winning_model.get_n_horses_model(n_horses=n_horses)
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

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

    race_f = np.clip(
        (expected_return_race - R_S) / (1 - track_take) / odds_race, a_min=0, a_max=None
    )
    if capital_fraction:
        if race_f.sum():
            race_f = race_f * capital_fraction / np.sum(race_f)
    return race_f


def race_betting_proportional_winning_proba(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Returns bettings proportional to winning proba given by winning_model"""
    n_horses = x_race.shape[0]
    model = winning_model.get_n_horses_model(n_horses=n_horses)
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]
    assert np.isclose(y_hat_race.sum(), 1.0)
    betting = y_hat_race * capital_fraction

    return betting


def race_betting_best_winning_proba(
    x_race: np.array,
    odds_rac: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Returns bettings with all capital_fraction on the best horse according to winning_model"""
    n_horses = x_race.shape[0]
    model = winning_model.get_n_horses_model(n_horses=n_horses)
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]
    assert np.isclose(y_hat_race.sum(), 1.0)
    betting = y_hat_race == y_hat_race.max()
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_random_one_horse(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """(Baseline)Returns uniformly draws of betting"""
    n_horses = x_race.shape[0]
    betting = np.random.rand(n_horses)
    betting = betting == np.max(betting)
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_random_all_horses(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """(Baseline) Returns uniform bettings on all horses"""
    n_horses = x_race.shape[0]
    betting = np.ones((n_horses,)) / n_horses
    betting = betting * capital_fraction
    return betting


def race_rickiest_horse(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Returns bettings with capital_fraction put on rickiest horse according to pari-mutual odds"""
    betting = odds_race == odds_race.max()
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_least_risky_horse(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Returns bettings with capital_fraction put on least ricky horse according to pari-mutual odds"""
    betting = odds_race == odds_race.min()
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_proportional_odds(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Return bettings proportional to pari-mutual odds"""
    betting = odds_race / np.sum(odds_race)
    betting = betting * capital_fraction
    return betting


def race_proportional_pari_mutual_proba(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: AbstractWinningModel,
    capital_fraction: float,
) -> np.array:
    """Return bettings proportional to inverse of pari-mutual odds (pari-mutual probabilities)"""
    betting = (1 / odds_race) / np.sum((1 / odds_race))
    betting = betting * capital_fraction
    return betting