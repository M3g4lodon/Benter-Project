from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from constants import PMU_BETTINGS
from machine_learning import WinningModel
from utils import import_data

initial_capital = 100

# TODO add more stat (max drawdown, look at master thesis, max number of losses, expexted return, return distribution,
#  standard deviation of returns, EDA on returns to find bias, expected winning proba, average length of loss streak...)
# TODO mininum betting of 150 (1.5€)
# TODO add feedback effet of betting


def race_betting_proportional_positive_return(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: WinningModel,
    capital_fraction: float,
) -> np.array:
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
    winning_model: WinningModel,
    capital_fraction: float,
) -> np.array:
    n_horses = x_race.shape[0]

    model = winning_model.get_n_horses_model(n_horses=n_horses)
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

    expected_return_race = y_hat_race * odds_race * (1 - track_take)
    if expected_return_race.max() <= 1.0:
        return np.zeros((n_horses,))

    betting = expected_return_race == expected_return_race.max()
    assert betting.sum()
    betting = betting / betting.sum()
    betting = betting * capital_fraction
    return betting


def race_bettings_kelly(
    x_race: np.array,
    odds_race: np.array,
    track_take: float,
    winning_model: WinningModel,
    capital_fraction: Optional[float],
) -> np.array:
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
    winning_model: WinningModel,
    capital_fraction: float,
) -> np.array:
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
    winning_model: WinningModel,
    capital_fraction: float,
) -> np.array:
    n_horses = x_race.shape[0]
    model = winning_model.get_n_horses_model(n_horses=n_horses)
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]
    assert np.isclose(y_hat_race.sum(), 1.0)
    betting = y_hat_race == y_hat_race.max()
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_random_one_horse(
    x_race: np.array, odds_race: np.array, track_take: float,winning_model: WinningModel, capital_fraction: float,
) -> np.array:
    n_horses = x_race.shape[0]
    betting = np.random.rand(n_horses)
    betting = betting == np.max(betting)
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_random_all_horses(
    x_race: np.array, odds_race: np.array, track_take: float,winning_model: WinningModel,, capital_fraction: float,
) -> np.array:
    n_horses = x_race.shape[0]
    betting = np.ones((n_horses,)) / n_horses
    betting = betting * capital_fraction
    return betting


def race_rickiest_horse(
    x_race: np.array, odds_race: np.array, track_take: float,winning_model: WinningModel, capital_fraction: float,
) -> np.array:
    betting = odds_race == odds_race.max()
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_least_risky_horse(
    x_race: np.array, odds_race: np.array, track_take: float,winning_model: WinningModel, capital_fraction: float,
) -> np.array:
    betting = odds_race == odds_race.min()
    betting = betting / np.sum(betting)
    betting = betting * capital_fraction
    return betting


def race_proportional_odds(
    x_race: np.array, odds_race: np.array, track_take: float,winning_model: WinningModel, capital_fraction: float,
) -> np.array:
    betting = odds_race / np.sum(odds_race)
    betting = betting * capital_fraction
    return betting


def race_proportional_pari_mutual_proba(
    x_race: np.array, odds_race: np.array, track_take: float,winning_model: WinningModel, capital_fraction: float,
) -> np.array:
    betting = (1 / odds_race) / np.sum((1 / odds_race))
    betting = betting * capital_fraction
    return betting


def compute_expected_return(
    compute_betting: Callable, source: str, code_pari: str, winning_model:WinningModel, show: bool = False
) -> pd.DataFrame:
    """For each races, compute expected return (1 basis)
    Without taking into account the feedback effect"""
    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    records = []
    for x_race, y_race, odds_race in tqdm(
        import_data.get_dataset_races(
            source=source, on_split="val", y_format="rank", remove_nan_odds=True
        ),
        leave=False,
    ):
        betting_race = compute_betting(
            x_race=x_race,
            odds_race=odds_race,
            winning_model=winning_model,
            track_take=track_take,
            capital_fraction=1.0,
        )

        assert 0 <= np.sum(betting_race)
        assert np.sum(betting_race) <= 1 or np.isclose(np.sum(betting_race), 1.0)

        actual_betting = np.round(betting_race, decimals=2)
        expected_return = np.where(
            y_race == 1,
            actual_betting * odds_race * (1 - track_take),
            np.zeros_like(actual_betting),
        ).sum() - np.sum(actual_betting)
        records.append(
            {
                "sum_betting": actual_betting.sum(),
                "expected_return": expected_return,
                "n_horse": x_race.shape[0],
            }
        )

    data = pd.DataFrame.from_records(records)

    return data


def compute_scenario(
    compute_betting_fun: Callable,
    source: str,
    code_pari: str,
    capital_fraction: float,
    winning_model:WinningModel,
    show: bool = False,
) -> pd.DataFrame:
    """Return scenario on validation dataset"""
    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    np.random.seed(42)
    capital_value = initial_capital
    records = []
    for x_race, y_race, odds_race in tqdm(
        import_data.get_dataset_races(
            source=source, on_split="val", y_format="rank", remove_nan_odds=True
        ),
        leave=False,
    ):
        betting_race = compute_betting_fun(
            x_race=x_race,
            odds_race=odds_race,
            track_take=track_take,
            winning_model=winning_model,
            capital_fraction=capital_fraction,
        )
        assert 0 <= np.sum(betting_race)
        assert np.sum(betting_race) <= 1 or np.isclose(np.sum(betting_race), 1.0)

        actual_betting = np.round(betting_race * capital_value, decimals=2)

        capital_value_old = capital_value
        capital_value += np.where(
            y_race == 1,
            actual_betting * odds_race * (1 - track_take),
            np.zeros_like(actual_betting),
        ).sum() - np.sum(actual_betting)

        records.append(
            {
                "Capital Variation": capital_value - capital_value_old,
                "Capital": capital_value,
                "Relative Return": (capital_value - capital_value_old)
                / capital_value_old,
            }
        )

    data = pd.DataFrame.from_records(records)

    data["#_races"] = data.index.to_series()
    if show:
        ax = sns.lineplot(data=data, x="#_races", y="Capital",)
        ax.set(yscale="log")
        plt.show()

        exp_growth_rate = np.log(
            data["Capital"].iloc[-1] / data["Capital"].iloc[0]
        ) / len(data)
        print(
            f"End capital: {data['Capital'].iloc[-1]:.2f}, exponential growth rate: {exp_growth_rate:.2%}"
        )
        sns.distplot(data["Relative Return"])
        plt.show()

        if (data["Capital"] == 0).any():
            zero_capital_index = data[data["Capital"] == 0]["#_races"].iloc[0]
            print(
                f"No more capital after race n°{zero_capital_index + 1} (index {zero_capital_index})"
            )
        else:
            print("Capital is never null!")
    return data


def run():
    compute_scenario(
        compute_betting_fun=race_betting_proportional_positive_return,
        source="PMU",
        code_pari="E_SIMPLE_GAGNANT",
        capital_fraction=0.01,
    )
