from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from constants import PMU_BETTINGS, PMU_MINIMUM_BET_SIZE
from utils.expected_return import get_race_odds
from wagering_stategies import race_betting_proportional_positive_return
from winning_horse_models import AbstractWinningModel
from utils import import_data

initial_capital = 100 * 100  # 100.00€

# TODO add more stat (max drawdown, look at master thesis, max number of losses, expexted return, return distribution,
#  standard deviation of returns, EDA on returns to find bias, expected winning proba, average length of loss streak...)
# TODO mininum betting of 150 (1.5€)
# TODO add feedback effect of betting


def compute_expected_return(
    compute_betting: Callable,
    source: str,
    code_pari: str,
    winning_model: AbstractWinningModel,
) -> pd.DataFrame:
    """For each races, compute expected return (1 basis)
    Without taking into account the feedback effect"""
    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    records = []
    n_races = import_data.get_n_races(source=source, on_split="val", remove_nan_previous_stakes=True)
    for x_race, y_race, race_df in tqdm(
        import_data.get_dataset_races(
            source=source,
            on_split="val",
            y_format="rank",
            x_format="sequential_per_horse",
            remove_nan_previous_stakes=True,
        ),
        leave=False,
        total=n_races,
    ):
        betting_race = compute_betting(
            x_race=x_race,
            previous_stakes=race_df["totalEnjeu"],
            winning_model=winning_model,
            track_take=track_take,
            capital_fraction=1.0,

        )

        assert 0 <= np.sum(betting_race) or np.isclose(np.sum(betting_race), 0.0)
        assert np.sum(betting_race) <= 1 or np.isclose(np.sum(betting_race), 1.0)

        actual_betting = np.round(betting_race, decimals=2)
        expected_return = np.where(
            y_race == 1,
            get_race_odds(
                track_take=track_take,
                previous_stakes=race_df["totalEnjeu"],
                race_bet=actual_betting,
            ),
            np.zeros_like(actual_betting),
        ).sum() - np.sum(actual_betting)
        records.append(
            {
                "sum_betting": actual_betting.sum(),
                "expected_return": expected_return,
                "n_horse": x_race.shape[0],
                "date": race_df["date"].iloc[0],
                "race_id": race_df["race_id"].iloc[0],
            }
        )

    return pd.DataFrame.from_records(records)


def compute_scenario(
    compute_betting_fun: Callable,
    source: str,
    code_pari: str,
    capital_fraction: float,
    winning_model: AbstractWinningModel,
) -> pd.DataFrame:
    """Return scenario on validation dataset"""
    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    np.random.seed(42)
    capital_value = initial_capital
    records = []
    n_races = import_data.get_n_races(source=source, on_split="val", remove_nan_previous_stakes=True)
    for x_race, y_race, race_df in tqdm(
        import_data.get_dataset_races(
            source=source,
            on_split="val",
            y_format="rank",
            x_format="sequential_per_horse",
            remove_nan_previous_stakes=True,
        ),
        leave=False,
        total=n_races,
    ):
        betting_race = compute_betting_fun(
            x_race=x_race,
            previous_stakes=race_df["totalEnjeu"],
            track_take=track_take,
            winning_model=winning_model,
            capital_fraction=capital_fraction,
        )
        assert 0 <= np.sum(betting_race) or np.isclose(np.sum(betting_race), 0.0), (
            f"{betting_race} sum {np.sum(betting_race)} can not be below 0.0, "
            f"on race id {race_df['id'].iloc[0]}"
        )
        assert np.sum(betting_race) <= 1 or np.isclose(np.sum(betting_race), 1.0)

        actual_betting = np.round(betting_race * capital_value, decimals=2)

        capital_value_old = capital_value
        capital_value += np.where(
            y_race == 1,
            get_race_odds(
                track_take=track_take,
                previous_stakes=race_df["totalEnjeu"],
                race_bet=actual_betting,
            ),
            np.zeros_like(actual_betting),
        ).sum() - np.sum(actual_betting)

        records.append(
            {
                "Capital Variation": capital_value - capital_value_old,
                "Capital": capital_value,
                "Relative Return": (capital_value - capital_value_old)
                / capital_value_old,
                "n_horse": x_race.shape[0],
                "date": race_df["date"].iloc[0],
                "race_id": race_df["race_id"].iloc[0],
            }
        )

    return pd.DataFrame.from_records(records)


def plot_scenario(scenario_df: pd.DataFrame):
    ax = sns.lineplot(data=scenario_df, x="#_races", y="Capital",)
    ax.set(yscale="log")
    plt.show()

    exp_growth_rate = np.log(
        scenario_df["Capital"].iloc[-1] / scenario_df["Capital"].iloc[0]
    ) / len(scenario_df)
    print(
        f"End capital: {scenario_df['Capital'].iloc[-1]:.2f}, exponential growth rate: {exp_growth_rate:.2%}"
    )
    sns.distplot(scenario_df["Relative Return"])
    plt.show()

    if (scenario_df["Capital"] == 0).any():
        zero_capital_index = scenario_df[scenario_df["Capital"] == 0]["#_races"].iloc[0]
        print(
            f"No more capital after race n°{zero_capital_index + 1} (index {zero_capital_index})"
        )
    else:
        print("Capital is never null!")


def run():
    from winning_horse_models.baselines import RandomModel

    compute_scenario(
        compute_betting_fun=race_betting_proportional_positive_return,
        source="PMU",
        code_pari="E_SIMPLE_GAGNANT",
        capital_fraction=0.01,
        winning_model=RandomModel(),
    )


if __name__ == "__main__":
    run()
