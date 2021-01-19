import itertools
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from constants import PMU_BETTINGS
from constants import PMU_MINIMUM_BET_SIZE
from utils import import_data
from utils.expected_return import get_race_odds
from wagering_stategies import race_betting_proportional_positive_return
from winning_horse_models import AbstractWinningModel

START_CAPITAL = 100 * 100  # 100.00€
DEFAULT_BET_SIZE = PMU_MINIMUM_BET_SIZE * 10

# TODO add more stat (max drawdown, look at master thesis, max number of losses,
#  expexted return, return distribution,
#  standard deviation of returns, EDA on returns to find bias,
#  expected winning proba, average length of loss streak...)


def compute_expected_return(
    compute_betting_fun: Callable,
    source: str,
    code_pari: str,
    winning_model: AbstractWinningModel,
    bet_size: float = DEFAULT_BET_SIZE,
) -> pd.DataFrame:
    """For each races, compute expected return (1 basis)
    Without taking into account the feedback effect"""
    assert bet_size > PMU_MINIMUM_BET_SIZE

    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    records = []
    n_races = import_data.get_n_races(
        source=source, on_split="val", remove_nan_previous_stakes=True
    )
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
        # TODO get races per n_horses and sort the result dataframe
        betting_race = compute_betting_fun(
            x_race=x_race,
            previous_stakes=race_df["totalEnjeu"],
            winning_model=winning_model,
            track_take=track_take,
            capital_fraction=1.0,
        )

        assert np.sum(betting_race) >= 0 or np.isclose(np.sum(betting_race), 0.0)
        assert np.sum(betting_race) <= 1 or np.isclose(np.sum(betting_race), 1.0)

        actual_betting = betting_race * bet_size
        actual_betting = np.where(
            actual_betting < PMU_MINIMUM_BET_SIZE,
            np.zeros_like(actual_betting),
            np.round(actual_betting),
        )

        expected_return = (
            np.where(
                y_race == 1,
                get_race_odds(
                    track_take=track_take,
                    previous_stakes=race_df["totalEnjeu"],
                    race_bet=actual_betting,
                ),
                np.zeros_like(actual_betting),
            ).sum()
            - np.sum(actual_betting)
        )
        relative_expected_return = expected_return / np.sum(actual_betting)
        records.append(
            {
                "betting_fraction": betting_race.sum(),
                "bet_size": actual_betting.sum(),
                "expected_return": expected_return,
                "relative_expected_return": relative_expected_return,
                "computed_bets_on_n_horses": np.sum(betting_race > 0.0),
                "actual_bets_on_n_horses": np.sum(actual_betting > 0.0),
                "n_horse": x_race.shape[0],
                "date": race_df["date"].iloc[0],
                "race_id": race_df["race_id"].iloc[0],
            }
        )

    return pd.DataFrame.from_records(records)


def plot_expected_return(expected_return_df: pd.DataFrame) -> None:
    # TODO streak per day/month
    # TODO look at computed_bets_on_n_horses, actual_bets_on_n_horses
    print(
        f"On all races, your expected return is "
        f"{expected_return_df.relative_expected_return.mean():+.2%} "
        f"(std: {expected_return_df.relative_expected_return.std():.1f})"
    )

    print(
        f"You bet {(expected_return_df.bet_size > 0).mean():.2%} of the time "
        f"({(expected_return_df.bet_size > 0).sum():.0f} out of"
        f" {len(expected_return_df):.0f} races)"
    )

    not_betting_streaks = [
        sum(1 for _ in grouper_)
        for is_betting, grouper_ in itertools.groupby(expected_return_df.bet_size > 0)
        if not is_betting
    ]
    if not_betting_streaks:
        print(f"Average not betting streaks: {np.mean(not_betting_streaks):.2f} races")
    else:
        print("No races without bettings!")
    betting_expected_return = expected_return_df[expected_return_df.bet_size > 0]
    print(
        f"When you bet, on average your expected return is "
        f"{betting_expected_return.relative_expected_return.mean():+.2%} "
        f"(std: {betting_expected_return.relative_expected_return.std():.1f})"
    )

    print(
        f"When you bet, you win "
        f"{(betting_expected_return.relative_expected_return > 0).mean():.2%} "
        f"of the time "
        f"({(betting_expected_return.relative_expected_return > 0).sum():.0f} out of "
        f"{len(betting_expected_return):.0f} bets)"
    )
    print(
        f"When you bet, you lose "
        f"{(betting_expected_return.relative_expected_return < 0).mean():.2%} "
        f"of the time"
        f"({(betting_expected_return.relative_expected_return < 0).sum():.0f} out of "
        f"{len(betting_expected_return):.0f} bets)"
    )

    winning_expected_return = expected_return_df[
        (expected_return_df.relative_expected_return > 0)
    ]
    if not winning_expected_return.empty:
        print(
            "When you bet&win, you make "
            f"{winning_expected_return.relative_expected_return.mean():+.2%} ("
            f"std: {winning_expected_return.relative_expected_return.std():.1f})"
        )
    else:
        print("Not winning any bets!")

    win_streaks = [
        (is_winning, sum(1 for _ in grouper_))
        for is_winning, grouper_ in itertools.groupby(
            betting_expected_return.expected_return > 0
        )
    ]

    winning_streaks = [
        streak_length for is_winning, streak_length in win_streaks if is_winning
    ]
    losing_streaks = [
        streak_length for is_winning, streak_length in win_streaks if not is_winning
    ]
    if losing_streaks:
        print(
            f"While betting, Average losing streaks {np.mean(losing_streaks)} "
            f"(std: {np.std(losing_streaks):.1f})"
        )
        print(f"While betting, Longest losing streak: {np.max(losing_streaks):.0f}")
    else:
        print("There is no losing streaks!")

    if winning_streaks:
        print(f"While betting, Longest winning streak: {np.max(winning_streaks):.0f}")
    else:
        print("There is no winning streaks!")


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
    capital_value = START_CAPITAL
    records = []
    n_races = import_data.get_n_races(
        source=source, on_split="val", remove_nan_previous_stakes=True
    )
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
        assert np.sum(betting_race) >= 0 or np.isclose(np.sum(betting_race), 0.0), (
            f"{betting_race} sum {np.sum(betting_race)} can not be below 0.0, "
            f"on race id {race_df['id'].iloc[0]}"
        )
        assert np.sum(betting_race) <= 1 or np.isclose(np.sum(betting_race), 1.0)

        actual_betting = np.round(betting_race * capital_value)
        actual_betting = np.where(
            actual_betting < PMU_MINIMUM_BET_SIZE,
            np.zeros_like(actual_betting),
            np.round(actual_betting),
        )

        capital_value_old = capital_value
        capital_value += (
            np.where(
                y_race == 1,
                get_race_odds(
                    track_take=track_take,
                    previous_stakes=race_df["totalEnjeu"],
                    race_bet=actual_betting,
                ),
                np.zeros_like(actual_betting),
            ).sum()
            - np.sum(actual_betting)
        )

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


def plot_scenario(scenario_df: pd.DataFrame) -> None:
    scenario_df["#_races"] = scenario_df.index.to_series()
    ax = sns.lineplot(data=scenario_df, x="#_races", y="Capital")
    ax.set(yscale="log")
    plt.show()

    exp_growth_rate = np.log(
        scenario_df["Capital"].iloc[-1] / scenario_df["Capital"].iloc[0]
    ) / len(scenario_df)
    print(
        f"End capital: {scenario_df['Capital'].iloc[-1]:.2f}, "
        f"exponential growth rate: {exp_growth_rate:.2%}"
    )
    sns.distplot(scenario_df["Relative Return"])
    plt.show()

    if (scenario_df["Capital"] == 0).any():
        zero_capital_index = scenario_df[scenario_df["Capital"] == 0]["#_races"].iloc[0]
        print(
            f"No more capital after race n°{zero_capital_index + 1} "
            f"(index {zero_capital_index})"
        )
    else:
        print("Capital is never null!")


def run():
    # pylint:disable=import-outside-toplevel
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
