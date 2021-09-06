from typing import Optional

import numpy as np


# TODO test it
def get_race_odds(
    track_take: float, previous_stakes: np.array, race_bet: np.array
) -> np.array:
    return (
        race_bet
        / (previous_stakes + race_bet)
        * (1 - track_take)
        * (previous_stakes.sum() + race_bet.sum())
    )


# TODO test it
def get_race_expected_return(
    y_hat_race: np.array,
    track_take: float,
    race_bet: np.array,
    previous_stakes: Optional[np.array] = None,
    odds: Optional[np.array] = None,
) -> np.array:
    assert (previous_stakes is not None) or (odds is not None)
    odds_race = (
        get_race_odds(
            track_take=track_take, previous_stakes=previous_stakes, race_bet=race_bet
        )
        if previous_stakes is not None
        else odds
    )
    assert not np.any(np.isinf(odds_race)), (odds_race, previous_stakes, odds)
    return odds_race * y_hat_race - race_bet * (1 - y_hat_race)
