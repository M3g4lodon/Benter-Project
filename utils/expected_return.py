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
    y_hat_race: np.array, track_take: float, previous_stakes: np.array, race_bet: np.array
) -> np.array:
    odds_race = get_race_odds(
        track_take=track_take, previous_stakes=previous_stakes, race_bet=race_bet
    )
    return odds_race * y_hat_race - race_bet * (1 - y_hat_race)

