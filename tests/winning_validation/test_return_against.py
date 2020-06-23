import numpy as np
import pytest

from constants import PMU_MINIMUM_BET_SIZE
from winning_validation.return_against import compute_race_return_against
from winning_validation.return_against import get_rectified_pari_mutual_probabilities


@pytest.mark.parametrize(
    "rank_race, predicted_probabilities, base_probabilities, result",
    [
        (
            np.arange(1, 5).reshape((1, 4)),
            np.array([[1, 0, 0, 0]]),
            np.ones((1, 4)) * 1 / 4,
            [3.75],
        ),
        (
            np.arange(1, 5).reshape((1, 4)),
            np.array([[0, 1, 0, 0]]),
            np.ones((1, 4)) * 1 / 4,
            [-1.25],
        ),
        (
            np.stack([np.arange(1, 5), np.arange(1, 5)]),
            np.array([[0, 1, 0, 0], [0, 1, 0, 0]]),
            np.stack([np.ones((4,)) * 1 / 4, np.ones((4,)) * 1 / 4]),
            [-1.25, -1.25],
        ),
    ],
)
def test_compute_race_return_against(
    rank_race, predicted_probabilities, base_probabilities, result
):

    print(
        compute_race_return_against(
            rank_race=rank_race,
            predicted_probabilities=predicted_probabilities,
            base_probabilities=base_probabilities,
        )
    )
    assert np.allclose(
        compute_race_return_against(
            rank_race=rank_race,
            predicted_probabilities=predicted_probabilities,
            base_probabilities=base_probabilities,
        ),
        result,
    )


@pytest.mark.parametrize(
    "previous_stakes, result",
    [
        (PMU_MINIMUM_BET_SIZE * np.ones((1, 4)), 1 / 4 * np.ones((1, 4))),
        (PMU_MINIMUM_BET_SIZE * np.ones((2, 4)), 1 / 4 * np.ones((2, 4))),
    ],
)
def test_get_recitified_pari_mutual_probabilities(previous_stakes, result):
    assert np.allclose(
        get_rectified_pari_mutual_probabilities(previous_stakes=previous_stakes), result
    )
