import numpy as np
import pytest

from winning_validation.probability_at_rank import compute_rank_proba


@pytest.mark.parametrize(
    "proba_distribution, rank_race, k, result",
    [
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 2, 3, 4]), 1, 1 / 4),
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 2, 3, 4]), 2, 1 / 4 / 3),
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 2, 3, 4]), 3, 1 / 4 / 3 / 2),
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 2, 3, 4]), 4, 1 / 4 / 3 / 2 / 1),
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 1, 1, 1]), 1, 1 / 4 ** 4),
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 1, 1, 1]), 2, 1 / 4 ** 4),
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 1, 1, 1]), 3, 1 / 4 ** 4),
        (np.array([1, 1, 1, 1]) / 4, np.array([1, 1, 1, 1]), 4, 1 / 4 ** 4),
    ],
)
def test_compute_rank_proba(proba_distribution, rank_race, k, result):

    assert np.isclose(
        compute_rank_proba(
            proba_distribution=proba_distribution, rank_race=rank_race, k=k
        ),
        result,
    )
