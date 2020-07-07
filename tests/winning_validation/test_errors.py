import numpy as np
import pytest

from winning_validation.errors import exact_top_k
from winning_validation.errors import kappa_cohen_like
from winning_validation.errors import precision_at_k
from winning_validation.errors import same_top_k


@pytest.mark.parametrize(
    "validation_method, rank_race, rank_hat, k, result",
    [
        (exact_top_k, np.array([3, 1, 2, 5, 5]), np.array([2, 1, 3, 5, 5]), 1, 1.0),
        (same_top_k, np.array([3, 1, 2, 5, 5]), np.array([2, 1, 3, 5, 5]), 1, 1.0),
        (precision_at_k, np.array([3, 1, 2, 5, 5]), np.array([2, 1, 3, 5, 5]), 1, 1.0),
        (
            kappa_cohen_like,
            np.array([3, 1, 2, 5, 5]),
            np.array([2, 1, 3, 5, 5]),
            1,
            1.0,
        ),
        (exact_top_k, np.array([3, 1, 2, 5, 5]), np.array([2, 1, 3, 5, 5]), 2, 0.0),
        (same_top_k, np.array([3, 1, 2, 5, 5]), np.array([2, 1, 3, 5, 5]), 2, 0.0),
        (precision_at_k, np.array([3, 1, 2, 5, 5]), np.array([2, 1, 3, 5, 5]), 2, 0.5),
        (
            exact_top_k,
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3, 4, 5, 7, 6]),
            5,
            1.0,
        ),
        (
            same_top_k,
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3, 4, 5, 7, 6]),
            5,
            1.0,
        ),
        (
            precision_at_k,
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3, 4, 5, 7, 6]),
            5,
            1.0,
        ),
        (
            exact_top_k,
            np.array([1, 2, 3, 4, 5, 6, 6]),
            np.array([1, 2, 3, 4, 5, 7, 6]),
            5,
            1.0,
        ),
        (
            same_top_k,
            np.array([1, 2, 3, 4, 5, 6, 6]),
            np.array([1, 2, 3, 4, 5, 7, 6]),
            5,
            1.0,
        ),
        (
            precision_at_k,
            np.array([1, 2, 3, 4, 5, 6, 6]),
            np.array([1, 2, 3, 4, 5, 7, 6]),
            5,
            1.0,
        ),
        (
            exact_top_k,
            np.array([4, 1, 3, 2, 5, 5]),
            np.array([1, 2, 3, 4, 5, 5]),
            5,
            0.0,
        ),
        (
            same_top_k,
            np.array([4, 1, 3, 2, 5, 5]),
            np.array([1, 2, 3, 4, 5, 5]),
            5,
            1.0,
        ),
        (
            precision_at_k,
            np.array([4, 1, 3, 2, 5, 5]),
            np.array([1, 2, 3, 4, 5, 5]),
            5,
            1.0,
        ),
        (
            precision_at_k,
            np.array([4, 1, 3, 2, 5, 5]),
            np.array([1, 2, 3, 4, 5, 5]),
            1,
            0.0,
        ),
    ],
)
def test_winning_validation_functions(
    validation_method, rank_race, rank_hat, k, result
):
    assert validation_method(rank_race=rank_race, rank_hat=rank_hat, k=k) == result
