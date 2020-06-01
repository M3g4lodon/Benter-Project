import numpy as np
import pytest
from utils.winning_validation import (
    exact_top_k,
    same_top_k,
    precision_at_k,
    compute_rank_proba,
    kappa_cohen_like,
)


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
            proba_distribution=proba_distribution, rank_race=rank_race, k=k,
        ),
        result,
    )
