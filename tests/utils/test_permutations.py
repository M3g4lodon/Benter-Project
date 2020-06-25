import pytest

from utils.permutations import get_n_cut_permutations
from utils.permutations import get_n_permutations


@pytest.mark.parametrize(
    "sequence, n, result",
    [
        (range(3), 10, [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]),
        (range(3), 3, [(2, 0, 1), (2, 1, 0), (0, 2, 1)]),
    ],
)
def test_get_n_permutations(sequence, n, result):
    assert get_n_permutations(sequence=sequence, n=n) == result


@pytest.mark.parametrize(
    "sequence, n, r, result",
    [
        (range(3), 10, 2, [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]),
        (range(3), 3, 2, [(2, 0), (0, 1), (0, 2)]),
    ],
)
def test_get_n_cut_permutations(sequence, n, r, result):
    assert get_n_cut_permutations(sequence=sequence, n=n, r=r) == result
