import itertools
import math
import random
from typing import Set


def random_permutation_generator(sequence):
    seen = {tuple(sequence)}
    random.seed(42)
    while True:
        permutation = tuple(random.sample(sequence, len(sequence)))
        if permutation not in seen:
            seen.add(permutation)
            yield permutation


def get_n_permutations(sequence, n: int):
    if n >= math.factorial(len(sequence)) - 1:
        return [
            perm for perm in itertools.permutations(sequence) if perm != tuple(sequence)
        ]
    rand_perms = random_permutation_generator(sequence)
    return [next(rand_perms) for _ in range(n)]


def random_cut_permutation_generator(sequence, r: int):
    assert 0 < r < len(sequence)
    seen: Set[tuple] = set()
    random.seed(42)
    while True:
        permutation = tuple(random.sample(sequence, r))
        if permutation not in seen:
            seen.add(permutation)
            yield permutation


def get_n_cut_permutations(sequence, n: int, r: int):
    assert 0 < r < len(sequence)

    if n >= math.factorial(len(sequence)) / math.factorial(len(sequence) - r):
        return [perm for perm in itertools.permutations(sequence, r=r)]
    rand_perms = random_cut_permutation_generator(sequence, r=r)
    return [next(rand_perms) for _ in range(n)]
