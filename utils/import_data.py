import datetime as dt
import functools
import os
from itertools import combinations
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import pytz
from joblib import Memory

from constants import CACHE_DIR
from constants import DATA_DIR

memory = Memory(location=CACHE_DIR, verbose=0)

ZETURF_BEFORE_TRAIN_DATE = dt.datetime(2012, 11, 1)
ZETURF_BEFORE_VALIDATION_DATE = dt.datetime(2012, 12, 1)

PMU_BEFORE_TRAIN_DATE = dt.datetime(2018, 1, 1, tzinfo=pytz.UTC)
PMU_BEFORE_VALIDATION_DATE = dt.datetime(2019, 7, 1, tzinfo=pytz.UTC)

MAX_NAN_PROP = 0.5


@functools.lru_cache(maxsize=None)
def load_featured_data(source: str) -> pd.DataFrame:
    """Load stored CSV of horse/races for the given source"""
    assert source in ["ZETURF", "PMU"]

    if source == "ZETURF":
        race_horse_df = pd.read_csv(
            os.path.join(DATA_DIR, "2012_data_with_features.csv")
        )

    else:
        race_horse_df = pd.read_csv(
            os.path.join(DATA_DIR, "pmu_data_with_features.csv")
        )

    race_horse_df["race_datetime"] = pd.to_datetime(race_horse_df["race_datetime"])
    race_horse_df["duration_since_last_race"] = pd.to_timedelta(
        race_horse_df["duration_since_last_race"]
    )

    return race_horse_df


def get_splitted_featured_data(
    source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/Validation/Test Split the Horse/Races of a given source"""
    race_horse_df = load_featured_data(source=source)
    if source == "ZETURF":
        before_train_date, before_validation_date = (
            ZETURF_BEFORE_TRAIN_DATE,
            ZETURF_BEFORE_VALIDATION_DATE,
        )

    else:

        before_train_date, before_validation_date = (
            PMU_BEFORE_TRAIN_DATE,
            PMU_BEFORE_VALIDATION_DATE,
        )

    train_race_horse_df = race_horse_df[race_horse_df.race_datetime < before_train_date]
    val_race_horse_df = race_horse_df[
        (race_horse_df.race_datetime >= before_train_date)
        & (race_horse_df.race_datetime < before_validation_date)
    ]
    test_race_horse_df = race_horse_df[
        race_horse_df.race_datetime >= before_validation_date
    ]
    assert len(race_horse_df) == len(train_race_horse_df) + len(
        val_race_horse_df
    ) + len(test_race_horse_df)

    return train_race_horse_df, val_race_horse_df, test_race_horse_df


def get_split_date(source: str, on_split: str) -> pd.DataFrame:
    (
        train_race_horse_df,
        val_race_horse_df,
        test_race_horse_df,
    ) = get_splitted_featured_data(source)
    assert on_split in {"train", "val", "test"}

    if on_split == "train":
        return train_race_horse_df
    if on_split == "val":
        return val_race_horse_df
    # on_split == "test"
    return test_race_horse_df


def extract_x_y(  # pylint:disable=too-many-branches
    race_df: pd.DataFrame,
    x_format: str,
    y_format: str,
    source: str,
    ignore_y: bool = False,
) -> Tuple[Optional[np.array], Optional[np.array]]:
    """For a given race in `race_df` returns features, y in the asked format and odds"""
    assert x_format in {"sequential_per_horse", "flattened"}
    assert y_format in {"first_position", "rank", "index_first"}
    from utils import preprocess  # pylint:disable=cyclic-import,import-outside-toplevel

    x_race = preprocess.preprocess(race_horse_df=race_df, source=source)

    if pd.isnull(x_race).mean().mean() > MAX_NAN_PROP:
        return None, None

    # If there is at least one horse features full of nan
    if pd.isnull(x_race).all().any():
        return None, None

    x_race = x_race.values

    if x_format == "flattened":
        x_race = x_race.flatten(order="F")

    if race_df["horse_place"].isna().all() and not ignore_y:
        return None, None
    if ignore_y:
        y_race = np.empty(shape=(x_race.shape[0]))
        y_race[:] = np.NaN

    else:
        if y_format == "first_position":
            y_race = (race_df["horse_place"] == 1).values
            if y_race.sum() == 0:
                return None, None
            assert y_race.sum() > 0
            y_race = y_race / y_race.sum()  # Beware that ExAequo is possible
        elif y_format == "rank":
            y_race = race_df["horse_place"].values
            y_race[np.isnan(y_race)] = y_race.shape[0]
        else:  # y_format == 'index_first'
            # TODO take into account exaequo
            # in case of exAequo, only the first horse in index is returned as first
            candidates = (race_df.horse_place == 1).values
            if np.sum(candidates) == 0:
                # no winner in this race
                return None, None
            y_race = np.argwhere(candidates)[0][0]
        assert not np.any(np.isnan(y_race))

    return x_race, y_race


@memory.cache
def _get_races_per_horse_number(
    source: str,
    n_horses: int,
    on_split: str,
    y_format: str,
    remove_nan_odds: bool = False,
) -> Tuple[np.array, np.array, List[pd.DataFrame]]:
    """For the given source, the given split, the given number of horses per races,
    the given y_format,
    returns numpy arrays of features, y, and race_dfs (is cached)"""

    rh_df = get_split_date(source=source, on_split=on_split)

    x = []
    y = []
    race_dfs = []
    for _, race_df in rh_df[rh_df["n_horses"] == n_horses].groupby("race_id"):
        if source == "PMU":
            race_df = race_df[race_df["statut"] == "PARTANT"]
        assert len(race_df) == n_horses

        x_race, y_race = extract_x_y(
            race_df=race_df,
            x_format="sequential_per_horse",
            y_format=y_format,
            source=source,
        )
        if any([x_race is None, y_race is None]):
            continue

        if remove_nan_odds and np.any(np.isnan(race_df["odds"].values)):
            continue

        x.append(x_race)
        y.append(y_race)
        race_dfs.append(race_df)

    return (
        np.asarray(x).astype(np.float32),
        np.asarray(y).astype(np.float32),
        race_dfs,
    )


def get_races_per_horse_number(
    source: str,
    n_horses: int,
    on_split: str,
    x_format: str,
    y_format: str,
    remove_nan_odds: bool = False,
    extra_features_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Tuple[np.array, np.array, List[pd.DataFrame]]:
    """For the given source, the given split, the given number of horses per races,
    the given y_format,
    returns numpy arrays of features, y, and race_dfs

    We expect `extra_features_func` to return a DataFrame in the same format as race_df
    (column names are extra feature name, with the length of n_horses)"""

    assert x_format in {"sequential_per_horse", "flattened"}

    x, y, race_dfs = _get_races_per_horse_number(
        source=source,
        n_horses=n_horses,
        on_split=on_split,
        y_format=y_format,
        remove_nan_odds=remove_nan_odds,
    )
    if x_format == "flattened":
        x = np.reshape(a=x, newshape=(x.shape[0], x.shape[1] * x.shape[2]), order="F")

    if extra_features_func is not None and x.size != 0:
        extra_features = np.stack(
            [extra_features_func(race_df).values for race_df in race_dfs]
        )
        extra_features = np.asarray(extra_features).astype(np.float32)
        x = np.append(x, extra_features, axis=2)
    return x, y, race_dfs


@memory.cache
def get_dataset_races(
    source: str,
    on_split: str,
    x_format: str,
    y_format: str,
    remove_nan_previous_stakes: bool = False,
) -> List[Tuple[np.array, np.array, pd.DataFrame]]:
    """For the given data source, the given split, the given y_format,
    returns list of every races (features, y, race dataframe)"""

    rh_df = get_split_date(source=source, on_split=on_split)

    res = []
    for _, race_df in rh_df.groupby("race_id"):
        x_race, y_race = extract_x_y(
            race_df=race_df, source=source, x_format=x_format, y_format=y_format
        )
        if any([x_race is None, y_race is None]):
            continue
        if np.any(np.isnan(race_df["totalEnjeu"])) and remove_nan_previous_stakes:
            continue
        res.append(
            (
                np.asarray(x_race).astype(np.float32),
                np.asarray(y_race).astype(np.float32),
                race_df,
            )
        )
    return res


@memory.cache
def get_min_max_horse(source: str) -> Tuple[int, int]:
    race_horse_df = load_featured_data(source=source)
    count_per_n_horses = race_horse_df.groupby("race_id")["horse_id"].count()
    max_n_horses = count_per_n_horses.max()
    min_n_horses = max(2, count_per_n_horses.min())
    return min_n_horses, max_n_horses


@memory.cache
def get_n_races(
    source: str, on_split: str, remove_nan_previous_stakes: bool = False
) -> int:
    rh_df = get_split_date(source=source, on_split=on_split)
    if remove_nan_previous_stakes:
        return (
            rh_df.groupby("race_id")["totalEnjeu"]
            .agg(lambda s: np.logical_not(np.any(np.isnan(s))))
            .sum()
        )
    return rh_df["race_id"].nunique()


def run():
    for source in ["PMU", "ZETURF"]:
        print(f"Looking at data from source {source}")
        race_horse_df = load_featured_data(source=source)

        n_races = race_horse_df["race_id"].nunique()
        print(f"Number of races: {n_races}")

        min_n_horses, max_n_horses = get_min_max_horse(source=source)
        print(f"Races go from {min_n_horses} horses to {max_n_horses} horses!")

        horse_to_number_combinaison = {}
        for n_horse in range(2, max_n_horses + 1):
            positions = list(range(1, n_horse + 1))
            combins = []
            for race_size in range(2, n_horse + 1):
                combins.extend(list(combinations(positions, race_size)))
            horse_to_number_combinaison[n_horse] = len(combins)
        count_per_n_horses = race_horse_df.groupby("race_id")["n_horses"].first()
        print(
            f"Number of combinaisons "
            f"{count_per_n_horses.map(horse_to_number_combinaison).sum():.2e}"
        )

        print("Counting number of races per dataset per number of horses")
        for on_split in ["train", "val", "test"]:
            for n_horses in range(2, max_n_horses + 1):
                print(
                    on_split,
                    n_horses,
                    len(
                        get_races_per_horse_number(
                            source=source,
                            n_horses=n_horses,
                            on_split=on_split,
                            x_format="sequential_per_horse",
                            y_format="rank",
                        )[0]
                    ),
                )


if __name__ == "__main__":
    run()
