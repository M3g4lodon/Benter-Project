from typing import Tuple

import pandas as pd
from joblib import Memory

from constants import CACHE_DIR
from utils.import_data import get_splitted_featured_data

memory = Memory(location=CACHE_DIR, verbose=0)

FEATURE_COLUMNS = [
    "is_pregnant",
    "handicap_distance",
    "handicap_weight",
    "allure",
    "course_discipline",
    "duration_since_last_race",
    "course_condition_sexe",
    "course_prize_pool",
    "course_winner_prize",
    "blinkers",
    "horse_current_year_prize",
    "horse_career_prize",
    "unshod",
    "handicap_value",
    "course_track_type",
    "indicateurInedit",
    "placeCorde",
    "horse_age",
    "horse_sex",
    "n_run_races",
    "n_won_races",
    "n_placed_races",
    "win_ratio",
    "mean_place",
    "jockey_win_rate",
    "jockey_mean_place",
    "trainer_win_rate",
    "trainer_mean_place",
]
TO_ONE_HOT_ENCODE_COLUMNS = [
    "allure",
    "course_discipline",
    "course_condition_sexe",
    "blinkers",
    "course_track_type",
    "horse_sex",
]
IF_NAN_PUT_ZERO_COLUMNS = ["handicap_distance", "handicap_weight", "handicap_value"]
assert set(TO_ONE_HOT_ENCODE_COLUMNS).intersection(set(FEATURE_COLUMNS)) == set(
    TO_ONE_HOT_ENCODE_COLUMNS
)
assert set(IF_NAN_PUT_ZERO_COLUMNS).intersection(set(FEATURE_COLUMNS)) == set(
    IF_NAN_PUT_ZERO_COLUMNS
)
assert not set(IF_NAN_PUT_ZERO_COLUMNS).intersection(set(TO_ONE_HOT_ENCODE_COLUMNS))


NUMERICAL_FEATURES = [
    "handicap_distance",
    "handicap_weight",
    "duration_since_last_race",
    "course_prize_pool",
    "course_winner_prize",
    "horse_current_year_prize",
    "horse_career_prize",
    "handicap_value",
    "placeCorde",
    "horse_age",
    "n_run_races",
    "n_won_races",
    "n_placed_races",
    "win_ratio",
    "mean_place",
    "jockey_win_rate",
    "jockey_mean_place",
    "trainer_win_rate",
    "trainer_mean_place",
]


@memory.cache
def load_preprocess(source: str) -> Tuple[dict, dict]:
    train_race_horse_df, _, _ = get_splitted_featured_data(source=source)
    standard_scaler_parameters = {}
    for numerical_feature in NUMERICAL_FEATURES:
        standard_scaler_parameters[numerical_feature] = {
            "mean": train_race_horse_df[numerical_feature].mean(),
            "std": train_race_horse_df[numerical_feature].std(),
        }

    ohe_features_values = {
        ohe_feature: set(train_race_horse_df[ohe_feature].unique())
        for ohe_feature in TO_ONE_HOT_ENCODE_COLUMNS
    }

    return standard_scaler_parameters, ohe_features_values


@memory.cache
def get_n_preprocessed_feature_columns(source: str) -> int:
    _, ohe_features_values = load_preprocess(source=source)
    res = len(FEATURE_COLUMNS)
    res = (
        res
        + sum(len(values) for values in ohe_features_values.values())
        - len(TO_ONE_HOT_ENCODE_COLUMNS)
    )
    res = res + 2 - 1  # Unshod
    return res


def preprocess(race_horse_df: pd.DataFrame, source: str) -> pd.DataFrame:
    features_df = race_horse_df[FEATURE_COLUMNS]
    standard_scaler_parameters, ohe_features_values = load_preprocess(source=source)

    features_df[IF_NAN_PUT_ZERO_COLUMNS] = features_df[IF_NAN_PUT_ZERO_COLUMNS].fillna(
        0
    )
    for numerical_feature in NUMERICAL_FEATURES:
        features_df[numerical_feature] = (
            features_df[numerical_feature]
            - standard_scaler_parameters[numerical_feature]["mean"]
        ) / standard_scaler_parameters[numerical_feature]["std"]
        features_df[numerical_feature].fillna(0, inplace=True)

    for ohe_feature in TO_ONE_HOT_ENCODE_COLUMNS:
        for value in ohe_features_values[ohe_feature]:
            features_df[f"{ohe_feature}_{value}"] = features_df[ohe_feature] == value

        features_df.drop([ohe_feature], inplace=True, axis=1)

    features_df["unshod_deferre_anterieurs"] = features_df["unshod"].isin(
        ["DEFERRE_ANTERIEURS_POSTERIEURS", "DEFERRE_ANTERIEURS"]
    )
    features_df["unshod_deferre_posterieurs"] = features_df["unshod"].isin(
        ["DEFERRE_ANTERIEURS_POSTERIEURS", "DEFERRE_POSTERIEURS"]
    )
    features_df.drop(["unshod"], inplace=True, axis=1)

    return features_df
