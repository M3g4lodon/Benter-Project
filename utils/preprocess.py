import math
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from joblib import Memory

from constants import CACHE_DIR
from constants import Sources
from utils import import_data

memory = Memory(location=CACHE_DIR, verbose=0)

FEATURE_COLUMNS: Dict[Sources, List[str]] = {
    Sources.PMU: [
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
    ],
    Sources.UNIBET: [
        "n_horse_previous_races",
        "n_horse_previous_positions",
        "average_horse_position",
        "average_horse_top_3",
        "n_jockey_previous_races",
        "n_jockey_previous_positions",
        "average_jockey_position",
        "average_jockey_top_1",
        "average_jockey_top_3",
        "n_trainer_previous_races",
        "n_trainer_previous_positions",
        "average_trainer_position",
        "average_trainer_top_1",
        "average_trainer_top_3",
        "n_owner_previous_races",
        "n_owner_previous_positions",
        "average_owner_position",
        "average_owner_top_1",
        "average_owner_top_3",
        "mean_horse_place",
        "average_horse_top_1_y",
    ],
}
TO_ONE_HOT_ENCODE_COLUMNS: Dict[Sources, List[str]] = {
    Sources.PMU: [
        "allure",
        "course_discipline",
        "course_condition_sexe",
        "blinkers",
        "course_track_type",
        "horse_sex",
    ],
    Sources.UNIBET: [],
}

IF_NAN_IMPUTE_COLUMNS: Dict[Sources, Dict[str, Union[str, float]]] = {
    Sources.PMU: {"handicap_distance": 0, "handicap_weight": 0, "handicap_value": 0},
    Sources.UNIBET: {
        "n_horse_previous_races": 0,
        "n_horse_previous_positions": 0,
        "average_horse_position": "mean",
        "average_horse_top_3": "mean",
        "n_jockey_previous_races": 0,
        "n_jockey_previous_positions": 0,
        "average_jockey_position": "mean",
        "average_jockey_top_1": "mean",
        "average_jockey_top_3": "mean",
        "n_trainer_previous_races": 0,
        "n_trainer_previous_positions": 0,
        "average_trainer_position": "mean",
        "average_trainer_top_1": "mean",
        "average_trainer_top_3": "mean",
        "n_owner_previous_races": 0,
        "n_owner_previous_positions": 0,
        "average_owner_position": "mean",
        "average_owner_top_1": "mean",
        "average_owner_top_3": "mean",
        "mean_horse_place": "mean",
        "average_horse_top_1_y": "mean",
    },
}

NUMERICAL_FEATURES: Dict[Sources, List[str]] = {
    Sources.PMU: [
        "handicap_distance",
        "handicap_weight",
        "duration_since_last_race",
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
    ],
    Sources.UNIBET: [
        "average_horse_position",
        "average_horse_top_3",
        "average_jockey_position",
        "average_jockey_top_1",
        "average_jockey_top_3",
        "average_trainer_position",
        "average_trainer_top_1",
        "average_trainer_top_3",
        "average_owner_position",
        "average_owner_top_1",
        "average_owner_top_3",
        "mean_horse_place",
        "average_horse_top_1_y",
    ],
}

LOG_NUMERICAL_FEATURES: Dict[Sources, List[str]] = {
    Sources.PMU: [
        "course_prize_pool",
        "course_winner_prize",
        "horse_current_year_prize",
        "horse_career_prize",
    ],
    Sources.UNIBET: [
        "n_horse_previous_races",
        "n_horse_previous_positions",
        "n_jockey_previous_races",
        "n_jockey_previous_positions",
        "n_trainer_previous_races",
        "n_trainer_previous_positions",
        "n_owner_previous_races",
        "n_owner_previous_positions",
    ],
}


def _check_features():
    for source in (Sources.UNIBET, Sources.PMU):
        assert set(FEATURE_COLUMNS[source]).issuperset(
            set(TO_ONE_HOT_ENCODE_COLUMNS[source])
        )
        assert set(FEATURE_COLUMNS[source]).issuperset(
            set(IF_NAN_IMPUTE_COLUMNS[source].keys())
        )
        assert not set(IF_NAN_IMPUTE_COLUMNS[source].keys()).intersection(
            set(TO_ONE_HOT_ENCODE_COLUMNS[source])
        )
        for feature, value in IF_NAN_IMPUTE_COLUMNS[source].items():
            if value == "mean":
                assert (
                    feature
                    in NUMERICAL_FEATURES[source] + LOG_NUMERICAL_FEATURES[source]
                )


_check_features()


@memory.cache
def load_preprocess(source: Sources) -> Tuple[dict, dict]:
    train_race_horse_df, _, _ = import_data.get_splitted_featured_data(source=source)
    standard_scaler_parameters = {}
    for numerical_feature in NUMERICAL_FEATURES[source]:
        standard_scaler_parameters[numerical_feature] = {
            "mean": train_race_horse_df[numerical_feature].mean(),
            "std": train_race_horse_df[numerical_feature].std(),
        }
    for numerical_feature in LOG_NUMERICAL_FEATURES[source]:
        standard_scaler_parameters[f"log_{numerical_feature}"] = {
            "mean": np.log(1 + train_race_horse_df[numerical_feature]).mean(),
            "std": np.log(1 + train_race_horse_df[numerical_feature]).std(),
        }
    ohe_features_values = {
        ohe_feature: set(train_race_horse_df[ohe_feature].unique())
        for ohe_feature in TO_ONE_HOT_ENCODE_COLUMNS[source]
    }

    return standard_scaler_parameters, ohe_features_values


@memory.cache
def get_preprocessed_columns(source: Sources) -> List[str]:
    race_horse_df = import_data.load_featured_data(source=source)
    return list(
        preprocess(race_horse_df=race_horse_df.iloc[:1, :], source=source).keys()
    )


@memory.cache
def get_n_preprocessed_feature_columns(source: Sources) -> int:
    _, ohe_features_values = load_preprocess(source=source)
    res = len(FEATURE_COLUMNS[source])
    res = (
        res
        + sum(len(values) for values in ohe_features_values.values())
        - len(TO_ONE_HOT_ENCODE_COLUMNS[source])
    )
    if source == Sources.PMU:
        res = res + 2 - 1  # Unshod

    assert res == len(get_preprocessed_columns(source=source))
    return res


def preprocess(race_horse_df: pd.DataFrame, source: Sources) -> pd.DataFrame:
    features_df = race_horse_df[FEATURE_COLUMNS[source]]
    standard_scaler_parameters, ohe_features_values = load_preprocess(source=source)

    for feature_col, feature_value in IF_NAN_IMPUTE_COLUMNS[source].items():
        if isinstance(feature_value, float) or isinstance(feature_value, int):
            features_df[feature_col] = features_df[feature_col].fillna(feature_value)
            continue
        if feature_value == "mean":
            features_df[feature_col] = features_df[feature_col].fillna(
                standard_scaler_parameters[feature_col]["mean"]
            )
            continue

        raise ValueError(
            f"Unknown imputation value {feature_value} for feature {feature_col}"
        )
    for numerical_feature in NUMERICAL_FEATURES[source]:
        features_df[numerical_feature] = (
            features_df[numerical_feature]
            - standard_scaler_parameters[numerical_feature]["mean"]
        ) / standard_scaler_parameters[numerical_feature]["std"]
        features_df[numerical_feature].fillna(0, inplace=True)

    for numerical_feature in LOG_NUMERICAL_FEATURES[source]:
        features_df[numerical_feature] = (
            np.log(1 + features_df[numerical_feature])
            - standard_scaler_parameters[f"log_{numerical_feature}"]["mean"]
        ) / standard_scaler_parameters[f"log_{numerical_feature}"]["std"]
        features_df[numerical_feature].fillna(0, inplace=True)

    for ohe_feature in TO_ONE_HOT_ENCODE_COLUMNS[source]:
        for value in ohe_features_values[ohe_feature]:
            if isinstance(value, float) and math.isnan(value):
                features_df[f"{ohe_feature}_{value}"] = features_df[ohe_feature].isna()
                continue

            features_df[f"{ohe_feature}_{value}"] = features_df[ohe_feature] == value

        features_df.drop([ohe_feature], inplace=True, axis=1)

    if source == Sources.PMU:
        features_df["unshod_deferre_anterieurs"] = features_df["unshod"].isin(
            ["DEFERRE_ANTERIEURS_POSTERIEURS", "DEFERRE_ANTERIEURS"]
        )
        features_df["unshod_deferre_posterieurs"] = features_df["unshod"].isin(
            ["DEFERRE_ANTERIEURS_POSTERIEURS", "DEFERRE_POSTERIEURS"]
        )
        features_df.drop(["unshod"], inplace=True, axis=1)

    return features_df
