import os

import numpy as np
import pandas as pd

from constants import DATA_DIR
from utils import features


def run() -> None:
    race_horse_df = pd.read_csv(
        os.path.join(DATA_DIR, "export_zeturf_2012.csv"), delimiter=";"
    )

    race_horse_df = race_horse_df[
        ~race_horse_df.duplicated(
            subset=[
                "scraped",
                "race_id",
                "race_name",
                "race_date_dmy",
                "race_date_mdy",
                "course",
                "r_number",
                "c_number",
                "race_type",
                "distance",
                "prize",
                "race_time",
                "horse_id",
                "horse_place",
                "horse_number",
                "horse_name",
                "horse_sex",
                "horse_age",
                "blinkers",
                "jockey",
                "driver",
                "weight",
                "trainer",
                "lane",
                "horse_time",
                "horse_time_km",
                "odds",
                "morning_win",
                "live_win",
                "show_from",
                "show_to",
                "ze_4th",
                "musique",
                "our_choice",
                "horse_distance",
                "win_single",
                "show_single",
                "straight_forecast",
                "trifecta",
                "trifecta_box",
                "ze_4th_result",
            ]
        )
    ]

    race_horse_df = race_horse_df[race_horse_df.horse_place.notna()]
    race_horse_df["race_datetime"] = pd.to_datetime(
        race_horse_df.race_date_dmy + " " + race_horse_df.race_time.fillna("12h00")
    )

    # Odds rectification
    race_horse_df["odds"] = race_horse_df["odds"] * race_horse_df["race_id"].map(
        race_horse_df.groupby("race_id")["odds"].agg(lambda s: np.sum(1 / s))
    )

    # Remove duplicated races
    date_count_per_race = race_horse_df.groupby("race_id")["race_datetime"].nunique()
    for race_id in date_count_per_race[date_count_per_race != 1].index:
        n_dates = race_horse_df[
            race_horse_df.race_id == race_id
        ].race_datetime.nunique()
        max_race_datetime = max(
            race_horse_df[race_horse_df.race_id == race_id].race_datetime.unique()
        )
        assert (
            len(
                race_horse_df[
                    (race_horse_df.race_id == race_id)
                    & (race_horse_df.race_datetime == max_race_datetime)
                ]
            )
            == len(race_horse_df[race_horse_df.race_id == race_id]) / n_dates
        )
        race_horse_df[race_horse_df.race_id == race_id] = race_horse_df[
            (race_horse_df.race_id == race_id)
            & (race_horse_df.race_datetime == max_race_datetime)
        ]
    race_horse_df = race_horse_df[
        race_horse_df.race_datetime.notna()
    ]  # removing empty races
    # (because of previous operation)

    race_horse_df["n_horses"] = race_horse_df["race_id"].map(
        race_horse_df.groupby("race_id")["horse_id"].count()
    )
    race_horse_df = features.append_features(
        race_horse_df=race_horse_df, historical_race_horse_df=race_horse_df
    )
    race_horse_df.to_csv(
        os.path.join(DATA_DIR, "2012_data_with_features.csv"), index=False
    )


if __name__ == "__main__":
    run()
