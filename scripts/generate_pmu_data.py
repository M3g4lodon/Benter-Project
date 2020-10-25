import os
import re

import pandas as pd
from tqdm import tqdm

import utils
from constants import DATA_DIR
from constants import PMU_DATA_DIR
from utils import features
from utils.features import check_df
from utils.pmu_api_data import convert_queried_data_to_race_horse_df
from utils.pmu_api_data import get_race_horses_records


# TODO investigate driverchange column
# TODO investigate 'engagement' column


def load_queried_data() -> pd.DataFrame:
    race_records = []

    race_count = 0
    for date in tqdm(
        iterable=os.listdir(PMU_DATA_DIR), desc="Loading races per date", unit="day"
    ):
        if date == ".ipynb_checkpoints":
            continue

        assert re.match(r"\d{4}-\d{2}-\d{2}", date)

        folder_path = os.path.join(PMU_DATA_DIR, date)

        programme_json = utils.load_json(
            filename=os.path.join(folder_path, "programme.json")
        )

        if (
            programme_json is None
            or "programme" not in programme_json
            or "reunions" not in programme_json["programme"]
        ):
            continue

        for reunion in programme_json["programme"]["reunions"]:
            for course in reunion["courses"]:
                if course["statut"] in [
                    "COURSE_ANNULEE",
                    "PROGRAMMEE",
                    "ARRIVEE_PROVISOIRE",
                    "COURSE_ANNULEE",
                    "DEPART_CONFIRME",
                    "DEPART_DANS_TROIS_MINUTES",
                ]:
                    continue
                assert course["statut"] in [
                    "FIN_COURSE",
                    "ARRIVEE_DEFINITIVE",
                    "ARRIVEE_DEFINITIVE_COMPLETE",
                    "COURSE_ARRETEE",
                ], course["statut"]

                race_horses_records = get_race_horses_records(
                    programme=programme_json,
                    date=date,
                    r_i=course["numReunion"],
                    c_i=course["numOrdre"],
                    should_be_on_disk=True,
                )
                if race_horses_records is None:
                    continue

                race_records.extend(race_horses_records)
                race_count += 1

    return pd.DataFrame.from_records(data=race_records)


def run():
    queried_race_horse_df = load_queried_data()

    race_horse_df = convert_queried_data_to_race_horse_df(
        queried_race_horse_df=queried_race_horse_df, historical_race_horse_df=None
    )

    # Compute features
    featured_race_horse_df = features.append_features(
        race_horse_df=race_horse_df, historical_race_horse_df=race_horse_df
    )

    check_df(featured_race_horse_df=featured_race_horse_df)

    featured_race_horse_df.to_csv(
        os.path.join(DATA_DIR, "pmu_data_with_features.csv"), index=False
    )


if __name__ == "__main__":
    run()
