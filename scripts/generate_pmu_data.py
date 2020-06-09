import os
import re
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from constants import PMU_DATA_DIR, DATA_DIR
from utils.pmu_api_data import get_race_horses_records
from utils import features

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
                r_i = course["numReunion"]
                c_i = course["numOrdre"]

                race_horses_records = get_race_horses_records(
                    programme=programme_json,
                    date=date,
                    r_i=r_i,
                    c_i=c_i,
                    should_be_on_disk=True,
                )
                if race_horses_records is None:
                    continue

                race_records.extend(race_horses_records)
                race_count += 1

    return pd.DataFrame.from_records(data=race_records)


def _create_horse_name_mapper(rh_df: pd.DataFrame) -> Dict[str, int]:
    assert "horse_name" in rh_df
    assert "father_horse_name" in rh_df
    assert "mother_horse_name" in rh_df
    assert "father_mother_horse_name" in rh_df

    horse_names = rh_df["horse_name"].append(
        (
            rh_df["father_horse_name"],
            rh_df["mother_horse_name"],
            rh_df["father_mother_horse_name"],
        )
    )
    horse_names = horse_names.str.upper()
    horse_names.dropna(inplace=True)
    horse_names = pd.Series(horse_names.unique())
    horse_names.name = "horse_name"
    mapper_df = horse_names.reset_index().set_index("horse_name")
    return mapper_df.to_dict()["index"]


def convert_queried_data_to_race_horse_df(
    queried_race_horse_df: pd.DataFrame,
    historical_race_horse_df: Optional[pd.DataFrame],
)->pd.DataFrame:
    """Will convert queried race horse df into the right format.

    If `historical_race_horse_df`is provided, the converted race/horses will be formatted as an extension of it """

    race_horse_df = queried_race_horse_df

    for col_name in ["date", "n_reunion", "n_course", "statut", "totalEnjeu"]:
        assert col_name in race_horse_df, f"{col_name} column is missing race_horse_df"

    for col_name in [
        "nom",
        "nomPere",
        "nomMere",
        "nomPereMere",
        "sexe",
        "numPmu",
        "driver",
        "age",
        "race",
        "jumentPleine",
        "entraineur",
        "proprietaire",
        "eleveur",
        "ordreArrivee",
        "nombreCourses",
        "nombreVictoires",
        "nombrePlaces",
        "deferre",
        "tempsObtenu",
        "gainsAnneeEnCours",
        "gainsAnneePrecedente",
        "gainsCarriere",
        "oeilleres",
        "handicapDistance",
        "handicapValeur",
        "last_race_date"
    ]:
        if col_name not in race_horse_df:
            race_horse_df[col_name] = np.nan

    columns_renaming = {
        "nom": "horse_name",
        "nomPere": "father_horse_name",
        "nomMere": "mother_horse_name",
        "nomPereMere": "father_mother_horse_name",
        "sexe": "horse_sex",
        "numPmu": "horse_number",
        "driver": "jockey_name",
        "age": "horse_age",
        "race": "horse_race",
        "jumentPleine": "is_pregnant",
        "entraineur": "trainer_name",
        "proprietaire": "owner_name",
        "eleveur": "breeder_name",
        "ordreArrivee": "horse_place",
        "nombreCourses": "n_run_races",
        "nombreVictoires": "n_won_races",
        "nombrePlaces": "n_placed_races",
        "deferre": "unshod",
        "tempsObtenu": "horse_race_duration",
        "gainsAnneeEnCours": "horse_current_year_prize",
        "gainsAnneePrecedente": "horse_last_year_prize",
        "gainsCarriere": "horse_career_prize",
        "oeilleres": "blinkers",
        "handicapDistance": "handicap_distance",
        "handicapValeur": "handicap_value",
    }
    race_horse_df = queried_race_horse_df.rename(columns=columns_renaming)

    race_horse_df[race_horse_df["unshod"] == "REFERRE_ANTERIEURS_POSTERIEURS"][
        "unshod"
    ] = np.nan

    # Compute horse_id
    # TODO check unique mother/father/father_mother

    mapper_dict = {}
    max_old_race_id = -1

    if historical_race_horse_df is not None:
        for col in ["horse", "father_horse", "mother_horse", "father_mother_horse"]:
            col_id = f"{col}_id"
            col_name = f"{col}_name"
            horse_name_df = historical_race_horse_df[
                [col_id, col_name]
            ].drop_duplicates()
            horse_name_df.dropna(inplace=True)
            horse_name_df[col_name] = horse_name_df[col_name].str.upper()
            horse_name_df.drop_duplicates(inplace=True)
            mapper_dict.update(horse_name_df.set_index(col_name)[col_id].to_dict())

        max_old_race_id = max(mapper_dict.values())

    mapper_dict.update(
        {
            horse_name: index + max_old_race_id + 1
            for horse_name, index in _create_horse_name_mapper(
                rh_df=race_horse_df
            ).items()
            if horse_name not in mapper_dict
        }
    )

    race_horse_df["horse_id"] = (
        race_horse_df["horse_name"].str.upper().map(mapper_dict, na_action="ignore")
    )
    race_horse_df["father_horse_id"] = (
        race_horse_df["father_horse_name"]
        .str.upper()
        .map(mapper_dict, na_action="ignore")
    )
    race_horse_df["mother_horse_id"] = (
        race_horse_df["mother_horse_name"]
        .str.upper()
        .map(mapper_dict, na_action="ignore")
    )
    if race_horse_df["father_mother_horse_name"].isna().all():
        race_horse_df["father_mother_horse_id"] = np.nan
    else:
        race_horse_df["father_mother_horse_id"] = (
            race_horse_df["father_mother_horse_name"]
            .str.upper()
            .map(mapper_dict, na_action="ignore")
        )

    # Compute race_id
    mapper_dict = {}
    max_old_race_id = -1
    if historical_race_horse_df is not None:
        horse_name_df = historical_race_horse_df[
            ["date", "n_reunion", "n_course", "race_id"]
        ].drop_duplicates()
        horse_name_df.dropna(inplace=True)
        mapper_dict.update(
            horse_name_df.set_index(["date", "n_reunion", "n_course"])[
                "race_id"
            ].to_dict()
        )

        max_old_race_id = max(mapper_dict.values())

    drc_df = race_horse_df[["date", "n_reunion", "n_course"]]
    mapper_df = drc_df.drop_duplicates().reset_index(drop=True)
    mapper_dict.update(
        {
            (row.date, row.n_reunion, row.n_course): index + max_old_race_id + 1
            for index, row in mapper_df.iterrows()
            if (row.date, row.n_reunion, row.n_course) not in mapper_dict
        }
    )
    race_horse_df["race_id"] = [
        mapper_dict[drc] for drc in drc_df.itertuples(index=False)
    ]
    race_horse_df["n_horses"] = race_horse_df["race_id"].map(
        race_horse_df.groupby("race_id")["statut"].agg(lambda s: (s == "PARTANT").sum())
    )

    # Compute odds and pari_mutuel_proba
    race_horse_df["totalEnjeu"] = race_horse_df["totalEnjeu"]  # in cents
    sum_per_race = race_horse_df.groupby("race_id")["totalEnjeu"].sum()

    race_horse_df["pari_mutuel_proba"] = race_horse_df["totalEnjeu"] / race_horse_df[
        "race_id"
    ].map(sum_per_race)
    race_horse_df["odds"] = 1 / race_horse_df["pari_mutuel_proba"]

    race_horse_df["duration_since_last_race"] = (
            pd.to_datetime(race_horse_df["date"]).dt.date
            - pd.to_datetime(race_horse_df["last_race_date"]).dt.date
    )

    return race_horse_df


def check_df(featured_race_horse_df: pd.DataFrame):
    assert not featured_race_horse_df["odds"].isna().all()
    assert not featured_race_horse_df["totalEnjeu"].isna().all()


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
