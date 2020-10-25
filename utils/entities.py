import featuretools as ft
import numpy as np
import pandas as pd

from utils import import_data

SOURCE = "PMU"


def get_entity_set(source: str) -> ft.EntitySet:
    train_races_df = import_data.get_split_date(source=source, on_split="train")

    racetrack_columns = ["course_hippodrome", "reunion_pays"]
    racetrack_df = (
        train_races_df[racetrack_columns]
        .drop_duplicates()
        .reset_index()
        .drop(columns="index")
        .reset_index()
    )
    racetrack_df.rename(
        columns={
            "course_hippodrome": "racetrack_name",
            "reunion_pays": "country",
            "index": "id",
        },
        inplace=True,
    )
    racetrack_df = racetrack_df[pd.notna(racetrack_df["racetrack_name"])]
    assert racetrack_df["racetrack_name"].value_counts().max() == 1

    horse_show_columns = ["reunion_nature", "reunion_audience", "course_hippodrome"]
    horse_show_df = (
        train_races_df.groupby(["date", "n_reunion"])[horse_show_columns]
        .first()
        .reset_index()
        .reset_index()
    )
    horse_show_df.rename(columns={"index": "id"}, inplace=True)
    horse_show_df["racetrack_id"] = horse_show_df["course_hippodrome"].map(
        racetrack_df.set_index("racetrack_name")["id"]
    )
    horse_show_df["racetrack_id"].fillna(0, inplace=True)
    horse_show_df["racetrack_id"] = horse_show_df["racetrack_id"].astype(np.int64)

    race_columns = [
        "n_course",
        "n_horses",
        "course_statut",
        "course_discipline",
        "course_specialite",
        "course_condition_sexe",
        "course_condition_age",
        "course_track_type",
        "course_penetrometre",
        "course_corde",
        "course_hippodrome",
        "course_parcours",
        "course_distance",
        "course_distance_unit",
        "course_duration",
        "course_prize_pool",
        "course_winner_prize",
        "date",
        "race_datetime",
        "n_reunion",
        "allure",
    ]

    race_df = train_races_df[race_columns].drop_duplicates()
    race_df["name"] = race_df["date"].str.cat(
        ["R" + race_df["n_reunion"].astype(str), "C" + race_df["n_course"].astype(str)],
        sep="_",
    )

    assert race_df["name"].value_counts().max() == 1

    race_df = race_df.reset_index(drop=True).reset_index()

    race_df.rename(
        columns={"index": "id", "race_datetime": "datetime", "allure": "pace"},
        inplace=True,
    )
    race_df.rename(
        columns={
            column_name: column_name.replace("course_", "")
            for column_name in race_df.columns
            if column_name.startswith("course_")
        },
        inplace=True,
    )

    race_df["horse_show_id"] = [
        horse_show_df[
            (horse_show_df["date"] == d) & (horse_show_df["n_reunion"] == n_r)
        ]["id"].iloc[0]
        for d, n_r in zip(race_df["date"], race_df["n_reunion"])
    ]

    race_df["horse_show_id"] = race_df["horse_show_id"].astype(np.int64)

    trainer_df = pd.Series(
        train_races_df["trainer_name"].unique(), name="trainer_name"
    ).reset_index()
    trainer_df.rename(columns={"index": "id", "trainer_name": "trainer"}, inplace=True)

    jockey_df = pd.Series(
        train_races_df["jockey_name"].unique(), name="name"
    ).reset_index()
    jockey_df.rename(columns={"index": "id"}, inplace=True)

    owner_df = pd.Series(
        train_races_df["owner_name"].unique(), name="name"
    ).reset_index()
    owner_df.rename(columns={"index": "id"}, inplace=True)

    breeder_df = pd.Series(
        train_races_df["breeder_name"].unique(), name="name"
    ).reset_index()
    breeder_df.rename(columns={"index": "id"}, inplace=True)

    stable_df = pd.Series(train_races_df["ecurie"].unique(), name="name").reset_index()
    stable_df.rename(columns={"index": "id"}, inplace=True)

    horse_entity_set = ft.EntitySet("HorseEntitySet")
    horse_entity_set.entity_from_dataframe(
        entity_id="racetrack", dataframe=racetrack_df, index="id"
    )
    horse_entity_set.entity_from_dataframe(
        entity_id="horse_show", dataframe=horse_show_df, index="id"
    )
    horse_entity_set.entity_from_dataframe(
        entity_id="race", dataframe=race_df, index="id"
    )
    horse_entity_set.entity_from_dataframe(
        entity_id="jockey", dataframe=jockey_df, index="id"
    )
    horse_entity_set.entity_from_dataframe(
        entity_id="trainer", dataframe=trainer_df, index="id"
    )
    horse_entity_set.entity_from_dataframe(
        entity_id="owner", dataframe=owner_df, index="id"
    )
    horse_entity_set.entity_from_dataframe(
        entity_id="breeder", dataframe=breeder_df, index="id"
    )
    horse_entity_set.entity_from_dataframe(
        entity_id="stable", dataframe=stable_df, index="id"
    )
    horse_entity_set.add_relationship(
        ft.Relationship(
            horse_entity_set["racetrack"]["id"],
            horse_entity_set["horse_show"]["racetrack_id"],
        )
    )
    horse_entity_set.add_relationship(
        ft.Relationship(
            horse_entity_set["horse_show"]["id"],
            horse_entity_set["race"]["horse_show_id"],
        )
    )

    return horse_entity_set


if __name__ == "__main__":
    get_entity_set(source=SOURCE)
