"""

Features ideas

https://www.zone-turf.fr/guide-turf/plat-2.html
https://communaute-aide.pmu.fr/questions/1336908-lire-tableau-partants-pmu-fr
https://www.lexiqueducheval.net/lexique_turf_engl.html

DONE v1

    win rate horse, jockey, trainer
    mean place horse, jockey, trainer
    horse_age, horse_sex
    horse race
    pregnant
    breeder
    unshod
    corde
    parcours
    blinker
    handicap
    gains en cours, carrière, année précédente,
    enjeu de la course (pool prize, winner prize)
    weight, actual weights
    driver
    race_type
    état du sol (type sol, penetrometre)
    duration since last race -> performance.json (remaining to
        compute timedelta between last_race and date)


TODO

    ecuries
From programme

    numéro course jockey dans la journée
    nombre de course courus par le jockey dans la journée

From past races

    expérience: nombre de course du cheval/horse/trainer dans l'historique
    inédit pour cheval (or in partipants), jockey,trainer
    new race type for horse, jockey, trainer
    number of days since last race (horse  from perforamnce.json)
    last race position
    previous horse speed
    mean speed (bins?)
    max speed (bins?)
    elo ranking horse, jockey, trainer (from historic)
    max elo ranking father/mother/father_mother
    track draw advatage from historic
    already race together horse+driver

directly in course


in participants
"""
import functools

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.music import parse_music


def is_complete_past_races(rh_serie: pd.Series, horse_history: pd.DataFrame) -> bool:
    if rh_serie.indicateurInedit:
        return True

    previous_horse_races = horse_history[
        horse_history["race_datetime"] < rh_serie.race_datetime
    ]
    if not (
        previous_horse_races["n_run_races"] == np.arange(len(previous_horse_races))
    ).all():
        return False

    n_past_races = len(previous_horse_races.index)
    n_run_races = max(
        parse_music(rh_serie.musique).n_races_in_music or 0, rh_serie.n_run_races
    )

    return n_run_races == n_past_races


def get_race_horse_features(
    rh_serie: pd.Series,
    horse_history: pd.DataFrame,
    jockey_history: pd.DataFrame,
    trainer_history: pd.DataFrame,
):
    parsed_music = parse_music(music=rh_serie.musique)

    win_rate = parsed_music.win_rate

    if "n_run_races" in rh_serie and "n_won_races" in rh_serie:
        win_rate = (
            None
            if rh_serie["n_run_races"] == 0.0
            else rh_serie["n_won_races"] / rh_serie["n_run_races"]
        )

    previous_jockey_races = jockey_history[
        jockey_history.race_datetime < rh_serie.race_datetime
    ]
    jockey_win_rate = (previous_jockey_races.horse_place == 1).mean()
    jockey_mean_place = previous_jockey_races.horse_place.mean()

    previous_trainer_races = trainer_history[
        trainer_history.race_datetime < rh_serie.race_datetime
    ]
    trainer_win_rate = (previous_trainer_races.horse_place == 1).mean()
    trainer_mean_place = previous_trainer_races.horse_place.mean()

    return {
        "win_ratio": win_rate,
        "mean_place": parsed_music.mean_place,
        "jockey_win_rate": jockey_win_rate,
        "jockey_mean_place": jockey_mean_place,
        "trainer_win_rate": trainer_win_rate,
        "trainer_mean_place": trainer_mean_place,
        "complete_past_races": is_complete_past_races(
            rh_serie=rh_serie, horse_history=horse_history
        ),
    }


def append_features(
    race_horse_df: pd.DataFrame, historical_race_horse_df: pd.DataFrame
) -> pd.DataFrame:
    @functools.lru_cache(maxsize=None)
    def get_horse_history(horse_id: int) -> pd.DataFrame:
        return race_horse_df[
            (race_horse_df["horse_id"] == horse_id)
            & (race_horse_df["statut"] == "PARTANT")
        ][["race_datetime", "n_run_races"]].dropna(axis=0)

    @functools.lru_cache(maxsize=None)
    def get_jockey_history(jockey_name: str) -> pd.DataFrame:
        return historical_race_horse_df[
            historical_race_horse_df["jockey_name"] == jockey_name
        ][["race_datetime", "horse_place"]].dropna(axis=0)

    @functools.lru_cache(maxsize=None)
    def get_trainer_history(trainer_name: str) -> pd.DataFrame:
        return historical_race_horse_df[
            historical_race_horse_df["trainer_name"] == trainer_name
        ][["race_datetime", "horse_place"]].dropna(axis=0)

    records = []

    # TODO multiprocess this (https://github.com/tqdm/tqdm/issues/484)
    for rh_serie in tqdm(
        race_horse_df.itertuples(index=True, name="Race_Horse"),
        total=len(race_horse_df),
        desc="Appending features",
        unit="horse_in_race",
    ):
        features_dict = get_race_horse_features(
            rh_serie=rh_serie,
            horse_history=get_horse_history(horse_id=rh_serie.horse_id),
            jockey_history=get_jockey_history(jockey_name=rh_serie.jockey_name),
            trainer_history=get_trainer_history(trainer_name=rh_serie.trainer_name),
        )
        features_dict.update({"Index": rh_serie.Index})
        records.append(features_dict)

    res = race_horse_df.join(pd.DataFrame(records).set_index("Index"))
    return res


def check_df(featured_race_horse_df: pd.DataFrame):
    assert not featured_race_horse_df["odds"].isna().all()
    assert not featured_race_horse_df["totalEnjeu"].isna().all()
