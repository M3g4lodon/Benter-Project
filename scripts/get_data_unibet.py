import math
import os
from typing import Iterator
from typing import List

import dask.dataframe as dd
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from constants import DATA_DIR
from database.setup import create_sqlalchemy_session
from models.runner import Runner
from utils.logger import setup_logger
from utils.music import MusicRank
from utils.music import parse_unibet_music

BATCH_SIZE = int(1e3)
n_parallel_jobs = 1
N_RETRY = 2

logger = setup_logger(__name__)

# TODO features on comment (NLP)
# TODO features on pronostic (NLP)
# TODO features on conditions (NLP)
# TODO features on owner (has_person, has_organisation)
# TODO n_days_since_last_days
# TODO time features on months (season)
# TODO features on lat long
# TODO horse_previous_speed
# TODO horse_median_speed
# TODO horse_has_already_run_in_racetrack


def get_feature_for_runners(runner_ids: List[int], db_session) -> pd.DataFrame:

    horse_query = f"""
    select
    r.id,
    r.horse_id,
    races.id,
    races.date,
    races.start_at,
    r.music,
    r.final_odds as odds,
    r.position as horse_place,
    r.weight,
    r.rope_n,
    r.blinkers,
    r.shoes,
    r.stakes,
    r.sex,
    r.age,
    r.team !=0 as is_in_team,
    r.coat,
    races.distance,
    races.stake,
    races.type,
    runners_with_history.n_horse_previous_races,
    runners_with_history.n_horse_previous_positions,
    runners_with_history.average_horse_position,
    runners_with_history.average_horse_top_1,
    runners_with_history.average_horse_top_3
from
    runners r
join races on
    r.race_id = races.id
left join(
    select
        r.id,
        count(previous_horse_runners.id) as n_horse_previous_races,
        count(cast( previous_horse_runners."position" as integer)) as n_horse_previous_positions,
        AVG(cast( previous_horse_runners."position" as integer)) as average_horse_position,
        AVG(cast(previous_horse_runners.top_1 as integer)) as average_horse_top_1,
        AVG(cast(previous_horse_runners.top_3 as integer)) as average_horse_top_3
    from
        runners r
    join races on
        races.id = r.race_id
    join (
        select
            r1.*,
            cast(r1."position" as integer)= 1 as top_1,
            cast(r1."position" as integer)<= 3 as top_3
        from
            runners r1) previous_horse_runners on
        previous_horse_runners.horse_id = r.horse_id
    join races previous_horse_races on
        previous_horse_races.id = previous_horse_runners.race_id
    where
        previous_horse_races.date < races.date
        and r.id in ({','.join(str(runner_id) for runner_id in runner_ids)})
    group by
        r.id,
        races.date) runners_with_history on
    runners_with_history.id = r.id
where
    r.id in ({','.join(str(runner_id) for runner_id in runner_ids)})
    """

    df_features = pd.DataFrame(
        db_session.execute(horse_query).fetchall(),
        columns=[
            "runner_id",
            "horse_id",
            "race_id",
            "race_date",
            "race_datetime",
            "music",
            "odds",
            "horse_place",
            "weight",
            "rope_n",
            "blinkers",
            "shoes",
            "horse_stakes",
            "sex",
            "age",
            "is_in_team",
            "coat",
            "race_distance",
            "race_stake",
            "race_type",
            "n_horse_previous_races",
            "n_horse_previous_positions",
            "average_horse_position",
            "average_horse_top_1",
            "average_horse_top_3",
        ],
    )
    df_features.set_index("runner_id", inplace=True)

    for entity_name in ["jockey", "trainer", "owner"]:
        query = f"""
select
    r.id,
    r.{entity_name}_id,
    runners_with_history.n_{entity_name}_previous_races,
    runners_with_history.n_{entity_name}_previous_positions,
    runners_with_history.average_{entity_name}_position,
    runners_with_history.average_{entity_name}_top_1,
    runners_with_history.average_{entity_name}_top_3
from
    runners r
join races on
    r.race_id = races.id
left join(
        select
            r.id,
            r.{entity_name}_id,
            count(previous_{entity_name}_runners.id) as n_{entity_name}_previous_races,
            count(cast( previous_{entity_name}_runners."position" as integer)) as n_{entity_name}_previous_positions,
            AVG(cast( previous_{entity_name}_runners."position" as integer)) as average_{entity_name}_position,
            AVG(cast(previous_{entity_name}_runners.top_1 as integer)) as average_{entity_name}_top_1,
            AVG(cast(previous_{entity_name}_runners.top_3 as integer)) as average_{entity_name}_top_3
        from
            runners r
        join races on
            races.id = r.race_id
        join (select r1.*, cast(r1."position" as integer)=1  as top_1,
            cast(r1."position" as integer)<=3  as top_3 from runners r1
              ) previous_{entity_name}_runners on
            previous_{entity_name}_runners.{entity_name}_id = r.{entity_name}_id
        join races previous_{entity_name}_races on
            previous_{entity_name}_races.id = previous_{entity_name}_runners.race_id
        where
            previous_{entity_name}_races.date < races.date
            and r.id in ({','.join(str(runner_id) for runner_id in runner_ids)})
        group by r.id) runners_with_history on
    runners_with_history.id = r.id
where
    r.id in ({','.join(str(runner_id) for runner_id in runner_ids)})"""
        df_feature_sub = pd.DataFrame(
            db_session.execute(query).fetchall(),
            columns=[
                "runner_id",
                f"{entity_name}_id",
                f"n_{entity_name}_previous_races",
                f"n_{entity_name}_previous_positions",
                f"average_{entity_name}_position",
                f"average_{entity_name}_top_1",
                f"average_{entity_name}_top_3",
            ],
        )
        df_feature_sub.set_index("runner_id", inplace=True)
        df_features = df_features.join(df_feature_sub, on="runner_id")

    query = f"""
    select
        r.id,
        horse_shows.ground,
        race_tracks.country_name
    from
        runners r
    join races on
        races.id = r.race_id
    join horse_shows on
        horse_shows.id = races.horse_show_id
    join race_tracks on
        race_tracks.id = horse_shows.race_track_id
        """
    df_feature_sub = pd.DataFrame(
        db_session.execute(query).fetchall(),
        columns=["runner_id", "horse_show_ground", "race_track_country"],
    )
    df_feature_sub.set_index("runner_id", inplace=True)
    df_features = df_features.join(df_feature_sub, on="runner_id")

    df_features.fillna(value=np.nan, inplace=True)
    df_features["race_date"] = pd.to_datetime(df_features["race_date"])
    df_features["horse_place"] = pd.to_numeric(
        df_features["horse_place"], downcast="integer"
    )
    race_id_n_horses = {
        race_id: n_horses
        for (race_id, n_horses) in db_session.execute(
            f"""select r.race_id, count(1) as "n_horses"
    from runners r
    group by r.race_id"""
        ).fetchall()
    }
    df_features["n_horses"] = df_features["race_id"].map(race_id_n_horses)
    return df_features


def _get_all_runner_ids(db_session) -> List[int]:
    return [r[0] for r in db_session.query(Runner.id).all()]


def _get_all_race_ids_with_n_horse(n_horses: int, db_session) -> List[int]:

    race_ids = db_session.execute(
        f"""select race_id
from (
select r.race_id, count(1) as "n_horses"
from runners r
group by r.race_id ) as race_n_horses
where race_n_horses.n_horses={n_horses}"""
    ).fetchall()

    return [race_id[0] for race_id in race_ids]


def fusion_horse_feature(row):
    parsed_music = parse_unibet_music(row.race_date.year, row.music)

    mean_place, win_ratio = np.nan, np.nan
    if parsed_music and len(parsed_music.events) > row.n_horse_previous_races:
        mean_place = np.mean(
            [
                event.rank.value
                for event in parsed_music.events
                if isinstance(event.rank.value, int)
                and event.rank != MusicRank.TENTH_AND_BELOW
            ]
        )
        win_ratio = np.mean(
            [event.rank == MusicRank.FIRST for event in parsed_music.events]
        )

    elif row.n_horse_previous_races:
        mean_place = row.average_horse_position
        win_ratio = row.average_horse_top_1

    return pd.Series({"mean_horse_place": mean_place, "average_horse_top_1": win_ratio})


def chunk_producer(runner_ids: List[int]) -> Iterator[List[int]]:
    for i in range(0, len(runner_ids), BATCH_SIZE):
        yield runner_ids[i : i + BATCH_SIZE]


def compute_feature_on_chunk(runner_ids: List[int]) -> pd.DataFrame:
    with create_sqlalchemy_session() as db_session:
        df_features_batch = get_feature_for_runners(
            runner_ids=runner_ids, db_session=db_session
        )

        df_features_batch = pd.merge(
            df_features_batch,
            df_features_batch.apply(fusion_horse_feature, axis=1),
            left_index=True,
            right_index=True,
        )
    return df_features_batch


def try_hard_compute_feature_on_chunk(runner_ids: List[int]) -> int:
    first_runner_id = runner_ids[0]
    file_path = f"./data/unibet_cache/chunk_first_runner_id_{first_runner_id}.csv"
    if os.path.exists(file_path):
        return first_runner_id
    try:
        res = compute_feature_on_chunk(runner_ids=runner_ids)
        res.to_csv(file_path)
        return first_runner_id
    except Exception as e:
        logger.warning("%s exception occurred, retrying once", e)
        res = compute_feature_on_chunk(runner_ids=runner_ids)
        res.to_csv(file_path)
        return first_runner_id


def get_dataset_races() -> pd.DataFrame:
    with create_sqlalchemy_session() as db_session:
        runner_ids = _get_all_runner_ids(db_session=db_session)

    first_runner_ids = Parallel(n_jobs=n_parallel_jobs, verbose=1)(
        delayed(try_hard_compute_feature_on_chunk)(chunk)
        for chunk in tqdm(
            chunk_producer(runner_ids),
            total=math.ceil(len(runner_ids) / BATCH_SIZE),
            unit=f"x{BATCH_SIZE} Runners",
            desc="Computing Features",
        )
    )

    for runner_ids in chunk_producer(runner_ids):
        first_runner_id = runner_ids[0]
        assert first_runner_id in first_runner_ids
        file_path = f"./data/unibet_cache/chunk_first_runner_id_{first_runner_id}.csv"
        assert os.path.exists(file_path)
    return dd.read_csv(
        "./data/unibet_cache/chunk_first_runner_id_*.csv", assume_missing=True
    )


if __name__ == "__main__":
    df_features = get_dataset_races()
    print(df_features.shape)
    df_features.to_csv(os.path.join(DATA_DIR, "unibet_data_with_features.csv"))
    df_features.to_parquet(
        os.path.join(DATA_DIR, "unibet_data_with_features.parquet"),
        engine="fastparquet",
    )
