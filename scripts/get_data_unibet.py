import math
import os
from typing import List, Generator

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


def get_feature_for_runners(runner_ids: List[int], db_session) -> pd.DataFrame:

    horse_query = f"""
    select
    r.id,
    races.id,
    races.date,
    r.music,
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
            "race_id",
            "race_date",
            "music",
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
    df_features.fillna(value=np.nan, inplace=True)
    df_features["race_date"] = pd.to_datetime(df_features["race_date"])
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


def chunk_producer(runner_ids: List[int]) -> Generator[List[int]]:
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


def try_hard_compute_feature_on_chunk(runner_ids: List[int]) -> pd.DataFrame:
    try:
        return compute_feature_on_chunk(runner_ids=runner_ids)
    except Exception as e:
        logger.warning("%s exception occured, retrying once", e)
        return compute_feature_on_chunk(runner_ids=runner_ids)


def get_dataset_races() -> pd.DataFrame:
    with create_sqlalchemy_session() as db_session:
        runner_ids = _get_all_runner_ids(db_session=db_session)

    df_features_batches = Parallel(n_jobs=n_parallel_jobs, verbose=1)(
        delayed(try_hard_compute_feature_on_chunk)(chunk)
        for chunk in tqdm(
            chunk_producer(runner_ids),
            total=math.ceil(len(runner_ids) / BATCH_SIZE),
            unit=f"x{BATCH_SIZE} Runners",
            desc="Computing Features",
        )
    )

    return pd.concat(df_features_batches)


if __name__ == "__main__":
    df_features = get_dataset_races()
    print(df_features.shape)
    df_features.to_csv(os.path.join(DATA_DIR, "unibet_data_with_features.csv"))
    df_features.to_parquet(
        os.path.join(DATA_DIR, "unibet_data_with_features.parquet"),
        engine="fastparquet",
    )
