import datetime as dt
import os
import time

import pytz
import requests
from cachetools import func

import utils
from constants import UNIBET_DATA_DIR
from utils.scrape import create_day_folder
from utils.scrape import execute_get_query

TIMEZONE = "Europe/Paris"
# TODO if not next races

def execute_queries(seconds_before: int, race_times: dict) -> int:
    query_count = 0
    prefix = f"{seconds_before}s_before"
    candidates = {
        keys: time_to_race
        for keys, time_to_race in race_times.items()
        if time_to_race.total_seconds() < seconds_before
    }
    for (date_, r_i, c_i, race_id), _ in candidates.items():
        day_folder_path = os.path.join(UNIBET_DATA_DIR, date_.isoformat())
        filename = os.path.join(day_folder_path, f"{prefix}_R{r_i}_C{c_i}.json")
        if not os.path.exists(filename):
            url = f"https://www.unibet.fr/zones/turf/race.json?raceId={race_id}"
            utils.dump_json(data=execute_get_query(url=url), filename=filename)
            query_count += 1
    return query_count


@func.ttl_cache(maxsize=10, ttl=60)
def query_program(date: dt.date) -> dict:
    return execute_get_query(
        url=f"https://www.unibet.fr/zones/turf/program.json?"
        f"date={date.strftime('%Y-%m-%d')}"
    )


def update():
    tz = pytz.timezone(TIMEZONE)
    dt_now = tz.localize(dt.datetime.now())

    race_times = {}
    for date in [
        dt.date.today(),
        dt.date.today() + dt.timedelta(days=1),
        dt.date.today() - dt.timedelta(days=1),
    ]:
        reunions = query_program(date=date)
        if "data" not in reunions:
            print(f"\rNo program (yet?) for date {date.isoformat()}", end="")
            continue

        reunions = reunions["data"]

        create_day_folder(date=date, source="UNIBET")
        for reunion in reunions:
            for course in reunion["races"]:
                date_ = dt.date.fromtimestamp(course["starttime"] / 1000)
                race_times[
                    (date_, reunion["rank"], course["rank"], course["zeturfId"])
                ] = (
                    dt.datetime.fromtimestamp(course["starttime"] / 1000, tz=tz)
                    - dt_now
                )

    coming_races = {
        keys: time_to_race
        for keys, time_to_race in race_times.items()
        if time_to_race.total_seconds() > 0
    }

    if not coming_races:
        print(
            f"\r[{dt.datetime.now().isoformat()}] Can not find next race "
            ", waiting 1 min...",
            end="",
        )
        time.sleep(60)
        return 0

    time_to_next_race = min(coming_races.values())
    if time_to_next_race.total_seconds() > 60 * 10:
        print(
            f"\r[{dt.datetime.now().isoformat()}] Next race in "
            f"{time_to_next_race}, waiting 1 min...",
            end="",
        )
        time.sleep(60)
        return 0

    query_count = 0
    # 5s
    query_count += execute_queries(seconds_before=5, race_times=coming_races)

    # 30s
    query_count += execute_queries(seconds_before=30, race_times=coming_races)

    # 1min
    query_count += execute_queries(seconds_before=60, race_times=coming_races)

    return query_count


def run():

    print("Query Odds before races from Unibet/turf")

    query_count = 0
    retry_count = 0
    while True:
        start_date = dt.datetime.now()
        new_queries = None
        try:
            new_queries = update()
        except requests.ConnectionError as e:
            retry_count += 1
            print(
                f"\r[{start_date.isoformat()}] Connection Error n°{retry_count}: {e}",
                end="",
            )
        end_date = dt.datetime.now()
        if new_queries:
            print(
                f"\r[{end_date.isoformat()}] Time to execute: "
                f"{(end_date-start_date).total_seconds():.2}s, "
                f"{new_queries} queries made (total: {query_count})",
                end="\n",
            )
            query_count += new_queries
        print(
            f"\r[{end_date.isoformat()}] Time to execute: "
            f"{(end_date-start_date).total_seconds():.2}s, "
            f"{query_count} queries in total",
            end="",
        )


if __name__ == "__main__":
    run()
