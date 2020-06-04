import datetime as dt
import os
import time

from cachetools import func
import pytz

from scripts.scrape_pmu_data import get_pmu_api_url
from utils.scrape import execute_get_query, create_day_folder

import utils
from constants import PMU_DATA_DIR

TIMEZONE = "Europe/Paris"
# TODO scrape pronostics in real time (only available for few weeks)
# TODO Handle ConnectionError error

def execute_queries(seconds_before: int, race_times: dict) -> int:
    query_count = 0
    prefix = f"{seconds_before}s_before"
    candidates = {
        (date, r_i, c_i): time_to_race
        for (date, r_i, c_i), time_to_race in race_times.items()
        if time_to_race.total_seconds() < seconds_before
    }
    for (date_, r_i, c_i), _ in candidates.items():
        for url_name in [
            "CITATIONS_INTERNET",
            "CITATIONS",
        ]:
            day_folder_path = os.path.join(PMU_DATA_DIR, date_.isoformat())
            filename = os.path.join(
                day_folder_path, f"{prefix}_R{r_i}_C{c_i}_{url_name.lower()}.json"
            )
            if not os.path.exists(filename):
                url = get_pmu_api_url(url_name=url_name, r_i=r_i, c_i=c_i, date=date_)
                utils.dump_json(data=execute_get_query(url=url), filename=filename)
                query_count += 1
    return query_count


@func.ttl_cache(maxsize=10, ttl=60 * 10)
def query_program(date: dt.date) -> dict:
    return execute_get_query(url=get_pmu_api_url(url_name="PROGRAMME", date=date))


def update():
    tz = pytz.timezone(TIMEZONE)
    dt_now = tz.localize(dt.datetime.now())

    race_times = {}
    for date in [
        dt.date.today(),
        dt.date.today() + dt.timedelta(days=1),
        dt.date.today() - dt.timedelta(days=1),
    ]:
        programme = query_program(date=date)

        if "programme" not in programme:
            print(f"\rNo program (yet?) for date {date.isoformat()}", end="")
            continue

        create_day_folder(date=date, source="PMU")
        for reunion in programme["programme"]["reunions"]:
            for course in reunion["courses"]:
                date_ = dt.date.fromtimestamp(programme["programme"]["date"] / 1000)
                race_times[(date_, course["numReunion"], course["numOrdre"])] = (
                    dt.datetime.fromtimestamp(
                        course["heureDepart"] / 1000,
                        tz=dt.timezone(
                            dt.timedelta(milliseconds=course["timezoneOffset"])
                        ),
                    )
                    - dt_now
                )

    coming_races = {
        (date, r_i, c_i): time_to_race
        for (date, r_i, c_i), time_to_race in race_times.items()
        if time_to_race.total_seconds() > 0
    }
    time_to_next_race = min(coming_races.values())
    if time_to_next_race.total_seconds() > 60 * 10:
        print(
            f"\r[{dt.datetime.now().isoformat()}] Next race in {time_to_next_race}, waiting 1 min...",
            end="",
        )
        time.sleep(60)

    query_count = 0
    # 5s
    query_count += execute_queries(seconds_before=5, race_times=coming_races)

    # 30s
    query_count += execute_queries(seconds_before=30, race_times=coming_races)

    # 1min
    query_count += execute_queries(seconds_before=60, race_times=coming_races)

    return query_count


def run():

    print("Query Odds before races from pmu.fr/turf")

    query_count = 0
    while True:
        start_date = dt.datetime.now()
        new_queries = update()
        end_date = dt.datetime.now()
        if new_queries:
            print(
                f"\r[{end_date.isoformat()}] Time to execute: "
                f"{(end_date-start_date).total_seconds():.2}s, "
                f"{new_queries} queries made",
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
