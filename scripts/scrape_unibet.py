import datetime as dt
import os

from tqdm import tqdm

import utils
from constants import UNIBET_DATA_DIR
from constants import UNIBET_MIN_DATE
from utils.scrape import check_query_json
from utils.scrape import create_day_folder
from utils.scrape import execute_get_query


def download_day_races(date: dt.date, replace_if_exists: bool = True) -> int:
    query_count = 0
    day_folder_path = os.path.join(UNIBET_DATA_DIR, date.isoformat())

    filename = os.path.join(day_folder_path, f'{"PROGRAMME".lower()}.json')
    url = (
        f"https://www.unibet.fr/zones/turf/program.json?"
        f"date={date.strftime('%Y-%m-%d')}"
    )
    if (
        replace_if_exists
        or not os.path.exists(filename)
        or not check_query_json(filename=filename, url=url)
    ):
        day_races = execute_get_query(url=url)
        query_count += 1
        utils.dump_json(data=day_races, filename=filename)
    else:
        day_races = utils.load_json(filename=filename)
        assert day_races

    if "data" not in day_races:
        print(f"No data for {date.isoformat()}")
        return query_count

    for session in tqdm(day_races["data"], desc=date.isoformat(), leave=False):
        for race in session["races"]:

            filename = os.path.join(
                day_folder_path, f"R{session['rank']}_C{race['rank']}.json"
            )
            url = (
                f'https://www.unibet.fr/zones/turf/race.json?raceId={race["zeturfId"]}'
            )
            if (
                replace_if_exists
                or not os.path.exists(filename)
                or not check_query_json(filename=filename, url=url)
            ):
                utils.dump_json(data=execute_get_query(url=url), filename=filename)
                query_count += 1

            for runner in race["runners"]:
                filename = os.path.join(
                    day_folder_path,
                    f"R{session['rank']}_C{race['rank']}_RN{runner['rank']}.json",
                )
                url = (
                    f"https://www.unibet.fr/zones/turf/statistiques.json?"
                    f'raceId={race["zeturfId"]}&runnerRank={runner["rank"]}'
                )
                if (
                    replace_if_exists
                    or not os.path.exists(filename)
                    or not check_query_json(filename=filename, url=url)
                ):
                    utils.dump_json(data=execute_get_query(url=url), filename=filename)
                    query_count += 1

    return query_count


def run():
    current_date = dt.date(2021, 5, 21)
    today = dt.date.today()

    print(
        f"Scraping Unibet website from "
        f"{UNIBET_MIN_DATE.isoformat()} to {today.isoformat()}"
    )

    query_count = 0
    while current_date < today:
        create_day_folder(date=current_date, source="UNIBET")
        query_count += download_day_races(date=current_date, replace_if_exists=False)
        current_date += dt.timedelta(days=1)

    print(f"{query_count} queries done!")


if __name__ == "__main__":
    run()
