import datetime as dt
import os

from tqdm import tqdm

import utils
from constants import PMU_MIN_DATE
from constants import SOURCE_PMU
from utils.pmu_api_data import get_pmu_api_url
from utils.scrape import check_query_json
from utils.scrape import create_day_folder
from utils.scrape import execute_get_query


def download_day_races(date: dt.date, replace_if_exists: bool = True) -> int:
    query_count = 0
    day_folder_path = utils.get_folder_path(source=SOURCE_PMU, date=date)
    assert day_folder_path
    filename = os.path.join(day_folder_path, f'{"PROGRAMME".lower()}.json')
    url = get_pmu_api_url(url_name="PROGRAMME", date=date)

    if (
        replace_if_exists
        or not os.path.exists(filename)
        or not check_query_json(filename=filename, url=url)
    ):
        day_races = execute_get_query(url=url)
        query_count += 1
        utils.dump_json(data=day_races, filename=filename)
    else:
        day_races = utils.load_json(filename)
        assert day_races is not None
        # TODO scrape again if bad statut in course/reunion

    if "programme" not in day_races or "reunions" not in day_races["programme"]:
        return query_count
    reunions = day_races["programme"]["reunions"]

    for reunion in tqdm(reunions, desc=date.isoformat(), leave=False):
        for course in reunion["courses"]:
            r_i = course["numReunion"]
            c_i = course["numOrdre"]

            for url_name in [
                "PARTICIPANTS",
                "PRONOSTIC",
                "PRONOSTIC_DETAILLE",
                "ENJEU",
                "PERFORMANCE",
                "CITATIONS",
                "RAPPORTS_DEF",
                "COMBINAISONS",
                "COMBINAISONS_INTERNET",
                "CITATIONS_INTERNET",
                "RAPPORTS_DEF_INTERNET",
                "ENJEU_INTERNET",
            ]:
                filename = os.path.join(
                    day_folder_path, f"R{r_i}_C{c_i}_{url_name.lower()}.json"
                )
                url = get_pmu_api_url(url_name=url_name, r_i=r_i, c_i=c_i, date=date)
                if (
                    replace_if_exists
                    or not os.path.exists(filename)
                    or not check_query_json(filename=filename, url=url)
                ):
                    utils.dump_json(data=execute_get_query(url=url), filename=filename)
                    query_count += 1

            for url_name in ["RAPPORTS"]:
                for code_pari in ["E_SIMPLE_GAGNANT", "SIMPLE_GAGNANT"]:
                    filename = os.path.join(
                        day_folder_path,
                        f"R{r_i}_C{c_i}_{code_pari}_{url_name.lower()}.json",
                    )
                    url = get_pmu_api_url(
                        url_name=url_name,
                        r_i=r_i,
                        c_i=c_i,
                        code_pari=code_pari,
                        date=date,
                    )
                    if (
                        replace_if_exists
                        or not os.path.exists(filename)
                        or not check_query_json(filename=filename, url=url)
                    ):
                        utils.dump_json(
                            data=execute_get_query(url=url), filename=filename
                        )
                        query_count += 1

    return query_count


def run():
    current_date = dt.date(2021, 4, 1)
    today = dt.date.today()

    print(
        f"Scraping PMU website from {PMU_MIN_DATE.isoformat()} to {today.isoformat()}"
    )

    query_count = 0
    while current_date < today:
        create_day_folder(date=current_date, source="PMU")
        query_count += download_day_races(date=current_date, replace_if_exists=False)
        current_date += dt.timedelta(days=1)

    print(f"{query_count} queries done!")


if __name__ == "__main__":
    run()
