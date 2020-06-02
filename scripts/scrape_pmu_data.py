import datetime as dt
import os
from typing import Optional, Union

from tqdm import tqdm

import utils
from constants import PMU_DATA_DIR, PMU_MIN_DATE
from utils.scrape import execute_get_query, create_day_folder, check_query_json


def get_pmu_api_url(
    url_name: str,
    date: Union[dt.datetime, dt.date],
    r_i: Optional[int] = None,
    c_i: Optional[int] = None,
    code_pari: Optional[str] = None,
) -> str:
    pmu_programme_api_url = f"https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{date.strftime('%d%m%Y')}"
    if url_name == "PROGRAMME":
        return pmu_programme_api_url

    assert r_i is not None
    assert c_i is not None

    assert r_i > 0
    assert c_i > 0

    _pmu_programme_course_api_url = f"{pmu_programme_api_url}/R{r_i}/C{c_i}"

    if url_name == "PARTICIPANTS":
        return f"{_pmu_programme_course_api_url}/participants"
    elif url_name == "PRONOSTIC":
        return f"{_pmu_programme_course_api_url}/pronostics"
    elif url_name == "PRONOSTIC_DETAILLE":
        return f"{_pmu_programme_course_api_url}/pronostics-detailles"
    elif url_name == "PERFORMANCE":
        return f"{_pmu_programme_course_api_url}/performances-detaillees/pretty"
    elif url_name == "ENJEU":
        return f"{_pmu_programme_course_api_url}/masse-enjeu-v2"
    elif url_name == "ENJEU_INTERNET":
        return f"{_pmu_programme_course_api_url}/masse-enjeu-v2?specialisation=INTERNET"
    elif url_name == "CITATIONS":
        return f"{_pmu_programme_course_api_url}/citations"
    elif url_name == "RAPPORTS_DEF":
        return f"{_pmu_programme_course_api_url}/rapports-definitifs"
    elif url_name == "RAPPORTS_DEF_INTERNET":
        return f"{_pmu_programme_course_api_url}/rapports-definitifs?specialisation=INTERNET"
    elif url_name == "COMBINAISONS":
        return f"{_pmu_programme_course_api_url}/combinaisons"
    elif url_name == "COMBINAISONS_INTERNET":
        return f"{_pmu_programme_course_api_url}/combinaisons?specialisation=INTERNET"
    elif url_name == "CITATIONS_INTERNET":
        return f"{_pmu_programme_course_api_url}/citations?specialisation=INTERNET"
    # TODO compare w/ or w/o specialisation on combinaisons, look at E_SIMPLE_GAGNANT vs SIMPLE_GAGNANT

    assert code_pari is not None

    PMU_RAPPORTS_API_URL = f"{_pmu_programme_course_api_url}/rapports/{code_pari}"
    assert url_name == "RAPPORTS"
    return PMU_RAPPORTS_API_URL


def download_day_races(date: dt.date, replace_if_exists: bool = True) -> int:
    query_count = 0
    day_folder_path = os.path.join(PMU_DATA_DIR, date.isoformat())

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
    current_date = PMU_MIN_DATE
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
