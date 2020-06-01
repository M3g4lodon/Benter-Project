import datetime as dt
import functools

import numpy as np
import pandas as pd
import pytz
from tabulate import tabulate

from constants import PMU_BETTINGS
from scripts.generate_pmu_data import (
    get_num_pmu_enjeu,
    convert_queried_data_to_race_horse_df,
)
from scripts.scrape_pmu_data import get_pmu_api_url
from utils.scrape import execute_get_query
from utils import features
from utils import import_data
from utils import model as utils_model
from utils import wagering_stategy

TIMEZONE = "Europe/Paris"

code_pari = "E_SIMPLE_GAGNANT"
capital_fraction = 0.01


def get_num_pmu_citation_enjeu(citations: dict, pari_type: str):
    if "listeCitations" not in citations:
        return None
    citations_ = [
        citation
        for citation in citations["listeCitations"]
        if citation["typePari"] == code_pari
    ]
    assert len(citations_) <= 1
    if not citations_:
        return None
    citation = citations_[0]
    if "participants" not in citation:
        return None

    return {
        part["numPmu"]: part["citations"][0]["enjeu"]
        for part in citation["participants"]
    }


@functools.lru_cache(maxsize=None)
def _variables_to_query_once():
    print("Setting up constants variables")
    historical_race_horse_df = import_data.load_featured_data(source="PMU")

    utils_model.load_shared_layers(trainable=False)
    print("Winning Horse model is loaded")
    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    return historical_race_horse_df, track_take


def suggest_betting_on_next_race():
    (
        historical_race_horse_df,
        track_take,
    ) = _variables_to_query_once()

    date = dt.date.today()

    tz = pytz.timezone(TIMEZONE)
    dt_now = tz.localize(dt.datetime.now())

    programme = execute_get_query(url=get_pmu_api_url(url_name="PROGRAMME", date=date))

    race_times = {}
    for reunion in programme["programme"]["reunions"]:
        for course in reunion["courses"]:
            race_times[(date, course["numReunion"], course["numOrdre"])] = (
                dt.datetime.fromtimestamp(
                    course["heureDepart"] / 1000,
                    tz=dt.timezone(dt.timedelta(milliseconds=course["timezoneOffset"])),
                )
                - dt_now
            )

    coming_races = {
        (date, r_i, c_i): time_to_race
        for (date, r_i, c_i), time_to_race in race_times.items()
        if time_to_race.total_seconds() > 0
    }
    next_date, r_i, c_i = min(coming_races, key=coming_races.get)
    print(
        f"Time to next race: {coming_races[(next_date, r_i, c_i)]}, date {date}, R{r_i} C{c_i}"
    )
    print(f"https://www.pmu.fr/turf/{date.strftime('%d%m%Y')}/R{r_i}/C{c_i}")

    courses_ = [
        course
        for reunion in programme["programme"]["reunions"]
        for course in reunion["courses"]
        if course["numReunion"] == r_i and course["numOrdre"] == c_i
    ]
    assert len(courses_) == 1
    course = courses_[0]

    course_race_datetime = dt.datetime.fromtimestamp(
        course["heureDepart"] / 1000,
        tz=dt.timezone(dt.timedelta(milliseconds=course["timezoneOffset"])),
    )

    participants_ = execute_get_query(
        url=get_pmu_api_url(url_name="PARTICIPANTS", date=date, r_i=r_i, c_i=c_i)
    )
    assert "participants" in participants_

    participants_ = participants_["participants"]
    participants = [
        {k: v for k, v in part.items() if not isinstance(v, dict)}
        for part in participants_
    ]
    course_incidents = course["incidents"] if "incidents" in course else []
    incident_nums = {
        num_part
        for incident in course_incidents
        for num_part in incident["numeroParticipants"]
    }

    combinaisons = execute_get_query(
        url=get_pmu_api_url(url_name="COMBINAISONS", date=date, r_i=r_i, c_i=c_i)
        + "?specialisation=INTERNET"
    )
    citations = execute_get_query(
        url=get_pmu_api_url(url_name="CITATIONS", date=date, r_i=r_i, c_i=c_i)
        + "?specialisation=INTERNET"
    )
    num_pmu_enjeu = get_num_pmu_enjeu(
        combinaisons=combinaisons, pari_type="E_SIMPLE_GAGNANT"
    )

    num_pmu_citation_enjeu = get_num_pmu_citation_enjeu(
        citations=citations, pari_type="E_SIMPLE_GAGNANT"
    )

    for part, part_ in zip(participants, participants_):
        # Other dict key found {'commentaireApresCourse',
        #  'dernierRapportDirect',
        #  'dernierRapportReference',
        #  'distanceChevalPrecedent',
        #  'gainsParticipant', # added here
        #  'robe'}
        if "gainsParticipant" in part_:
            part.update(part_["gainsParticipant"])
        part["n_reunion"] = r_i
        part["n_course"] = c_i
        part["date"] = date
        part["race_datetime"] = course_race_datetime
        part["in_incident"] = part["numPmu"] in incident_nums
        part["incident_type"] = (
            None
            if part["numPmu"] not in incident_nums
            else [
                incident["type"]
                for incident in course_incidents
                if part["numPmu"] in incident["numeroParticipants"]
            ][0]
        )
        part["totalEnjeu"] = (
            None
            if num_pmu_citation_enjeu is None
            else num_pmu_citation_enjeu.get(part["numPmu"], None)
        )

    race_df = pd.DataFrame.from_records(participants)

    race_df = convert_queried_data_to_race_horse_df(
        queried_race_horse_df=race_df, historical_race_horse_df=historical_race_horse_df
    )

    race_df = features.append_features(
        race_horse_df=race_df, historical_race_horse_df=historical_race_horse_df
    )
    race_df = race_df[race_df["statut"] != "NON_PARTANT"]

    race_df["horse_place"] = np.nan

    x_race, y_race, odds_race = import_data.extract_x_y_odds(
        race_df=race_df,
        y_format="first_position",
        ignore_y=True,
    )
    n_horses = x_race.shape[0]

    model = utils_model.create_model(n_horses=n_horses, y_format="probabilities")
    y_hat_race = model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

    expected_return_race = y_hat_race * odds_race * (1 - track_take)

    race_df["y_hat"] = y_hat_race
    race_df["expected_return"] = expected_return_race

    bettings = wagering_stategy.race_betting_best_expected_return(
        x_race=x_race,
        odds_race=odds_race,
        track_take=track_take,
        capital_fraction=capital_fraction,
    )

    race_df["betting"] = bettings

    df=race_df[
            [
                "horse_number",
                "horse_name",
                "y_hat",
                "totalEnjeu",
                "odds",
                "expected_return",
                "betting",
            ]
        ].set_index("horse_number")

    print(tabulate(df, headers='keys', tablefmt='psql'))
    print()


if __name__ == "__main__":
    while True:
        suggest_betting_on_next_race()
