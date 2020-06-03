import datetime as dt
import functools

import numpy as np
import pandas as pd
import pytz
from tabulate import tabulate

import wagering_stategies
from constants import PMU_BETTINGS, SOURCE_PMU
from scripts.generate_pmu_data import (
    convert_queried_data_to_race_horse_df,
    get_num_pmu_enjeu_from_citations,
)
from scripts.scrape_pmu_data import get_pmu_api_url
from utils.scrape import execute_get_query
from utils import features
from utils import import_data
from winning_horse_models.logistic_regression import LogisticRegressionModel

TIMEZONE = "Europe/Paris"

code_pari = "E_SIMPLE_GAGNANT"
capital_fraction = 0.01


@functools.lru_cache(maxsize=None)
def _variables_to_query_once():
    print("Setting up constants variables")
    historical_race_horse_df = import_data.load_featured_data(source="PMU")

    winning_model = LogisticRegressionModel.load_model(trainable=False)
    print("Winning Horse model is loaded")
    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    return historical_race_horse_df, track_take, winning_model


def suggest_betting_on_next_race():
    (historical_race_horse_df, track_take, winning_model) = _variables_to_query_once()

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

    citations = execute_get_query(
        url=get_pmu_api_url(url_name="CITATIONS", date=date, r_i=r_i, c_i=c_i)
        + "?specialisation=INTERNET"
    )

    num_pmu_citation_enjeu = get_num_pmu_enjeu_from_citations(
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
        source=SOURCE_PMU,
        x_format="sequential_per_horse",
        y_format="first_position",
        ignore_y=True,
    )

    y_hat_race = winning_model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

    expected_return_race = y_hat_race * odds_race * (1 - track_take)

    race_df["y_hat"] = y_hat_race
    race_df["expected_return"] = expected_return_race

    bettings = wagering_stategies.race_betting_best_expected_return(
        x_race=x_race,
        odds_race=odds_race,
        track_take=track_take,
        capital_fraction=capital_fraction,
        winning_model=winning_model,
    )

    race_df["betting"] = bettings

    df = race_df[
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

    print(tabulate(df, headers="keys", tablefmt="psql"))
    print()


if __name__ == "__main__":
    while True:
        suggest_betting_on_next_race()
