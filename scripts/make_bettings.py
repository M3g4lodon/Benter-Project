import datetime as dt
import functools

import numpy as np
import pandas as pd
import pytz
from tabulate import tabulate

import wagering_stategies
from constants import PMU_BETTINGS
from constants import SOURCE_PMU
from utils import features
from utils import import_data
from utils.pmu_api_data import convert_queried_data_to_race_horse_df
from utils.pmu_api_data import get_pmu_api_url
from utils.pmu_api_data import get_race_horses_records
from utils.scrape import execute_get_query
from winning_horse_models import AbstractWinningModel
from winning_horse_models.logistic_regression import LogisticRegressionModel

TIMEZONE = "Europe/Paris"

code_pari = "E_SIMPLE_GAGNANT"
capital_fraction = 0.01
WAGERING_STRATEGY_COMPUTE = wagering_stategies.race_betting_best_expected_return
WINNING_MODEL = LogisticRegressionModel


@functools.lru_cache(maxsize=None)
def _variables_to_query_once():
    print("Setting up constants variables")
    historical_race_horse_df = import_data.load_featured_data(source="PMU")

    winning_model = WINNING_MODEL.load_model(trainable=False)
    print("Winning Horse model is loaded")
    track_take = [betting for betting in PMU_BETTINGS if betting.name == code_pari][0][
        1
    ]
    return historical_race_horse_df, track_take, winning_model


# TODO check we read course the same way as in generate_pmu
def get_race_df(
    date: dt.date,
    r_i: int,
    c_i: int,
    programme: dict,
    historical_race_horse_df: pd.DataFrame,
) -> pd.DataFrame:
    race_horses_records = get_race_horses_records(
        programme=programme, date=date, r_i=r_i, c_i=c_i, should_be_on_disk=False
    )
    race_df = pd.DataFrame.from_records(race_horses_records)

    race_df = convert_queried_data_to_race_horse_df(
        queried_race_horse_df=race_df, historical_race_horse_df=historical_race_horse_df
    )

    race_df = features.append_features(
        race_horse_df=race_df, historical_race_horse_df=historical_race_horse_df
    )
    race_df["duration_since_last_race"] = (
        pd.to_datetime(race_df["date"]).dt.date
        - pd.to_datetime(race_df["last_race_date"]).dt.date
    )

    race_df = race_df[race_df["statut"] != "NON_PARTANT"]

    race_df["horse_place"] = np.nan

    return race_df


def append_bettings_to_race_df(
    race_df: pd.DataFrame, winning_model: AbstractWinningModel, track_take: float
) -> pd.DataFrame:
    x_race, _ = import_data.extract_x_y(
        race_df=race_df,
        source=SOURCE_PMU,
        x_format="sequential_per_horse",
        y_format="first_position",
        ignore_y=True,
    )

    odds_race = race_df["odds"].values
    y_hat_race = winning_model.predict(x=np.expand_dims(x_race, axis=0))[0, :]

    expected_return_race = y_hat_race * odds_race * (1 - track_take)

    race_df["y_hat"] = y_hat_race
    race_df["expected_return"] = expected_return_race

    bettings = WAGERING_STRATEGY_COMPUTE(
        x_race=x_race,
        previous_stakes=race_df["totalEnjeu"],
        track_take=track_take,
        capital_fraction=capital_fraction,
        winning_model=winning_model,
    )

    race_df["betting"] = bettings
    return race_df


def suggest_betting_on_next_race():
    (historical_race_horse_df, track_take, winning_model) = _variables_to_query_once()

    today = dt.date.today()

    dt_now = pytz.timezone(TIMEZONE).localize(dt.datetime.now())

    programme = execute_get_query(url=get_pmu_api_url(url_name="PROGRAMME", date=today))

    race_times = {}
    for reunion in programme["programme"]["reunions"]:
        for course in reunion["courses"]:
            race_times[(today, course["numReunion"], course["numOrdre"])] = (
                dt.datetime.fromtimestamp(
                    course["heureDepart"] / 1000,
                    tz=dt.timezone(dt.timedelta(milliseconds=course["timezoneOffset"])),
                )
                - dt_now
            )

    coming_races = {
        (today, r_i, c_i): time_to_race
        for (today, r_i, c_i), time_to_race in race_times.items()
        if time_to_race.total_seconds() > 0
    }
    next_date, r_i, c_i = min(coming_races, key=coming_races.get)
    print(
        f"Time to next race: {coming_races[(next_date, r_i, c_i)]}, "
        f"date {today}, R{r_i} C{c_i}"
    )
    print(f"https://www.pmu.fr/turf/{today.strftime('%d%m%Y')}/R{r_i}/C{c_i}")

    race_df = get_race_df(
        date=today,
        r_i=r_i,
        c_i=c_i,
        historical_race_horse_df=historical_race_horse_df,
        programme=programme,
    )

    race_df = append_bettings_to_race_df(
        race_df=race_df, winning_model=winning_model, track_take=track_take
    )

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
