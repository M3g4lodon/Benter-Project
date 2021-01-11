import json
import os
import random
from typing import Sequence

from tabulate import tabulate

from constants import UNIBET_DATA_PATH
from database.setup import create_sqlalchemy_session
from models import Race
from models import Runner
from utils import setup_logger

logger = setup_logger(name=__file__)


def describe_horse_runners(horse_id: int) -> None:
    with create_sqlalchemy_session() as db_session:
        runners = db_session.query(Runner).filter(Runner.horse_id == horse_id).all()
        headers = [
            "id",
            "date",
            "race_id",
            "name",
            "age",
            "sex",
            "stakes",
            "origins",
            "music",
        ]
        to_print = []
        for runner in runners:
            to_print.append(
                [
                    runner.id,
                    runner.date,
                    runner.race_id,
                    runner.horse.name,
                    runner.age,
                    runner.sex.name,
                    runner.stakes,
                    runner.origins,
                    runner.music,
                ]
            )

        print(tabulate(to_print, headers=headers))

        print()


def describe_sample_horses_runners(horse_ids: Sequence[int]) -> None:
    for horse_id in random.sample(horse_ids, 10):
        describe_horse_runners(horse_id=horse_id)


def describe_runner_from_queries(runner: Runner) -> None:
    date, horse_show_rank, race_rank, runner_rank = runner.unibet_code
    if not date.isoformat() in os.listdir(UNIBET_DATA_PATH):
        logger.warning("Could not find folder for date: %s", date.isoformat())

    day_folder_path = os.path.join(UNIBET_DATA_PATH, date.isoformat())
    if "programme.json" not in os.listdir(day_folder_path):
        logger.warning("Could not find programme.json for date: %s", date.isoformat())

    with open(os.path.join(day_folder_path, "programme.json"), "r") as fp:
        programme = json.load(fp=fp)

    horse_shows_dict = programme["data"]
    for horse_show_dict in horse_shows_dict:
        if horse_show_dict["rank"] == horse_show_rank:
            for race_dict in horse_show_dict["races"]:
                if race_dict["rank"] == race_rank:
                    for runner_dict in race_dict["runners"]:
                        if runner_dict["rank"] == runner_rank:
                            print("from programme.json", runner_dict)
    print()

    race_path = os.path.join(day_folder_path, f"R{horse_show_rank}_C{race_rank}.json")
    with open(race_path, "r") as fp:
        complete_race_dict = json.load(fp=fp)

    for runner_dict in complete_race_dict["runners"]:
        if runner_dict["rank"] == runner_rank:
            print("from race.json", runner_dict)

    print()

    runner_statistics_path = os.path.join(
        day_folder_path, f"R{horse_show_rank}_" f"C{race_rank}_" f"RN{runner_rank}.json"
    )
    with open(runner_statistics_path, "r") as fp:
        runner_stats = json.load(fp=fp)
    print(
        "from runner.json", runner_stats.get("fiche", {}).get("infos_generales", None)
    )


def describe_race(race_id: int) -> None:
    with create_sqlalchemy_session() as db_session:
        race = db_session.query(Race).filter(Race.id == race_id).one()
        print(
            tabulate(
                [[race.id, race.date.isoformat(), race.unibet_code, race.type.value]],
                headers=["id", "date", "unibet_code", "type"],
                tablefmt="grid",
            )
        )
        print("conditions", race.conditions)
        print()
        headers = [
            "id",
            "date",
            "race_id",
            "name",
            "age",
            "sex",
            "stakes",
            "hist_stakes",
            "origins",
        ]
        to_print = []
        for runner in race.runners:
            to_print.append(
                [
                    runner.id,
                    runner.date,
                    runner.race_id,
                    runner.horse.name,
                    runner.age,
                    runner.sex.name,
                    runner.stakes,
                    runner.historical_stakes,
                    runner.origins,
                ]
            )

        print(tabulate(to_print, headers=headers, tablefmt="grid"))
