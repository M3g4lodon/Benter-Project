import datetime as dt
import json
import logging
import os
import re
from typing import Optional
from typing import Tuple

from tqdm import tqdm

from constants import UNIBET_MIN_DATE
from constants import UnibetHorseSex
from constants import UnibetProbableType
from database.setup import create_sqlalchemy_session
from database.setup import SQLAlchemySession
from models.horse import Horse
from models.horse_show import HorseShow
from models.jockey import Jockey
from models.owner import Owner
from models.race import Race
from models.race_track import RaceTrack
from models.runner import Runner
from models.trainer import Trainer
from utils import convert_duration_in_sec
from utils import date_countdown_generator

UNIBET_DATA_PATH = "./data/Unibet"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _get_position(current_race_dict: dict, unibet_n: int) -> Optional[int]:
    if "results" not in current_race_dict:
        return None
    if "positions" not in current_race_dict["results"]:
        return None
    positions = current_race_dict["results"]["positions"]

    found_positions = [pos for pos in positions if unibet_n in pos["numbers"]]

    assert len(found_positions) <= 1

    if not found_positions:
        return None

    return found_positions[0]["position"]


def _process_race(
    current_race_dict: dict, horse_show: HorseShow, db_session: SQLAlchemySession
) -> Race:
    race_start_at = dt.datetime.fromtimestamp(current_race_dict["starttime"] / 1000)
    race_date = dt.date.fromisoformat(current_race_dict["date"])
    race_meeting_id = current_race_dict["meetingId"]
    assert race_start_at.date() == race_date
    assert race_meeting_id == horse_show.unibet_id
    race = Race.upsert(
        race_unibet_id=current_race_dict["zeturfId"],
        race_unibet_n=current_race_dict["rank"],
        race_name=current_race_dict["name"],
        race_start_at=race_start_at,
        race_date=race_date,
        race_type=current_race_dict["type"],
        race_conditions=current_race_dict["conditions"],
        race_stake=current_race_dict["stake"],
        race_arjel_level=current_race_dict["arjelLevel"],
        race_distance=current_race_dict["distance"],
        race_friendly_url=current_race_dict["friendlyUrl"],
        race_pronostic=current_race_dict["details"]["pronostic"],
        horse_show=horse_show,
        db_session=db_session,
    )

    return race


def _process_horse_show(
    horse_show_dict: dict, db_session: SQLAlchemySession
) -> Tuple[RaceTrack, HorseShow]:

    race_track = RaceTrack.upsert(
        race_track_name=horse_show_dict["place"],
        country_name=horse_show_dict["country"],
        db_session=db_session,
    )

    horse_show = HorseShow.upsert(
        horse_show_unibet_id=horse_show_dict["zeturfId"],
        horse_show_unibet_n=horse_show_dict["rank"],
        horse_show_datetime=dt.datetime.fromtimestamp(horse_show_dict["date"] / 1000),
        horse_show_ground=horse_show_dict["ground"],
        race_track=race_track,
        db_session=db_session,
    )

    return race_track, horse_show


def _process_horse(
    name_country: str, runner_dict: dict, race: Race, db_session: SQLAlchemySession
) -> Optional[Horse]:

    name = re.sub(r"\s\(.*\)", "", name_country)
    name = name.upper()
    country_code = re.sub(r"\)", "", re.sub(r".*\(", "", name_country))
    is_born_male = (
        UnibetHorseSex(runner_dict["details"].get("sex") or None)
        in [UnibetHorseSex.MALE, UnibetHorseSex.GELDING]
        if runner_dict.get("details")
        else None
    )
    assert name
    assert country_code

    current_horse_age = (
        runner_dict["details"].get("age") if runner_dict.get("details") else None
    )
    if current_horse_age:
        current_horse_age = int(current_horse_age)

    father_mother_names = (
        runner_dict["details"].get("origins") if runner_dict.get("details") else None
    )

    father, mother = Horse.upsert_father_mother(
        current_horse_age=current_horse_age,
        race=race,
        father_mother_names=father_mother_names,
        db_session=db_session,
    )

    horse = Horse.upsert(
        name=name,
        is_born_male=is_born_male,
        country_code=country_code,
        father=father,
        mother=mother,
        db_session=db_session,
    )

    return horse


def _process_runner(
    runner_dict: dict,
    race: Race,
    current_race_dict: dict,
    db_session: SQLAlchemySession,
) -> None:
    owner, trainer = None, None
    if runner_dict.get("details"):
        owner = Owner.upsert(
            name=runner_dict["details"]["owner"], db_session=db_session
        )
        trainer = Trainer.upsert(
            name=runner_dict["details"]["trainer"], db_session=db_session
        )
    jockey = Jockey.upsert(name=runner_dict["jockey"], db_session=db_session)
    horse = _process_horse(
        name_country=runner_dict["name"],
        runner_dict=runner_dict,
        race=race,
        db_session=db_session,
    )

    morning_odds, final_odds = None, None

    if current_race_dict.get("details") and current_race_dict["details"].get(
        "probables"
    ):
        morning_odds = current_race_dict["details"]["probables"][
            str(UnibetProbableType.MORNING_SIMPLE_GAGNANT_ODDS)
        ].get(str(runner_dict["rank"]))
        final_odds = current_race_dict["details"]["probables"][
            str(UnibetProbableType.FINAL_SIMPLE_GAGNANT_ODDS)
        ].get(str(runner_dict["rank"]))

    _ = Runner.upsert(
        unibet_id=runner_dict["zeturfId"],
        race=race,
        race_duration_sec=convert_duration_in_sec(
            time_str=runner_dict["details"].get("time")
        )
        if runner_dict.get("details")
        else None,
        weight=runner_dict["weight"],
        unibet_n=runner_dict["rank"],
        draw=runner_dict["draw"],
        blinkers=runner_dict["blinkers"],
        shoes=runner_dict["shoes"],
        silk=runner_dict["silk"],
        stakes=runner_dict["details"].get("stakes")
        if runner_dict.get("details")
        else None,
        music=runner_dict["details"].get("musique")
        if runner_dict.get("details")
        else None,
        sex=runner_dict["details"].get("sex") if runner_dict.get("details") else None,
        age=runner_dict["details"].get("age") if runner_dict.get("details") else None,
        coat=runner_dict["details"].get("coat") if runner_dict.get("details") else None,
        origins=runner_dict["details"].get("origins")
        if runner_dict.get("details")
        else None,
        comment=runner_dict["details"].get("comment")
        if runner_dict.get("details")
        else None,
        owner=owner,
        trainer=trainer,
        jockey=jockey,
        horse=horse,
        length=runner_dict["details"]["length"] if runner_dict.get("details") else None,
        position=_get_position(
            current_race_dict=current_race_dict, unibet_n=runner_dict["rank"]
        ),
        morning_odds=morning_odds,
        final_odds=final_odds,
        db_session=db_session,
    )


def run():  # pylint:disable=too-many-branches
    with create_sqlalchemy_session() as db_session:
        for date in tqdm(
            date_countdown_generator(
                start_date=UNIBET_MIN_DATE,
                end_date=dt.date.today() - dt.timedelta(days=1),
            ),
            total=(dt.date.today() - dt.timedelta(days=1) - UNIBET_MIN_DATE).days,
            unit="days",
        ):
            if not date.isoformat() in os.listdir(UNIBET_DATA_PATH):
                logger.info("Could not find folder for date: %s", date.isoformat())
                continue
            day_folder_path = os.path.join(UNIBET_DATA_PATH, date.isoformat())
            if "programme.json" not in os.listdir(day_folder_path):
                logger.info(
                    "Could not find programme.json for date: %s", date.isoformat()
                )
                continue

            with open(os.path.join(day_folder_path, "programme.json"), "r") as fp:
                programme = json.load(fp=fp)
            if "data" not in programme:
                logger.info("Can not import programme of %s", date.isoformat())
                continue

            horse_shows_dict = programme["data"]
            for horse_show_dict in horse_shows_dict:
                _, horse_show = _process_horse_show(
                    horse_show_dict=horse_show_dict, db_session=db_session
                )

                if horse_show_dict.get("races"):

                    for race_dict in horse_show_dict["races"]:
                        race_path = os.path.join(
                            day_folder_path,
                            f"R{horse_show.unibet_n}_C" f"{race_dict['rank']}.json",
                        )
                        with open(race_path, "r") as fp:
                            complete_race_dict = json.load(fp=fp)

                        current_race_dict = complete_race_dict
                        if complete_race_dict.get("note") == "server error, no json":
                            # Can not use complete_race
                            current_race_dict = race_dict
                        race = _process_race(
                            current_race_dict=current_race_dict,
                            horse_show=horse_show,
                            db_session=db_session,
                        )
                        for runner_dict in current_race_dict["runners"]:
                            _process_runner(
                                runner_dict=runner_dict,
                                race=race,
                                current_race_dict=current_race_dict,
                                db_session=db_session,
                            )


if __name__ == "__main__":
    run()
