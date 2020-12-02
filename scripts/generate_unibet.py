import datetime as dt
import json
import logging
import os
import re
from typing import List
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
from utils.logger import setup_logger

UNIBET_DATA_PATH = "./data/Unibet"

logger = setup_logger(__name__)


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


def _get_or_create_parent(
    name: str, is_born_male: bool, db_session: SQLAlchemySession
) -> Optional[Horse]:

    if not name:
        logger.warning(
            "Could not find %s name with name: %s",
            ("father" if is_born_male else "mother"),
            name,
        )
        return None
    potential_parents = (
        db_session.query(Horse)
        .filter(Horse.name == name, Horse.is_born_male.is_(is_born_male))
        .all()
    )
    if not potential_parents:
        parent = Horse(name=name, is_born_male=is_born_male)
        db_session.add(parent)
        db_session.commit()
        return parent
    if len(potential_parents) == 1:
        return potential_parents[0]

    assert len(potential_parents) > 1
    logger.warning(
        "Too many %s found for name: %s!",
        ("fathers" if is_born_male else "mothers"),
        name,
    )
    return None


def _process_horse(
    name: Optional[str],
    name_country: str,
    is_born_male: Optional[bool],
    parent_names: str,
    birth_year: Optional[int],
    db_session: SQLAlchemySession,
) -> Optional[Horse]:

    if not name:
        name = re.sub(r"\s\(.*\)", "", name_country)
    name = name.upper()
    name = name.strip()
    assert name

    country_code = re.sub(r"\)", "", re.sub(r".*\(", "", name_country))
    assert country_code

    current_horses = db_session.query(Horse).filter(Horse.name == name).all()
    # TODO create parents if necessary
    if birth_year:
        current_horses_with_same_birth_year = [
            horse for horse in current_horses if horse.birth_year == birth_year
        ]
        if len(current_horses_with_same_birth_year) == 1:
            horse = current_horses_with_same_birth_year[0]
            assert horse.father_id
            assert horse.mother_id
            return horse

    if parent_names:
        current_horses_with_origins = [
            horse
            for horse in current_horses
            if horse.first_found_origins == parent_names
        ]

        if len(current_horses_with_origins) == 1:
            horse = current_horses_with_origins[0]
            assert horse.father_id
            assert horse.mother_id
            return horse

    father_mother_names: List[str] = []
    if parent_names:
        father_mother_names = re.split("[-/]", parent_names)
        if len(father_mother_names) != 2:
            father_mother_names = re.split("\set\s", parent_names)
        if len(father_mother_names) != 2:
            father_mother_names = parent_names.strip().split()

    if len(father_mother_names) != 2:
        logger.warning(
            "Could not find father mother names in origins: %s", parent_names
        )

    father, mother = None, None
    if len(father_mother_names) == 2:
        father_name, mother_name = father_mother_names
        father_name = father_name.upper().strip()
        mother_name = mother_name.upper().strip()

        father = _get_or_create_parent(
            name=father_name, is_born_male=True, db_session=db_session
        )
        mother = _get_or_create_parent(
            name=mother_name, is_born_male=False, db_session=db_session
        )

    horse = Horse.upsert(
        name=name,
        is_born_male=is_born_male,
        country_code=country_code,
        father=father,
        mother=mother,
        origins=parent_names,
        birth_year=birth_year,
        db_session=db_session,
    )

    return horse


def _process_runner(
    runner_dict: dict,
    runner_stats: dict,
    race: Race,
    current_race_dict: dict,
    db_session: SQLAlchemySession,
) -> None:
    runner_stats_info: Optional[dict] = runner_stats.get("fiche", {}).get(
        "info_generales"
    )
    # We don't use last performances info in runner_stats

    if race.type is None and runner_stats_info:
        race.type = runner_stats_info["discipline"]
        db_session.commit()

    owner_name = runner_stats_info["proprietaire"] if runner_stats_info else None
    owner_name = (
        runner_dict["details"]["owner"]
        if runner_dict["details"] and owner_name is None
        else owner_name
    )

    owner = Owner.upsert(name=owner_name, db_session=db_session)
    trainer_name = runner_stats_info["entraineur"] if runner_stats_info else None
    trainer_name = (
        runner_dict["details"]["trainer"]
        if runner_dict["details"] and trainer_name is None
        else trainer_name
    )

    trainer = Trainer.upsert(name=trainer_name, db_session=db_session)

    jockey_name = (
        runner_stats_info["jockey"] if runner_stats_info else runner_dict["jockey"]
    )
    jockey = Jockey.upsert(name=jockey_name, db_session=db_session)

    horse_name = runner_stats_info["nom"] if runner_stats_info else None
    horse_name = runner_dict["name"] if not horse_name else None

    horse_sex = runner_stats_info.get("sex") if runner_stats_info else None
    horse_sex = (
        horse_sex
        if horse_sex
        else (runner_dict["details"].get("sex") if runner_dict.get("details") else None)
    )
    horse_sex = UnibetHorseSex(horse_sex)
    is_born_male = horse_sex.is_born_male

    parent_names = runner_stats_info.get("parents") if runner_stats_info else None
    parent_names = (
        parent_names
        if parent_names
        else (
            runner_dict["details"].get("origins")
            if runner_dict.get("details")
            else None
        )
    )
    if parent_names:
        parent_names = parent_names.strip()
    birth_year = runner_stats_info.get("age") if runner_stats_info else None
    horse = _process_horse(
        name=horse_name,
        name_country=runner_dict["name"],
        is_born_male=is_born_male,
        parent_names=parent_names,
        birth_year=birth_year,
        db_session=db_session,
    )

    coat = runner_stats_info.get("robe") if runner_stats_info else None
    coat = (
        runner_dict["details"].get("coat")
        if runner_dict.get("details") and coat is None
        else coat
    )

    blinkers = runner_stats_info.get("oeillere") if runner_stats_info else None
    blinkers = runner_dict["blinkers"] if blinkers is None else blinkers

    stakes = runner_stats_info.get("gain") if runner_stats_info else None
    stakes = (
        runner_dict["details"].get("stakes")
        if runner_dict.get("details") and stakes is None
        else stakes
    )

    music = runner_stats_info.get("musique") if runner_stats_info else None
    music = (
        runner_dict["details"].get("musique")
        if runner_dict.get("details") and music is None
        else music
    )

    kilometer_record_sec = (
        convert_duration_in_sec(runner_stats_info.get("record"))
        if runner_stats_info
        else None
    )

    shoes = runner_stats_info.get("ferrure") if runner_stats_info else None
    shoes = (
        runner_dict["details"].get("shoes")
        if runner_dict.get("details") and shoes is None
        else shoes
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
        rope_n=runner_stats_info.get("corde") if runner_stats_info else None,
        draw=runner_dict["draw"],
        blinkers=blinkers,
        shoes=shoes,
        silk=runner_dict["silk"],
        stakes=stakes,
        music=music,
        sex=horse_sex,
        age=runner_dict["details"].get("age") if runner_dict.get("details") else None,
        coat=coat,
        origins=parent_names,
        comment=runner_dict["details"].get("comment")
        if runner_dict.get("details")
        else None,
        kilometer_record_sec=kilometer_record_sec,
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
                logger.warning("Could not find folder for date: %s", date.isoformat())
                continue
            day_folder_path = os.path.join(UNIBET_DATA_PATH, date.isoformat())
            if "programme.json" not in os.listdir(day_folder_path):
                logger.warning(
                    "Could not find programme.json for date: %s", date.isoformat()
                )
                continue

            with open(os.path.join(day_folder_path, "programme.json"), "r") as fp:
                programme = json.load(fp=fp)
            if "data" not in programme:
                logger.warning("Can not import programme of %s", date.isoformat())
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
                            f"R{horse_show.unibet_n}_C{race_dict['rank']}.json",
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
                            runner_statistics_path = os.path.join(
                                day_folder_path,
                                f"R{horse_show.unibet_n}_C{race_dict['rank']}_"
                                f"RN{runner_dict['rank']}.json",
                            )
                            with open(runner_statistics_path, "r") as fp:
                                runner_stats = json.load(fp=fp)
                            _process_runner(
                                runner_dict=runner_dict,
                                runner_stats=runner_stats,
                                race=race,
                                current_race_dict=current_race_dict,
                                db_session=db_session,
                            )


if __name__ == "__main__":
    run()
