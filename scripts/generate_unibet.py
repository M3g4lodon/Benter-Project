import datetime as dt
import json
import logging
import os
import re
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import sqlalchemy as sa
from tqdm import tqdm

from constants import UNIBET_MIN_DATE
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

UNIBET_DATA_PATH = "./data/Unibet"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class UnibetBetTRateType:
    """As of Nov 27th 2020, found in Unibet JS code"""

    # Mise de base: 1€<br>Trouvez le 1er cheval de l’arrivée.
    SIMPLE_WINNER = 1

    # "Mise de base: 1€<br>Trouvez 1 cheval parmi les 3 premiers sur les courses de 8
    # chevaux et plus OU 1 cheval parmi les 2 premiers sur les courses de 4 à 7 chevaux.
    SIMPLE_PLACED = 2

    # Mise de base: 1€<br>Trouvez les 2 premiers chevaux dans l’ordre ou le désordre.
    JUMELE_WINNER = 3

    # Mise de base: 1€<br>Trouvez les 2 premiers chevaux dans l’ordre.
    JUMELE_ORDER = 4

    # Mise de base: 1€<br>Trouvez 2 chevaux parmi les 3 premiers à l’arrivée.
    JUMELE_PLACED = 5

    # Mise de base: 1€<br>Trouvez les 3 premiers chevaux.
    TRIO = 6

    # Mise de base: 1€<br>Trouvez le cheval qui arrive à la 4ème place.
    LEBOULET = 7

    # Mise de base: 0.50€<br>Trouvez les 4 premiers chevaux, dans l’ordre ou le
    # désordre.
    QUADRI = 8

    # Mise de base: 1€<br>Trouvez les 3 premiers chevaux dans l'ordre.
    TRIO_ORDER = 11

    # Mise de base: 0.50€<br>Trouvez les 5 premiers chevaux, dans l’ordre ou le
    # désordre.
    FIVE_OVER_FIVE = 12

    # Mise de base: 1€<br>Trouvez 2 chevaux parmi les 4 premiers à l’arrivée.
    TWO_OVER_FOUR = 13

    # Mise de base: 1€<br>Trouvez le cheval qui arrive à la 2ème place
    DEUZIO = 29

    # Mise de base : 1.50€<br>Mixez dans un même betslip: un Quadri + des Trios + des
    # Jumelés Gagnants.
    MIX_FOUR = 15

    # Mise de base: 2€<br>Mixez dans un même betslip: un 5 sur 5 + des Quadris + des
    # Trios.
    MIX_FIVE = 16

    # Mise de base: 3€<br>Mixez dans un même betslip: un Simple Gagnant + un Simple
    # Placé + un Deuzio + un Boulet
    MIX_S = 31

    # Mise de base: 1€<br>Trouvez les 2 premiers chevaux à l’arrivée dans l’ordre ou
    # le désordre.
    JUMELE = 32

    # Mise de base: 1€<br>Trouvez le 1er cheval à l’arrivée gagnant ou placé.
    SIMPLE = 33


class UnibetProbableType:
    # "matin cote" on simple_gagnant
    MORNING_SIMPLE_GAGNANT_ODDS = 5

    # "cote directe" or "rapport_final" on simple_gagnant
    FINAL_SIMPLE_GAGNANT_ODDS = 6

    PROBABLES_1 = 7
    PROBABLES_2 = 8
    PROBABLES_3 = 9

    # "rapport_final" on deuzio
    FINAL_DEUZIO_ODDS = 13


def date_countdown_generator(
    start_date: dt.date, end_date: Optional[dt.date]
) -> Generator[dt.date, None, None]:
    end_date = end_date or dt.date.today()
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += dt.timedelta(days=1)


def _get_or_create_race(  # pylint:disable=too-many-arguments, too-many-locals
    race_unibet_id: int,
    race_unibet_n: int,
    race_name: str,
    race_start_at: dt.datetime,
    race_date: dt.date,
    race_type: str,
    race_conditions: str,
    race_stake: int,
    race_arjel_level: str,
    race_distance: int,
    race_friendly_url: str,
    race_pronostic: str,
    horse_show: HorseShow,
    race_meeting_id: int,
    db_session: SQLAlchemySession,
) -> Race:

    assert race_start_at.date() == race_date
    assert race_meeting_id == horse_show.unibet_id

    found_race = (
        db_session.query(Race).filter(Race.unibet_id == race_unibet_id).one_or_none()
    )
    if found_race is not None:
        assert found_race.unibet_id == race_unibet_id
        assert found_race.name == race_name
        assert found_race.start_at == race_start_at
        assert found_race.date == race_date
        assert found_race.unibet_n == race_unibet_n
        assert found_race.type == race_type
        assert found_race.conditions == race_conditions
        assert found_race.stake == race_stake
        assert found_race.arjel_level == race_arjel_level
        assert found_race.distance == race_distance
        assert found_race.friendly_URL == race_friendly_url
        assert found_race.pronostic == race_pronostic
        assert found_race.horse_show_id == horse_show.id
        assert found_race.id
        return found_race

    race = Race(
        unibet_id=race_unibet_id,
        name=race_name,
        start_at=race_start_at,
        date=race_date,
        unibet_n=race_unibet_n,
        type=race_type,
        conditions=race_conditions,
        stake=race_stake,
        arjel_level=race_arjel_level,
        distance=race_distance,
        friendly_URL=race_friendly_url,
        pronostic=race_pronostic,
        horse_show_id=horse_show.id,
    )
    db_session.add(race)
    db_session.commit()

    assert race.id
    return race


def _process_race(
    current_race_dict: dict, horse_show: HorseShow, db_session: SQLAlchemySession
) -> Race:

    race = _get_or_create_race(
        race_unibet_id=current_race_dict["zeturfId"],
        race_unibet_n=current_race_dict["rank"],
        race_name=current_race_dict["name"],
        race_start_at=dt.datetime.fromtimestamp(current_race_dict["starttime"] / 1000),
        race_date=dt.date.fromisoformat(current_race_dict["date"]),
        race_type=current_race_dict["type"],
        race_conditions=current_race_dict["conditions"],
        race_stake=current_race_dict["stake"],
        race_arjel_level=current_race_dict["arjelLevel"],
        race_distance=current_race_dict["distance"],
        race_friendly_url=current_race_dict["friendlyUrl"],
        race_pronostic=current_race_dict["details"]["pronostic"],
        horse_show=horse_show,
        race_meeting_id=current_race_dict["meetingId"],
        db_session=db_session,
    )

    return race


def _create_or_get_race_track(
    race_track_name: str, country_name: str, db_session: SQLAlchemySession
) -> RaceTrack:
    found_race_track = (
        db_session.query(RaceTrack)
        .filter(
            RaceTrack.race_track_name == race_track_name,
            RaceTrack.country_name == country_name,
        )
        .one_or_none()
    )
    if found_race_track is not None:
        assert found_race_track.id
        return found_race_track

    race_track = RaceTrack(race_track_name=race_track_name, country_name=country_name)
    db_session.add(race_track)
    db_session.commit()

    assert race_track.id
    return race_track


def _get_or_create_horse_show(
    horse_show_unibet_id: int,
    horse_show_ground: str,
    horse_show_unibet_n: int,
    horse_show_datetime: dt.datetime,
    race_track: RaceTrack,
    db_session: SQLAlchemySession,
):
    found_horse_show = (
        db_session.query(HorseShow)
        .filter(
            HorseShow.unibet_n == horse_show_unibet_n,
            sa.func.date(HorseShow.datetime) == horse_show_datetime.date(),
        )
        .one_or_none()
    )

    if found_horse_show is not None:
        assert found_horse_show.unibet_id == horse_show_unibet_id
        assert found_horse_show.ground == horse_show_ground
        assert found_horse_show.race_track_id == race_track.id
        assert found_horse_show.id
        return found_horse_show

    horse_show = HorseShow(
        unibet_id=horse_show_unibet_id,
        datetime=horse_show_datetime,
        unibet_n=horse_show_unibet_n,
        ground=horse_show_ground,
        race_track_id=race_track.id,
    )
    db_session.add(horse_show)
    db_session.commit()
    assert horse_show.id
    return horse_show


def _process_horse_show(
    horse_show_dict: dict, db_session: SQLAlchemySession
) -> Tuple[RaceTrack, HorseShow]:

    race_track = _create_or_get_race_track(
        race_track_name=horse_show_dict["place"],
        country_name=horse_show_dict["country"],
        db_session=db_session,
    )

    horse_show = _get_or_create_horse_show(
        horse_show_unibet_id=horse_show_dict["zeturfId"],
        horse_show_unibet_n=horse_show_dict["rank"],
        horse_show_datetime=dt.datetime.fromtimestamp(horse_show_dict["date"] / 1000),
        horse_show_ground=horse_show_dict["ground"],
        race_track=race_track,
        db_session=db_session,
    )

    return race_track, horse_show


def _get_or_create_named_model(
    name: str,
    model_class: Union[Type[Owner], Type[Trainer], Type[Jockey]],
    db_session: SQLAlchemySession,
) -> Optional[Union[Owner, Trainer, Jockey]]:
    if not name and not name.strip():
        return None
    found_instance = (
        db_session.query(model_class).filter(model_class.name == name).one_or_none()
    )

    if found_instance is not None:
        assert found_instance.id
        return found_instance

    instance = model_class(name=name)
    db_session.add(instance)
    db_session.commit()
    assert instance.id
    return instance


def _get_or_create_runner(  # pylint:disable=too-many-arguments,too-many-locals,too-many-statements
    unibet_id: int,
    race: Race,
    weight: int,
    unibet_n: int,
    draw: int,
    blinkers: str,
    shoes: str,
    silk: str,
    stakes: int,
    music: str,
    sex: Optional[str],
    age: Optional[str],
    coat: str,
    origins: str,
    comment: Optional[str],
    length: str,
    owner: Optional[Owner],
    trainer: Optional[Trainer],
    jockey: Optional[Jockey],
    horse: Optional[Horse],
    position: Optional[int],
    race_duration_sec: Optional[float],
    morning_odds: Optional[float],
    final_odds: Optional[float],
    db_session: SQLAlchemySession,
) -> Runner:
    assert race

    found_runner = (
        db_session.query(Runner).filter(Runner.unibet_id == unibet_id).one_or_none()
    )
    age_: Optional[int] = None
    if age == "":
        age_ = None
    if age is not None and int(age) > 100:
        age_ = None
    elif age is not None:
        age_ = int(age)
    del age

    if sex == "":
        sex = None

    if comment == "":
        comment = None

    if found_runner is not None and found_runner.age != age_:
        found_runner.age = age_
        db_session.commit()

    if found_runner is not None and found_runner.sex != sex:
        found_runner.sex = sex
        db_session.commit()

    if found_runner is not None and found_runner.comment != comment:
        found_runner.comment = comment
        db_session.commit()

    if found_runner is not None:
        assert found_runner.race_id == race.id
        assert found_runner.weight == weight
        assert found_runner.unibet_n == unibet_n
        assert found_runner.draw == draw
        assert found_runner.blinkers == blinkers
        assert found_runner.shoes == shoes
        assert found_runner.silk == silk
        assert found_runner.stakes == stakes
        assert found_runner.music == music
        assert found_runner.sex == sex
        assert found_runner.age == age_
        assert found_runner.coat == coat
        assert found_runner.origins == origins
        assert found_runner.comment == comment
        assert found_runner.length == length
        assert found_runner.owner_id == (owner.id if owner else None)
        assert found_runner.trainer_id == (trainer.id if trainer else None)
        assert found_runner.jockey_id == (jockey.id if jockey else None)
        assert found_runner.horse_id == (horse.id if horse else None)
        assert found_runner.position == (str(position) if position else None)
        assert found_runner.race_duration_sec == race_duration_sec
        assert found_runner.morning_odds == morning_odds
        assert found_runner.final_odds == final_odds
        assert found_runner.id
        return found_runner

    runner = Runner(
        unibet_id=unibet_id,
        race_id=race.id,
        weight=weight,
        unibet_n=unibet_n,
        draw=draw,
        blinkers=blinkers,
        shoes=shoes,
        silk=silk,
        stakes=stakes,
        music=music,
        sex=sex,
        age=age_,
        coat=coat,
        origins=origins,
        comment=comment,
        length=length,
        owner_id=owner.id if owner else None,
        trainer_id=trainer.id if trainer else None,
        jockey_id=jockey.id if jockey else None,
        horse_id=horse.id if horse else None,
        position=position,
        race_duration_sec=race_duration_sec,
        morning_odds=morning_odds,
        final_odds=final_odds,
    )
    db_session.add(runner)
    db_session.commit()
    assert runner.id
    return runner


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


def _convert_time_in_sec(time_str: Optional[str]) -> Optional[float]:
    if not time_str:
        return None

    matches = re.match(r"(\d{1,2})'(\d{2})''(\d{2})", time_str)
    if not matches:
        return None

    n_min, n_sec, n_cs = matches.groups()
    return 60 * int(n_min) + int(n_sec) + 0.01 * int(n_cs)


def _get_or_create_horse(name_country: str, db_session: SQLAlchemySession) -> Horse:

    name = re.sub(r"\s\(.*\)", "", name_country)
    country_code = re.sub(r"\)", "", re.sub(r".*\(", "", name_country))
    assert name
    assert country_code
    found_horse = (
        db_session.query(Horse)
        .filter(Horse.name == name, Horse.country_code == country_code)
        .one_or_none()
    )
    # TODO find by origins

    if found_horse is not None:
        assert found_horse.id
        return found_horse

    horse = Horse(name=name, country_code=country_code)
    db_session.add(horse)
    db_session.commit()
    assert horse.id
    return horse


def _process_runner(
    runner_dict: dict,
    race: Race,
    current_race_dict: dict,
    db_session: SQLAlchemySession,
) -> None:
    owner, trainer, jockey, horse = None, None, None, None
    if runner_dict["details"]:
        owner = _get_or_create_named_model(
            model_class=Owner,
            name=runner_dict["details"]["owner"],
            db_session=db_session,
        )
        trainer = _get_or_create_named_model(
            model_class=Trainer,
            name=runner_dict["details"]["trainer"],
            db_session=db_session,
        )
    jockey = _get_or_create_named_model(
        model_class=Jockey, name=runner_dict["jockey"], db_session=db_session
    )
    horse = _get_or_create_horse(
        name_country=runner_dict["name"], db_session=db_session
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

    _ = _get_or_create_runner(
        unibet_id=runner_dict["zeturfId"],
        race=race,
        race_duration_sec=_convert_time_in_sec(
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
