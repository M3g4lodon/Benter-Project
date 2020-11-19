import datetime as dt
import json
import os
from typing import Generator
from typing import Optional
from typing import Tuple

import sqlalchemy as sa
from tqdm import tqdm

from constants import UNIBET_MIN_DATE
from database.setup import create_sqlalchemy_session
from database.setup import SQLAlchemySession
from models.horse_show import HorseShow
from models.race import Race
from models.race_track import RaceTrack

UNIBET_DATA_PATH = "./data/Unibet"


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
    morning_simple_gagnant_odds = 5

    # "cote directe" or "rapport_final" on simple_gagnant
    final_simple_gagnant_odds = 6

    morning_simple_place_odds = 7
    final_simple_place_odds = 8
    unknown_probable_type3 = 9

    # "rapport_final" on deuzio
    final_deuzio_odds = 13


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
    race_dict: dict,
    complete_race_dict: dict,
    race_track: RaceTrack,
    horse_show: HorseShow,
    db_session: SQLAlchemySession,
) -> Race:
    current_race_dict = complete_race_dict
    if complete_race_dict.get("note") == "server error, no json":
        # Can not use complete_race
        current_race_dict = race_dict
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


def run():
    with create_sqlalchemy_session() as db_session:
        for date in tqdm(
            date_countdown_generator(
                start_date=UNIBET_MIN_DATE,
                end_date=dt.date.today() - dt.timedelta(days=1),
            ),
            total=(dt.date.today() - dt.timedelta(days=1) - UNIBET_MIN_DATE).days,
            unit="days",
        ):

            day_folder_path = os.path.join(UNIBET_DATA_PATH, date.isoformat())
            if "programme.json" in os.listdir(day_folder_path):
                with open(os.path.join(day_folder_path, "programme.json"), "r") as fp:
                    programme = json.load(fp=fp)
                if "data" not in programme:
                    print(f"Can not import programme of {date.isoformat()}")
                    continue

                horse_shows_dict = programme["data"]
                for horse_show_dict in horse_shows_dict:
                    race_track, horse_show = _process_horse_show(
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
                            _process_race(
                                race_dict=race_dict,
                                complete_race_dict=complete_race_dict,
                                race_track=race_track,
                                horse_show=horse_show,
                                db_session=db_session,
                            )


if __name__ == "__main__":
    run()
