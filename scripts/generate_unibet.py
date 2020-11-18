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
from models.race_track import RaceTrack

UNIBET_DATA_PATH = "./data/Unibet"

# Found in JS unibet code
class UnibetBetTRateType:
    SIMPLE_WINNER = 1  # Mise de base: 1€<br>Trouvez le 1er cheval de l’arrivée.
    SIMPLE_PLACED = (
        2
    )  # "Mise de base: 1€<br>Trouvez 1 cheval parmi les 3 premiers sur les courses de 8 chevaux et plus OU 1 cheval parmi les 2 premiers sur les courses de 4 à 7 chevaux.
    JUMELE_WINNER = (
        3
    )  # Mise de base: 1€<br>Trouvez les 2 premiers chevaux dans l’ordre ou le désordre.
    JUMELE_ORDER = 4  # Mise de base: 1€<br>Trouvez les 2 premiers chevaux dans l’ordre.
    JUMELE_PLACED = (
        5
    )  # Mise de base: 1€<br>Trouvez 2 chevaux parmi les 3 premiers à l’arrivée.
    TRIO = 6  # Mise de base: 1€<br>Trouvez les 3 premiers chevaux.
    LEBOULET = 7  # Mise de base: 1€<br>Trouvez le cheval qui arrive à la 4ème place.
    QUADRI = (
        8
    )  # Mise de base: 0.50€<br>Trouvez les 4 premiers chevaux, dans l’ordre ou le désordre.
    TRIO_ORDER = 11  # Mise de base: 1€<br>Trouvez les 3 premiers chevaux dans l'ordre.
    FIVE_OVER_FIVE = (
        12
    )  # Mise de base: 0.50€<br>Trouvez les 5 premiers chevaux, dans l’ordre ou le désordre.
    TWO_OVER_FOUR = (
        13
    )  # Mise de base: 1€<br>Trouvez 2 chevaux parmi les 4 premiers à l’arrivée.
    DEUZIO = 29  # Mise de base: 1€<br>Trouvez le cheval qui arrive à la 2ème place
    MIX_FOUR = (
        15
    )  # Mise de base : 1.50€<br>Mixez dans un même betslip: un Quadri + des Trios + des Jumelés Gagnants.
    MIX_FIVE = (
        16
    )  # Mise de base: 2€<br>Mixez dans un même betslip: un 5 sur 5 + des Quadris + des Trios.
    MIX_S = (
        31
    )  # Mise de base: 3€<br>Mixez dans un même betslip: un Simple Gagnant + un Simple Placé + un Deuzio + un Boulet
    JUMELE = (
        32
    )  # Mise de base: 1€<br>Trouvez les 2 premiers chevaux à l’arrivée dans l’ordre ou le désordre.
    SIMPLE = (
        33
    )  # Mise de base: 1€<br>Trouvez le 1er cheval à l’arrivée gagnant ou placé.


class UnibetProbableType:
    morning_simple_gagnant_odds = 5  # "matin cote" on simple_gagnant
    final_simple_gagnant_odds = 6  # "cote directe" or "rapport_final" on simple_gagnant
    morning_simple_place_odds = 7  # "matin cote" on simple_place, not sure
    final_simple_place_odds = (
        8
    )  # "cote directe" or "rapport_final" on simple_place not sure
    unknown_probable_type3 = 9
    final_deuzio_odds = 13  # "rapport_final" on deuzio


def date_countdown_generator(
    start_date: dt.date, end_date: Optional[dt.date]
) -> Generator[dt.date, None, None]:
    end_date = end_date or dt.date.today()
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += dt.timedelta(days=1)


def process_race(
    race: dict, complete_race: dict, db_session: SQLAlchemySession
) -> None:
    if complete_race.get("note") == "server error, no json":
        # Can not use complete_race
        return None
    race_name = race["name"]
    assert complete_race["name"] == race["name"]

    race_datetime = dt.datetime.fromtimestamp(complete_race["starttime"] / 1000)
    if race_datetime.year < 2020:
        return None

    if complete_race["betCounter"] is not None:
        print(complete_race["betCounter"], complete_race["friendlyUrl"])

    pronostic = complete_race["details"]["pronostic"]


def process_horse_show(
    horse_show_dict: dict, db_session: SQLAlchemySession
) -> Tuple[RaceTrack, HorseShow]:
    horse_show_datetime = dt.datetime.fromtimestamp(horse_show_dict["date"] / 1000)
    unibet_n_horse_show = horse_show_dict["rank"]
    horse_show_unibet_id = horse_show_dict["zeturfId"]
    horse_show_ground = horse_show_dict["ground"]
    race_track_name = horse_show_dict["place"]
    country_name = horse_show_dict["country"]

    found_race_track = (
        db_session.query(RaceTrack)
        .filter(
            RaceTrack.race_track_name == race_track_name,
            RaceTrack.country_name == country_name,
        )
        .one_or_none()
    )
    if found_race_track is None:
        race_track = RaceTrack(
            race_track_name=race_track_name, country_name=country_name
        )
        db_session.add(race_track)
        db_session.commit()
        assert race_track.id
    else:
        assert found_race_track.id
        race_track = found_race_track

    found_horse_show = (
        db_session.query(HorseShow)
        .filter(
            HorseShow.unibet_n == unibet_n_horse_show,
            sa.func.date(HorseShow.datetime) == horse_show_datetime.date(),
        )
        .one_or_none()
    )
    if found_horse_show is None:
        horse_show = HorseShow(
            unibet_id=horse_show_unibet_id,
            datetime=horse_show_datetime,
            unibet_n=unibet_n_horse_show,
            ground=horse_show_ground,
            race_track_id=race_track.id,
        )
        db_session.add(horse_show)
        db_session.commit()
        assert horse_show.id
    else:
        assert found_horse_show.unibet_id == horse_show_unibet_id
        assert found_horse_show.ground == horse_show_ground
        assert found_horse_show.race_track_id == race_track.id
        assert found_horse_show.id
        horse_show = found_horse_show

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

                horse_shows = programme["data"]
                for horse_show_dict in horse_shows:
                    race_track, horse_show = process_horse_show(
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
                            process_race(
                                race=race_dict,
                                complete_race=complete_race_dict,
                                db_session=db_session,
                            )


if __name__ == "__main__":
    run()
