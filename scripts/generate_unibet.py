import datetime as dt
import json
import os
import re
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple

import Levenshtein as lev
import tqdm

from constants import DATA_DIR
from constants import UNIBET_DATA_PATH
from constants import UNIBET_MIN_DATE
from constants import UnibetBlinkers
from constants import UnibetCoat
from constants import UnibetHorseSex
from constants import UnibetHorseShowGround
from constants import UnibetProbableType
from constants import UnibetRaceType
from constants import UnibetShoes
from database.setup import create_sqlalchemy_session
from database.setup import SQLAlchemySession
from models import Entity
from models.horse import Horse
from models.horse_show import HorseShow
from models.race import Race
from models.race_track import RaceTrack
from models.runner import Runner
from utils import convert_duration_in_sec
from utils import date_countdown_generator
from utils import unibet_coat_parser
from utils.logger import setup_logger
from utils.music import is_matching_with_max_offset
from utils.music import parse_unibet_music
from utils.music import ParsedMusic

MAX_MUSIC_MTCHING_OFFSET = 8

logger = setup_logger(name=__file__)


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
    assert (
        race_meeting_id == horse_show.unibet_id
    ), f"{race_meeting_id} != {horse_show.unibet_id}"
    race = Race.upsert(
        race_unibet_id=current_race_dict["zeturfId"],
        race_unibet_n=current_race_dict["rank"],
        race_name=current_race_dict["name"],
        race_start_at=race_start_at,
        race_date=race_date,
        race_type=UnibetRaceType(current_race_dict["type"]),
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
        horse_show_ground=UnibetHorseShowGround(horse_show_dict["ground"]),
        race_track=race_track,
        db_session=db_session,
    )

    return race_track, horse_show


def _get_or_create_parent(
    parent_id: Optional[int],
    name_country: Optional[str],
    is_born_male: bool,
    child_country_code: str,
    db_session: SQLAlchemySession,
) -> Optional[Horse]:
    if parent_id is not None:
        found_parent = db_session.query(Horse).filter(Horse.id == parent_id).one()
        return found_parent

    if not name_country:
        logger.info(
            'Could not find %s name with name: "%s"',
            ("father" if is_born_male else "mother"),
            name_country,
        )
        return None
    name, country_code = _extract_name_country(name_country=name_country)
    potential_parents = (
        db_session.query(Horse)
        .filter(Horse.name == name, Horse.is_born_male.is_(is_born_male))
        .all()
    )
    if not potential_parents:
        parent = Horse(name=name, is_born_male=is_born_male, country_code=country_code)
        db_session.add(parent)
        db_session.commit()
        return parent
    if len(potential_parents) == 1:
        return potential_parents[0]

    assert len(potential_parents) > 1
    if country_code:
        potential_parents = [
            parent
            for parent in potential_parents
            if parent.country_code in (country_code, None)
        ]

    if not potential_parents:
        parent = Horse(name=name, is_born_male=is_born_male, country_code=country_code)
        db_session.add(parent)
        db_session.commit()
        return parent
    if len(potential_parents) == 1:
        return potential_parents[0]
    assert len(potential_parents) > 1

    potential_parents = [
        parent
        for parent in potential_parents
        if parent.country_code in (child_country_code, None)
    ]
    if not potential_parents:
        parent = Horse(name=name, is_born_male=is_born_male, country_code=country_code)
        db_session.add(parent)
        db_session.commit()
        return parent
    if len(potential_parents) == 1:
        return potential_parents[0]

    assert len(potential_parents) > 1
    logger.warning(
        "Too many %s (%s) found for name: %s (%s)",
        ("fathers" if is_born_male else "mothers"),
        len(potential_parents),
        name,
        country_code,
    )
    return None


def _get_or_create_parents(
    horse_id: Optional[int],
    parent_names: Optional[str],
    parent_name_mapping: dict,
    child_country_code: str,
    db_session: SQLAlchemySession,
) -> Tuple[Optional[Horse], Optional[Horse]]:
    father_id, mother_id = None, None
    if horse_id is not None:
        found_horse = db_session.query(Horse).filter(Horse.id == horse_id).one()
        father_id, mother_id = found_horse.father_id, found_horse.mother_id
        if found_horse.father_id and found_horse.mother_id:
            return found_horse.father, found_horse.mother

    father_name, mother_name = _split_parent_names(
        parent_names=parent_names, parent_name_mapping=parent_name_mapping
    )

    father = _get_or_create_parent(
        parent_id=father_id,
        name_country=father_name,
        is_born_male=True,
        child_country_code=child_country_code,
        db_session=db_session,
    )
    mother = _get_or_create_parent(
        parent_id=mother_id,
        name_country=mother_name,
        is_born_male=False,
        child_country_code=child_country_code,
        db_session=db_session,
    )
    return father, mother


def _create_horse(
    name: str,
    country_code: Optional[str],
    father: Optional[Horse],
    mother: Optional[Horse],
    parent_names: Optional[str],
    birth_year: Optional[int],
    is_born_male: Optional[bool],
    db_session: SQLAlchemySession,
) -> Horse:
    horse = Horse(
        name=name,
        country_code=country_code,
        father_id=father.id if father else None,
        mother_id=mother.id if mother else None,
        first_found_origins=parent_names,
        birth_year=birth_year,
        is_born_male=is_born_male,
    )
    db_session.add(horse)
    db_session.commit()
    assert horse.id
    return horse


def _check_update_or_create_horse(
    found_horse: Horse,
    name: str,
    is_born_male: Optional[bool],
    country_code: Optional[str],
    father: Optional[Horse],
    mother: Optional[Horse],
    parent_names: Optional[str],
    origins: Optional[str],
    birth_year: Optional[int],
    db_session: SQLAlchemySession,
) -> Horse:
    """Compare found_horses with found properties,
    if it differs, we create a new horse,
    otherwise we update missing properties with those we have."""

    assert found_horse.name
    assert found_horse.name == name

    if is_born_male is not None and found_horse.is_born_male is not None:
        if is_born_male != found_horse.is_born_male:
            return _create_horse(
                name=name,
                country_code=country_code,
                father=father,
                mother=mother,
                parent_names=parent_names,
                birth_year=birth_year,
                is_born_male=is_born_male,
                db_session=db_session,
            )

    if birth_year and found_horse.birth_year:
        if birth_year != found_horse.birth_year:
            return _create_horse(
                name=name,
                country_code=country_code,
                father=father,
                mother=mother,
                parent_names=parent_names,
                birth_year=birth_year,
                is_born_male=is_born_male,
                db_session=db_session,
            )

    if father and found_horse.father_id:
        if father.id != found_horse.father_id:
            return _create_horse(
                name=name,
                country_code=country_code,
                father=father,
                mother=mother,
                parent_names=parent_names,
                birth_year=birth_year,
                is_born_male=is_born_male,
                db_session=db_session,
            )

    if mother and found_horse.mother_id:
        if mother.id != found_horse.mother_id:
            return _create_horse(
                name=name,
                country_code=country_code,
                father=father,
                mother=mother,
                parent_names=parent_names,
                birth_year=birth_year,
                is_born_male=is_born_male,
                db_session=db_session,
            )

    # Update the horse
    if found_horse.country_code is None and country_code:
        found_horse.country_code = country_code
    if found_horse.father_id is None and father:
        found_horse.father_id = father.id
    if found_horse.mother_id is None and mother:
        found_horse.mother_id = mother.id
    if found_horse.is_born_male is None and is_born_male is not None:
        found_horse.is_born_male = is_born_male
    if found_horse.first_found_origins is None and origins:
        found_horse.first_found_origins = origins
    if found_horse.birth_year is None and birth_year:
        found_horse.birth_year = birth_year
    db_session.commit()
    return found_horse


def _count_not_none_horse_properties(horse: Horse) -> int:
    res = 0
    res += horse.name is not None
    res += horse.country_code is not None
    res += horse.is_born_male is not None
    res += horse.first_found_origins is not None
    res += horse.birth_year is not None
    res += horse.father_id is not None
    res += horse.mother_id is not None
    return res


def _extract_name_country(
    name_country: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """

    >>> _extract_name_country(None)
    (None, None)

    >>> _extract_name_country("Blublu ")
    ('BLUBLU', None)

    >>> _extract_name_country("Blublu (FR)")
    ('BLUBLU', 'FR')

    >>> _extract_name_country("Blublu {USA}")
    ('BLUBLU', 'USA')

    >>> _extract_name_country("Blublu {}")
    ('BLUBLU', None)
    """

    if not name_country:
        return None, None
    name = re.sub(r"\s[\(\{].*[\)\}]", "", name_country)
    name = name.upper()
    name = name.strip()

    country = None
    country_match = re.match(r".*[\(\{](.*)[\)\}]", name_country)
    if country_match:
        country = country_match.group(1)
        country = country.upper()
        country = country.strip()

    return name if name else None, country if country else None


def _process_horse(
    horse_id: Optional[int],
    name_country_from_info: Optional[str],
    name_country_from_runner: str,
    is_born_male: Optional[bool],
    parent_names: Optional[str],
    parent_name_mapping: dict,
    birth_year: Optional[int],
    parsed_music: Optional[ParsedMusic],
    db_session: SQLAlchemySession,
) -> Horse:

    name_from_info, country_from_info = _extract_name_country(
        name_country=name_country_from_info
    )
    name_from_runner, country_from_runner = _extract_name_country(
        name_country=name_country_from_runner
    )
    name = name_from_info if name_from_info else name_from_runner
    assert name
    country_code = country_from_info if country_from_info else country_from_runner
    assert country_code

    father, mother = _get_or_create_parents(
        horse_id=horse_id,
        parent_names=parent_names,
        parent_name_mapping=parent_name_mapping,
        child_country_code=country_code,
        db_session=db_session,
    )
    if horse_id is not None:
        found_horse = db_session.query(Horse).filter(Horse.id == horse_id).one()
        return found_horse

    current_horses = db_session.query(Horse).filter(Horse.name == name).all()

    if is_born_male is not None:
        current_horses = [
            horse
            for horse in current_horses
            if horse.is_born_male is is_born_male or horse.is_born_male is None
        ]

    if father:
        current_horses = [
            horse
            for horse in current_horses
            if horse.father_id == father.id or horse.father_id is None
        ]

    if mother:
        current_horses = [
            horse
            for horse in current_horses
            if horse.mother_id == mother.id or horse.mother_id is None
        ]
    if not current_horses:
        return _create_horse(
            name=name,
            country_code=country_code,
            father=father,
            mother=mother,
            parent_names=parent_names,
            birth_year=birth_year,
            is_born_male=is_born_male,
            db_session=db_session,
        )
    if len(current_horses) == 1:
        found_horse = current_horses[0]
        return _check_update_or_create_horse(
            found_horse=found_horse,
            name=name,
            is_born_male=is_born_male,
            country_code=country_code,
            father=father,
            mother=mother,
            parent_names=parent_names,
            origins=parent_names,
            birth_year=birth_year,
            db_session=db_session,
        )

    if parsed_music:
        current_horses = [
            horse
            for horse in current_horses
            if horse.runners
            and is_matching_with_max_offset(
                future_music=parsed_music,
                past_music=parse_unibet_music(
                    current_year=horse.runners[-1].date.year,
                    music=horse.runners[-1].music,
                ),
                max_offset=MAX_MUSIC_MTCHING_OFFSET,
            )
            is not False
        ] + [horse for horse in current_horses if not horse.runners]
    if not current_horses:
        return _create_horse(
            name=name,
            country_code=country_code,
            father=father,
            mother=mother,
            parent_names=parent_names,
            birth_year=birth_year,
            is_born_male=is_born_male,
            db_session=db_session,
        )

    if len(current_horses) > 1:
        max_not_none_properties = max(
            [_count_not_none_horse_properties(horse) for horse in current_horses]
        )
        selected_horses = [
            horse
            for horse in current_horses
            if _count_not_none_horse_properties(horse) == max_not_none_properties
        ]
        min_id = min([horse.id for horse in selected_horses])
        selected_horses = [horse for horse in selected_horses if horse.id == min_id]
        assert len(selected_horses) == 1
        found_horse = selected_horses[0]
        return _check_update_or_create_horse(
            found_horse=found_horse,
            name=name,
            is_born_male=is_born_male,
            country_code=country_code,
            father=father,
            mother=mother,
            parent_names=parent_names,
            origins=parent_names,
            birth_year=birth_year,
            db_session=db_session,
        )

    assert len(current_horses) == 1
    found_horse = current_horses[0]
    return _check_update_or_create_horse(
        found_horse=found_horse,
        name=name,
        is_born_male=is_born_male,
        country_code=country_code,
        father=father,
        mother=mother,
        parent_names=parent_names,
        origins=parent_names,
        birth_year=birth_year,
        db_session=db_session,
    )


def _filter_names(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        logger.warning("Can not upsert person with name: %s", name)
        return None

    if not name:
        return None
    if not re.search(r"\w", name):
        return None

    name = name.strip()
    if not name:
        return None

    if re.match(r"^\W.*", name):
        name = re.sub(r"^\.", "", name).strip()
        if not name:
            return None

    return name


def _process_runner(
    runner_dict: dict,
    runner_stats: dict,
    race: Race,
    current_race_dict: dict,
    parent_name_mapping: dict,
    db_session: SQLAlchemySession,
) -> None:
    unibet_id = runner_dict["zeturfId"]
    found_runner: Optional[Runner] = (
        db_session.query(Runner).filter(Runner.unibet_id == unibet_id).one_or_none()
    )

    runner_stats_info: Optional[dict] = runner_stats.get("fiche", {}).get(
        "infos_generales"
    )
    # We don't use last performances info in runner_stats
    # Can we trust this runner_stats

    if runner_stats_info:
        name_from_runner_dict, _ = _extract_name_country(
            name_country=runner_dict["name"]
        )

        name_from_info, _ = _extract_name_country(name_country=runner_stats_info["nom"])

        if name_from_info and lev.distance(name_from_runner_dict, name_from_info) > 2:
            logger.warning(
                "Ignoring runner_stats_info with name %s different "
                "from %s in runner_dict",
                name_from_info,
                name_from_runner_dict,
            )
            runner_stats_info = None

    if race.type is None and runner_stats_info:
        race.type = runner_stats_info["discipline"]
        db_session.commit()

    owner_name = runner_dict["details"]["owner"] if runner_dict["details"] else None
    owner_name = None if not isinstance(owner_name, str) else owner_name
    owner_name = (
        runner_stats_info["proprietaire"]
        if runner_stats_info and owner_name is None
        else owner_name
    )
    owner_name = None if not isinstance(owner_name, str) else owner_name

    owner = Entity.upsert(
        entity_id=(found_runner.owner_id if found_runner else None),
        name=owner_name,
        db_session=db_session,
    )
    trainer_name = (
        runner_dict["details"]["trainer"]
        if runner_dict["details"]
        else (runner_stats_info["entraineur"] if runner_stats_info else None)
    )

    trainer = Entity.upsert(
        entity_id=(found_runner.trainer_id if found_runner else None),
        name=trainer_name,
        db_session=db_session,
    )

    jockey = Entity.upsert(
        entity_id=(found_runner.jockey_id if found_runner else None),
        name=runner_dict["jockey"],
        db_session=db_session,
    )

    horse_sex = runner_stats_info.get("sex") if runner_stats_info else None
    horse_sex = (
        horse_sex
        if horse_sex
        else (runner_dict["details"].get("sex") if runner_dict.get("details") else None)
    )
    horse_sex = UnibetHorseSex(horse_sex or None)
    is_born_male = horse_sex.is_born_male

    parent_names = _get_parent_names(
        runner_dict=runner_dict, runner_stats_info=runner_stats_info
    )
    birth_year = runner_stats_info.get("age") if runner_stats_info else None

    music = (
        runner_dict["details"].get("musique") if runner_dict.get("details") else None
    )
    parsed_music = parse_unibet_music(current_year=race.date.year, music=music)
    if parsed_music is None and music:
        logger.warning('Can not parse music "%s"', music)

    horse = _process_horse(
        horse_id=(found_runner.horse_id if found_runner else None),
        name_country_from_info=runner_stats_info["nom"] if runner_stats_info else None,
        name_country_from_runner=runner_dict["name"],
        is_born_male=is_born_male,
        parent_names=parent_names,
        parent_name_mapping=parent_name_mapping,
        birth_year=birth_year,
        parsed_music=parsed_music,
        db_session=db_session,
    )

    coat_str = (
        runner_dict["details"].get("coat") if runner_dict.get("details") else None
    )
    coat = unibet_coat_parser.get_coat(coat=coat_str)
    if (
        coat == UnibetCoat.UNKNOWN
        and runner_stats_info
        and runner_stats_info.get("robe")
    ):
        coat = unibet_coat_parser.get_coat(runner_stats_info.get("robe"))

    blinkers = UnibetBlinkers(runner_dict["blinkers"])

    stakes = (
        runner_dict["details"].get("stakes") if runner_dict.get("details") else None
    )

    shoes = UnibetShoes(runner_dict["shoes"])
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

    weight = runner_dict["weight"]
    assert isinstance(weight, int)
    if weight > 200:
        logger.warning(
            "Removing weight %s >200 on race %s, runner unibet_id %s",
            weight,
            race.id,
            unibet_id,
        )
        weight = None

    age_str = runner_dict["details"].get("age") if runner_dict.get("details") else None
    age: Optional[int] = None
    if age_str == "":
        age = None
    elif age_str is not None and int(age_str) > 100:
        logger.warning(
            "Incorrect age '%s' for runner's Unibet_id %s", age_str, unibet_id
        )
        age = None
    elif age_str is not None:
        age = int(age_str)

    _ = Runner.upsert(
        unibet_id=unibet_id,
        race=race,
        race_duration_sec=convert_duration_in_sec(
            time_str=runner_dict["details"].get("time")
        )
        if runner_dict.get("details")
        else None,
        weight=weight,
        unibet_n=runner_dict["rank"],
        rope_n=runner_stats_info.get("corde") if runner_stats_info else None,
        draw=runner_dict["draw"],
        blinkers=blinkers,
        shoes=shoes,
        silk=runner_dict["silk"],
        stakes=stakes,
        music=music,
        sex=UnibetHorseSex(horse_sex or None),
        age=age,
        team=runner_dict["team"],
        coat=coat,
        origins=parent_names,
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


def _get_male_female_names(
    runner_dict: dict, runner_stats_info: dict
) -> Tuple[Optional[str], Optional[str]]:
    name_country_from_info = runner_stats_info["nom"] if runner_stats_info else None
    name_country_from_runner = runner_dict["name"]

    name_from_info, _ = _extract_name_country(name_country=name_country_from_info)
    name_from_runner, _ = _extract_name_country(name_country=name_country_from_runner)
    name = name_from_info if name_from_info else name_from_runner
    assert name

    horse_sex = runner_stats_info.get("sex") if runner_stats_info else None
    horse_sex = (
        horse_sex
        if horse_sex
        else (runner_dict["details"].get("sex") if runner_dict.get("details") else None)
    )
    horse_sex = UnibetHorseSex(horse_sex or None)

    if horse_sex == UnibetHorseSex.UNKNOWN:
        return None, None

    if horse_sex.is_born_male:
        return name, None

    return None, name


def _get_parent_names(
    runner_dict: dict, runner_stats_info: Optional[dict]
) -> Optional[str]:
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

    return parent_names


def _split_parent_names(
    parent_names: Optional[str], parent_name_mapping: dict
) -> Tuple[Optional[str], Optional[str]]:
    if not parent_names:
        return None, None
    if parent_names in parent_name_mapping:
        return parent_name_mapping[parent_names]
    father_mother_names = re.split(r"[-/]", parent_names)
    if len(father_mother_names) != 2:
        father_mother_names = re.split(r"\s[-/]\s", parent_names)
    if len(father_mother_names) != 2:
        father_mother_names = re.split(r"\set\s", parent_names)
    if len(father_mother_names) != 2:
        father_mother_names = parent_names.strip().split()

    if len(father_mother_names) != 2:
        logger.warning(
            'Could not find father mother names in origins: "%s"', parent_names
        )
        return None, None

    father_name, mother_name = father_mother_names
    father_name = father_name.upper().strip()
    mother_name = mother_name.upper().strip()
    return father_name, mother_name


def _resolve_parent_names_from_already_names(
    male_names: Set[str],
    female_names: Set[str],
    not_understood_parent_names: Set[str],
    current_parent_name_mapping: Dict[str, Tuple[str, str]],
) -> Dict[str, Tuple[str, str]]:

    parent_name_mapping = current_parent_name_mapping
    for parent_name in tqdm.tqdm(
        not_understood_parent_names, total=len(not_understood_parent_names)
    ):
        if not parent_name:
            continue

        if parent_name in parent_name_mapping:
            continue

        matching_mother_names = []
        matching_father_names = []
        for male_name in male_names:
            if "(" in male_name or ")" in male_name:
                continue

            if re.search(fr"^{male_name.upper()}\b", parent_name.upper()):
                matching_father_names.append(male_name)

        for female_name in female_names:
            if "(" in female_name or ")" in female_name:
                continue

            if re.search(fr"\b{female_name.upper()}$", parent_name.upper()):
                matching_mother_names.append(female_name)

        matched_mother_name = None
        if matching_mother_names:
            matched_mother_name = max(matching_mother_names, key=len)
        matched_father_name = None
        if matching_father_names:
            matched_father_name = max(matching_father_names, key=len)

        if (len(matched_mother_name or "") + len(matched_father_name or "")) > len(
            parent_name
        ):
            logger.warning(
                """Error with "%s": found mother_name (%s), found father_name (%s)""",
                parent_name,
                matched_mother_name,
                matched_father_name,
            )
            continue

        if matched_mother_name and not matched_father_name:
            new_father_name = re.sub(
                fr"\b{matched_mother_name.upper()}$", "", parent_name.upper()
            )
            new_father_name = new_father_name.strip()
            if not new_father_name:
                continue
            parent_name_mapping[parent_name] = (new_father_name, matched_mother_name)

        if matched_father_name and not matched_mother_name:
            new_mother_name = re.sub(
                fr"^{matched_father_name.upper()}\b", "", parent_name.upper()
            )

            new_mother_name = new_mother_name.strip()
            if not new_mother_name:
                continue
            parent_name_mapping[parent_name] = (matched_father_name, new_mother_name)

        if matched_father_name and matched_mother_name:
            parent_name_mapping[parent_name] = (
                matched_father_name,
                matched_mother_name,
            )

    return parent_name_mapping


def pre_run():
    """
    The goal of this function is to generate a json with hard to identify parent names.
    """

    with open(os.path.join(DATA_DIR, "unibet_parent_name_mapping.json"), "r") as fp:
        current_parent_name_mapping = json.load(fp=fp)

    male_names = set()
    female_names = set()
    not_understood_parent_names = set()
    for date in tqdm.tqdm(
        date_countdown_generator(
            start_date=UNIBET_MIN_DATE, end_date=dt.date.today() - dt.timedelta(days=1)
        ),
        total=(dt.date.today() - dt.timedelta(days=1) - UNIBET_MIN_DATE).days,
        unit="days",
    ):
        if not date.isoformat() in os.listdir(UNIBET_DATA_PATH):
            logger.info("Could not find folder for date: %s", date.isoformat())
            continue
        day_folder_path = os.path.join(UNIBET_DATA_PATH, date.isoformat())
        if "programme.json" not in os.listdir(day_folder_path):
            logger.info("Could not find programme.json for date: %s", date.isoformat())
            continue

        with open(os.path.join(day_folder_path, "programme.json"), "r") as fp:
            programme = json.load(fp=fp)
        if "data" not in programme:
            logger.info("Can not import programme of %s", date.isoformat())
            continue

        horse_shows_dict = programme["data"]
        for horse_show_dict in horse_shows_dict:

            if horse_show_dict.get("races"):

                for race_dict in horse_show_dict["races"]:
                    race_path = os.path.join(
                        day_folder_path,
                        f"R{horse_show_dict['rank']}_C" f"{race_dict['rank']}.json",
                    )
                    with open(race_path, "r") as fp:
                        complete_race_dict = json.load(fp=fp)

                    current_race_dict = complete_race_dict
                    if complete_race_dict.get("note") == "server error, no json":
                        # Can not use complete_race
                        current_race_dict = race_dict

                    for runner_dict in current_race_dict["runners"]:

                        runner_statistics_path = os.path.join(
                            day_folder_path,
                            f"R{horse_show_dict['rank']}_"
                            f"C{race_dict['rank']}_"
                            f"RN{runner_dict['rank']}.json",
                        )
                        with open(runner_statistics_path, "r") as fp:
                            runner_stats = json.load(fp=fp)

                        runner_stats_info: Optional[dict] = runner_stats.get(
                            "fiche", {}
                        ).get("infos_generales")

                        male_name, female_name = _get_male_female_names(
                            runner_dict=runner_dict, runner_stats_info=runner_stats_info
                        )
                        if male_name:
                            male_names.add(male_name)
                        if female_name:
                            female_names.add(female_name)

                        parent_name = _get_parent_names(
                            runner_dict=runner_dict, runner_stats_info=runner_stats_info
                        )
                        if parent_name in current_parent_name_mapping:
                            continue
                        father_name, mother_name = _split_parent_names(
                            parent_names=parent_name,
                            parent_name_mapping=current_parent_name_mapping,
                        )
                        father_name, _ = _extract_name_country(name_country=father_name)
                        mother_name, _ = _extract_name_country(name_country=mother_name)
                        if father_name is None and mother_name is None:
                            not_understood_parent_names.add(parent_name)

                        if father_name:
                            male_names.add(father_name)
                        if mother_name:
                            female_names.add(mother_name)
    logger.info(
        "%s male names, %s female names, %s not understood parent names",
        len(male_names),
        len(female_names),
        len(not_understood_parent_names),
    )
    parent_name_mapping = _resolve_parent_names_from_already_names(
        male_names=male_names,
        female_names=female_names,
        not_understood_parent_names=not_understood_parent_names,
        current_parent_name_mapping=current_parent_name_mapping,
    )

    with open(os.path.join(DATA_DIR, "unibet_parent_name_mapping.json"), "w+") as fp:
        json.dump(obj=parent_name_mapping, fp=fp)


def run():  # pylint:disable=too-many-branches
    with create_sqlalchemy_session() as db_session:
        with open(os.path.join(DATA_DIR, "unibet_parent_name_mapping.json"), "r") as fp:
            parent_name_mapping = json.load(fp=fp)
        for date in tqdm.tqdm(
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
                                f"R{horse_show.unibet_n}_"
                                f"C{race.unibet_n}_"
                                f"RN{runner_dict['rank']}.json",
                            )
                            with open(runner_statistics_path, "r") as fp:
                                runner_stats = json.load(fp=fp)
                            _process_runner(
                                runner_dict=runner_dict,
                                runner_stats=runner_stats,
                                race=race,
                                current_race_dict=current_race_dict,
                                parent_name_mapping=parent_name_mapping,
                                db_session=db_session,
                            )


if __name__ == "__main__":
    # print("Generating hard to identify parent names mapping from already "
    #       "known horse names...")
    # pre_run()
    print("Backfilling to DB")
    run()
