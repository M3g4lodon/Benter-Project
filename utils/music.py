"""
# PMU
https://communaute-aide.pmu.fr/questions/1332044-lire-musique-cheval
1. Pour tous les chevaux :

- " 0" indique qu'il n'a pas été classé parmi les 10 premiers, tout chiffre
    différent de 0 indique la place du cheval dans la course

- " T" indique qu'il est tombé

- " A" indique qu'il a été arrêté

- " Ret" signifie qu'il a été rétrogradé de la place.



2. Pour un galopeur :

- " s" désigne les épreuves du type steeple-chase

- " h" désigne les épreuves de haies

- " c" désigne les épreuves de cross

- " p" désigne les courses de Plat



3. Pour un trotteur :

- "a" signifie course Attelée"m" signifie course Montée

- "0" indique qu'il n'a pas été classé parmi les 10 premiers

- "D" signifie qu'il a été disqualifié pour allure irrégulière

# From UNIBET
https://www.unibet.fr/turf/unibet-lexique-turf-194416203.html
Pour un galopeur

    La lettre "S" désigne les épreuves du type steeple-chase
    La lettre "H" désigne les épreuves de haies
    La lettre "C" désigne les épreuves de cross
    La lettre "P" désigne les courses de Plat

    La lettre "T" indique qu'il est tombé
    La lettre "A" indique qu'il a été arrêté
    La mention "RET" indique qu'il a été rétrogradé de la place
    Le chiffre "0" indique qu'il n'a pas été classé parmi les 10 premiers
    Tout chiffre différent de 0 indique la place du cheval dans la course


Pour un trotteur

    La lettre "A" désigne les épreuve de course Attelée
    La lettre "M" désigne les épreuve de course Montée

    La lettre "D" indique qu'il a été disqualifié pour allure irrégulière
    La lettre "T" indique qu'il est tombé
    La lettre "A" indique qu'il a été arrêté
    La mention "RET" signifie qu'il a été rétrogradé de la place
    Le chiffre "0" indique qu'il n'a pas été classé parmi les 10 premiers
    Tout chiffre différent de 0 indique la place du cheval dans la course

"""
import re
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from constants import UnibetRaceType
from utils import setup_logger

logger = setup_logger(name=__file__)
ParsedMusicStats = namedtuple(
    "ParsedMusic", ["win_rate", "mean_place", "n_races_in_music"]
)


def parse_music(  # pylint:disable=too-many-branches
    music: str, verbose=False
) -> ParsedMusicStats:
    if pd.isna(music):
        return ParsedMusicStats(win_rate=None, mean_place=None, n_races_in_music=None)

    if music == "()":
        return ParsedMusicStats(win_rate=None, mean_place=None, n_races_in_music=None)

    events = []

    for event in re.findall(
        r"([ATRDN]|\d{1,2})([amshcpt])", re.sub(r"\(\d{1,2}\)", "", music)
    ):
        if event == "":
            continue
        if any("inédit" in e for e in event):
            continue

        if any(e == "(r)" for e in event):
            continue
        if len(event) != 2:
            continue

        events.append(event)

    if verbose:
        print(f"Found events: {events}")

    if not events:
        return ParsedMusicStats(win_rate=np.nan, mean_place=np.nan, n_races_in_music=0)

    win_rate = float(np.mean([position == "1" for position, _ in events]))
    mean_place = float(
        np.mean(
            [int(position) for position, _ in events if re.match(r"\d{1,2}", position)]
        )
    )
    n_races_in_music = len(events)
    return ParsedMusicStats(
        win_rate=win_rate, mean_place=mean_place, n_races_in_music=n_races_in_music
    )


class MusicRank(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    FIFTH = 5
    SIXTH = 6
    SEVENTH = 7
    EIGHTH = 8
    NINTH = 9
    TENTH_AND_BELOW = 0
    RETIRED = "R"
    DISQUALIFIED = "D"
    FALLEN = "T"
    STOPPED = "A"

    @classmethod
    def from_music(cls, value: str) -> "MusicRank":
        if re.match(r"^\d$", value):
            return MusicRank(value=int(value))
        if value == "N":
            return MusicRank.TENTH_AND_BELOW
        if re.match(r"\d?Dist|Disq?|Dq?", value):
            return MusicRank.DISQUALIFIED
        if value in ("R", "Ret", "RET", "Rd"):
            return MusicRank.RETIRED
        return MusicRank(value=value)


@dataclass
class MusicEvent:
    rank: MusicRank
    race_type: UnibetRaceType
    is_first_race: bool
    year: int


@dataclass
class ParsedMusic:
    events: List[MusicEvent]
    is_new: bool


race_types_mapping = {
    "a": UnibetRaceType.HARNESS,
    "e": UnibetRaceType.HARNESS,
    "m": UnibetRaceType.MOUNTED,
    "h": UnibetRaceType.HURDLE,
    "c": UnibetRaceType.CROSS_COUNTRY,
    "p": UnibetRaceType.FLAT,
    "o": UnibetRaceType.HURDLE,
    "s": UnibetRaceType.STEEPLE_CHASE,
    "t": UnibetRaceType.FLAT,
}


def parse_unibet_music(
    current_year: int, music: Optional[str]
) -> Optional[ParsedMusic]:
    if music is None:
        return None

    if re.match(r"^\W*$", music):
        return None

    if re.match(r"in([eé]|Ã©)dit|d[eé]b", music, flags=re.IGNORECASE):
        return ParsedMusic(events=[], is_new=True)

    music_events: List[MusicEvent] = []
    beginning_suffix = r"d[eé]b\.?$"
    shortened_music = re.sub(beginning_suffix, "", music, flags=re.IGNORECASE)
    events = re.findall(
        r"(\d?Dist|Disq?|Dq|RET|Rd|[ATRDN]|\d)([a-z])|\((\d{1,2})\)", shortened_music
    )
    for event in events:
        rank, race_type, year = event
        if year:
            assert not rank
            assert not race_type
            if int(year) >= 90:
                prev_current_year = int(f"19{year}")
            else:
                prev_current_year = int(f"20{year.zfill(2)}")
            if not 1989 < prev_current_year < 2040:
                logger.warning(
                    'Can not understand "%s" as a year in the past '
                    "(current_year: %s, computed prev_current_year %s)",
                    year,
                    current_year,
                    prev_current_year,
                )
                return None
            # TODO store error case in json
            # We read music from left to right to go backward in time
            if prev_current_year > current_year:
                logger.warning(
                    "Error while parsing music %s on current year %s",
                    music,
                    current_year,
                )
                return None

            current_year = prev_current_year
            continue
        assert not year
        music_events.append(
            MusicEvent(
                rank=MusicRank.from_music(value=rank),
                race_type=race_types_mapping.get(race_type, UnibetRaceType.UNKNOWN),
                year=current_year,
                is_first_race=False,
            )
        )
    if re.search(beginning_suffix, music, flags=re.IGNORECASE):
        music_events[-1] = MusicEvent(
            rank=music_events[-1].rank,
            race_type=music_events[-1].race_type,
            year=music_events[-1].year,
            is_first_race=True,
        )
    return ParsedMusic(events=music_events, is_new=False)


def is_matching_music_events(
    future_music: Optional[ParsedMusic], past_music: Optional[ParsedMusic], offset: int
) -> Optional[bool]:
    assert offset >= 0
    if not future_music:
        return None
    if not past_music:
        return None
    if not future_music.events:
        return False
    if future_music.is_new:
        return False
    if not past_music.events and len(future_music.events) == offset:
        return True
    if not past_music.events:
        return False
    if len(future_music.events) <= offset:
        return False
    # We don't look at if it is a first race or not
    return all(
        (event_1.year, event_1.rank, event_1.race_type)
        == (event_2.year, event_2.rank, event_2.race_type)
        for event_1, event_2 in zip(past_music.events, future_music.events[offset:])
    )


def is_matching_with_max_offset(
    future_music: Optional[ParsedMusic],
    past_music: Optional[ParsedMusic],
    max_offset: int,
) -> Optional[bool]:
    assert max_offset >= 0
    if not future_music:
        return None
    if not past_music:
        return None
    if not future_music.events:
        return False
    if future_music.is_new:
        return False
    for offset in range(max_offset + 1):
        if is_matching_music_events(
            future_music=future_music, past_music=past_music, offset=offset
        ):
            return True
    return False
