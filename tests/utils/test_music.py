import numpy as np
import pytest

from constants import UnibetRaceType
from scripts.generate_unibet import MAX_MUSIC_MTCHING_OFFSET
from utils.music import is_matching_with_max_offset
from utils.music import MusicEvent
from utils.music import MusicRank
from utils.music import parse_music
from utils.music import parse_unibet_music
from utils.music import ParsedMusic
from utils.music import ParsedMusicStats


def test_music_parser():
    assert parse_music("0p 3p Ap") == ParsedMusicStats(0.0, 1.5, 3)
    assert parse_music("0p3pAp") == ParsedMusicStats(0.0, 1.5, 3)

    assert parse_music("1p 3p 1a 0t") == ParsedMusicStats(0.5, 1.25, 4)

    assert np.isnan(parse_music(" (inédit) ")[0])
    assert np.isnan(parse_music(" (inédit) ")[1])
    assert parse_music(" (inédit) ")[2] == 0


@pytest.mark.parametrize(
    "music_str,current_year,expected_music",
    [
        ("", 2009, None),
        ("()", 2009, None),
        ("\n", 2009, None),
        ("Deb.", 2009, ParsedMusic(events=[], is_new=True)),
        ("Debut", 2009, ParsedMusic(events=[], is_new=True)),
        ("Début", 2009, ParsedMusic(events=[], is_new=True)),
        ("Inédit", 2009, ParsedMusic(events=[], is_new=True)),
        ("InÃ©dit", 2009, ParsedMusic(events=[], is_new=True)),
        ("Da (10) Da (11) Da", 2011, None),
        (
            "4p 1p (09) 5p 3p 1Distp",
            2010,
            ParsedMusic(
                events=[
                    MusicEvent(
                        rank=MusicRank.FOURTH,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2010,
                    ),
                    MusicEvent(
                        rank=MusicRank.FIRST,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2010,
                    ),
                    MusicEvent(
                        rank=MusicRank.FIFTH,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2009,
                    ),
                    MusicEvent(
                        rank=MusicRank.THIRD,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2009,
                    ),
                    MusicEvent(
                        rank=MusicRank.DISQUALIFIED,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2009,
                    ),
                ],
                is_new=False,
            ),
        ),
        (
            "4p 1p (09) 5p 3p 1p",
            2010,
            ParsedMusic(
                events=[
                    MusicEvent(
                        rank=MusicRank.FOURTH,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2010,
                    ),
                    MusicEvent(
                        rank=MusicRank.FIRST,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2010,
                    ),
                    MusicEvent(
                        rank=MusicRank.FIFTH,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2009,
                    ),
                    MusicEvent(
                        rank=MusicRank.THIRD,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2009,
                    ),
                    MusicEvent(
                        rank=MusicRank.FIRST,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2009,
                    ),
                ],
                is_new=False,
            ),
        ),
        (
            "0p 0p 0p (04) 0p 6p 9p Deb",
            2005,
            ParsedMusic(
                events=[
                    MusicEvent(
                        rank=MusicRank.TENTH_AND_BELOW,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2005,
                    ),
                    MusicEvent(
                        rank=MusicRank.TENTH_AND_BELOW,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2005,
                    ),
                    MusicEvent(
                        rank=MusicRank.TENTH_AND_BELOW,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2005,
                    ),
                    MusicEvent(
                        rank=MusicRank.TENTH_AND_BELOW,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2004,
                    ),
                    MusicEvent(
                        rank=MusicRank.SIXTH,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=False,
                        year=2004,
                    ),
                    MusicEvent(
                        rank=MusicRank.NINTH,
                        race_type=UnibetRaceType.FLAT,
                        is_first_race=True,
                        year=2004,
                    ),
                ],
                is_new=False,
            ),
        ),
    ],
)
def test_unibet_music_parser(music_str, current_year, expected_music):
    assert (
        parse_unibet_music(music=music_str, current_year=current_year) == expected_music
    )


@pytest.mark.parametrize(
    "prev_music_str,prev_year,future_music_str,future_year,expected",
    [
        (
            "1a 1a 1m Dm (05) 3m 5a 4a 3m 1a 1a",
            2006,
            "7a 1a Da 1m Dm (05) 3m 5a 4a 3m 1a",
            2006,
            False,
        ),
        ("0p 0p 0p (04) 0p 6p 9p Deb", 2005, "9p 0p 0p 0p (04) 0p 6p 9p", 2005, True),
        (
            "1a 1a Da Da 1a 7a 6a 1a 0a 2a (17) 2a 1a ",
            2018,
            "Da Da 1a 7a 6a 1a 0a 2a (17) 2a 1a Da 1a ",
            2018,
            False,
        ),
    ],
)
def test_matching_musics(
    prev_music_str, prev_year, future_music_str, future_year, expected
):
    prev_music = parse_unibet_music(current_year=prev_year, music=prev_music_str)
    current_music = parse_unibet_music(current_year=future_year, music=future_music_str)
    assert (
        is_matching_with_max_offset(
            future_music=current_music,
            past_music=prev_music,
            max_offset=MAX_MUSIC_MTCHING_OFFSET,
        )
        is expected
    )
