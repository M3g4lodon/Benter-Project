import numpy as np

from utils.music import parse_music
from utils.music import ParsedMusic


def test_music_parser():
    assert parse_music("0p 3p Ap") == ParsedMusic(0.0, 1.5, 3)
    assert parse_music("0p3pAp") == ParsedMusic(0.0, 1.5, 3)

    assert parse_music("1p 3p 1a 0t") == ParsedMusic(0.5, 1.25, 4)

    assert np.isnan(parse_music(" (inédit) ")[0])
    assert np.isnan(parse_music(" (inédit) ")[1])
    assert parse_music(" (inédit) ")[2] == 0
