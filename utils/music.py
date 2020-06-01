"""
# just read the music
https://communaute-aide.pmu.fr/questions/1332044-lire-musique-cheval
1. Pour tous les chevaux :

- " 0" indique qu'il n'a pas été classé parmi les 10 premiers, tout chiffre différent de 0 indique la place du cheval dans la course

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
"""
import re
from collections import namedtuple

import numpy as np
import pandas as pd

ParsedMusic = namedtuple("ParsedMusic", ["win_rate", "mean_place"])


def parse_music(music: str, verbose=False) -> ParsedMusic:
    if pd.isna(music):
        return ParsedMusic(win_rate=None, mean_place=None)

    if music == "()":
        return ParsedMusic(win_rate=None, mean_place=None)

    events = []

    for event in re.findall(
        r"([ATRDN]|\d{1,2})([amshcpt])", re.sub(r"\(\d{1,2}\)", "", music)
    ):
        if event == "":
            continue
        if any("inédit" in e for e in event):
            continue

        if any("(r)" == e for e in event):
            continue
        if len(event) != 2:
            continue

        events.append(event)

    if verbose:
        print(f"Found events: {events}")

    win_rate = float(np.mean([position == "1" for position, _ in events]))
    mean_place = float(
        np.mean(
            [int(position) for position, _ in events if re.match(r"\d{1,2}", position)]
        )
    )
    return ParsedMusic(win_rate=win_rate, mean_place=mean_place)


if __name__ == "__main__":
    assert parse_music("0p 3p Ap") == ParsedMusic(0.0, 1.5)
    assert parse_music("0p3pAp") == ParsedMusic(0.0, 1.5)

    assert parse_music("1p 3p 1a 0t") == ParsedMusic(0.5, 1.25)

    assert np.isnan(parse_music(" (inédit) ")[0])
    assert np.isnan(parse_music(" (inédit) ")[1])
