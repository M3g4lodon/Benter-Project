"""
# just read the music
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
"""
import re
from collections import namedtuple

import numpy as np
import pandas as pd

ParsedMusic = namedtuple("ParsedMusic", ["win_rate", "mean_place", "n_races_in_music"])


def parse_music(music: str, verbose=False) -> ParsedMusic:
    if pd.isna(music):
        return ParsedMusic(win_rate=None, mean_place=None, n_races_in_music=None)

    if music == "()":
        return ParsedMusic(win_rate=None, mean_place=None, n_races_in_music=None)

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
        return ParsedMusic(win_rate=np.nan, mean_place=np.nan, n_races_in_music=0)

    win_rate = float(np.mean([position == "1" for position, _ in events]))
    mean_place = float(
        np.mean(
            [int(position) for position, _ in events if re.match(r"\d{1,2}", position)]
        )
    )
    n_races_in_music = len(events)
    return ParsedMusic(
        win_rate=win_rate, mean_place=mean_place, n_races_in_music=n_races_in_music
    )
