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


# From https://github.com/pourquoi/cataclop/blob/
# 47298c16de1def16513f78db24c4d78d697fc8ce/cataclop/ml/preprocessing.py


def parse_cataclop_music(music, length):
    positions = np.zeros(length)

    pos = None
    cat = None
    is_year = False
    i = 0
    for c in music:
        if i + 1 > length:
            break

        if c == "(":
            is_year = True
            continue

        if c == ")":
            is_year = False
            continue

        if is_year:
            continue

        if pos is None:
            pos = c
            cat = None
            positions[i] = pos if pos.isdigit() else 0
            if positions[i] == 0:
                positions[i] = 10
            continue

        if cat is None:
            cat = c
            pos = None
            i = i + 1
            continue

    return pd.Series(
        [p for p in positions[:length]],
        index=["hist_{:d}_pos".format(i + 1) for i in range(length)],
    )
