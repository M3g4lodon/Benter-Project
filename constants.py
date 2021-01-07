import collections
import datetime as dt
import os
from enum import Enum
from typing import Optional

TIMEZONE = "Europe/Paris"


SOURCE_PMU = "PMU"
SOURCE_Unibet = "Unibet"
SOURCE_ZeTurf = "ZeTurf"

SOURCES = [SOURCE_PMU, SOURCE_Unibet, SOURCE_ZeTurf]

DATA_DIR = "./data"
CACHE_DIR = "./cache"
SAVED_MODELS_DIR = "./saved_models"
PMU_DATA_DIR = os.path.join(DATA_DIR, SOURCE_PMU)
UNIBET_DATA_DIR = os.path.join(DATA_DIR, SOURCE_Unibet)


PMU_INCIDENT_TYPES = {
    "ARRETE",
    "DEROBE",
    "DEROBE_RAMENE",
    "DISQUALIFIE_POTEAU_GALOP",
    "DISQUALIFIE_POUR_ALLURE_IRREGULIERE",
    "DISTANCE",
    "NON_PARTANT",
    "RESTE_AU_POTEAU",
    "TOMBE",
}

PMU_CODE_PARIS = {
    "CLASSIC_TIERCE",
    "COUPLE_GAGNANT",
    "COUPLE_ORDRE",
    "COUPLE_ORDRE_INTERNATIONAL",
    "COUPLE_PLACE",
    "DEUX_SUR_QUATRE",
    "E_CLASSIC_TIERCE",
    "E_COUPLE_GAGNANT",
    "E_COUPLE_ORDRE",
    "E_COUPLE_PLACE",
    "E_DEUX_SUR_QUATRE",
    "E_MINI_MULTI",
    "E_MULTI",
    "E_PICK5",
    "E_QUARTE_PLUS",
    "E_QUINTE_PLUS",
    "E_REPORT_PLUS",
    "E_SIMPLE_GAGNANT",
    "E_SIMPLE_PLACE",
    "E_SUPER_QUATRE",
    "E_TIC_TROIS",
    "E_TIERCE",
    "E_TRIO",
    "E_TRIO_ORDRE",
    "MINI_MULTI",
    "MULTI",
    "PICK5",
    "QUARTE_PLUS",
    "QUINTE_PLUS",
    "REPORT_PLUS",
    "SIMPLE_GAGNANT",
    "SIMPLE_GAGNANT_INTERNATIONAL",
    "SIMPLE_PLACE",
    "SIMPLE_PLACE_INTERNATIONAL",
    "SUPER_QUATRE",
    "TIC_TROIS",
    "TIERCE",
    "TRIO",
    "TRIO_ORDRE",
    "TRIO_ORDRE_INTERNATIONAL",
}
PMU_STATUT = {
    "ARRIVEE_DEFINITIVE",
    "ARRIVEE_DEFINITIVE_COMPLETE",
    "ARRIVEE_PROVISOIRE",
    "COURSE_ANNULEE",
    "COURSE_ARRETEE",
    "DEPART_CONFIRME",
    "DEPART_DANS_TROIS_MINUTES",
    "FIN_COURSE",
    "PROGRAMMEE",
}

PMU_MIN_DATE = dt.date(2013, 2, 20)

# From PMU rules
Betting = collections.namedtuple("Betting", ["name", "track_take"])

PMU_BETTINGS = [
    Betting("E_SIMPLE_GAGNANT", 0.139),
    Betting("e_simple_jackpot", 0.1725),
    Betting("e_couplé", 0.2670),
    Betting("e_2_sur_4", 0.2685),
    Betting("e_2_sur_4_jackpot", 0.2951),
    Betting("e_tiercé", 0.2265),
    Betting("e_trio", 0.2515),
    Betting("e_quarté+", 0.2465),
    Betting("e_multi", 0.3185),
    Betting("e_mini_multi", 0.3185),
    Betting("e_quinté+", 0.3565),
    Betting("e_pick_5", 0.3655),
    Betting("e_super_4", 0.1675),
]

PMU_MINIMUM_BET_SIZE = 150  # 1.50€

# Unibet
UNIBET_MIN_DATE = dt.date(2005, 6, 18)


class UnibetHorseSex(Enum):
    MALE = "M"
    FEMALE = "F"
    GELDING = "H"  # "hongre" in French
    UNKNOWN = None

    @property
    def is_born_male(self) -> Optional[bool]:
        if self == self.UNKNOWN:
            return None
        if self in [self.MALE, self.GELDING]:
            return True
        assert self == self.FEMALE
        return False


class UnibetBetRateType:
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


class UnibetRaceType(Enum):
    HURDLE = "Haies"
    HARNESS = "Attelé"
    STEEPLE_CHASE = "Steeple-Chase"
    CROSS_COUNTRY = "Cross-Country"
    MOUNTED = "Monté"
    FLAT = "Plat"
    UNKNOWN = None


class UnibetHorseShowGround(Enum):
    LIGHT = "Léger"
    SLOW = "Lent"
    UNKNOWN = "-"
    HEAVY = "Lourd"
    VERY_LOOSE = "Très souple"
    DRY = "Sec"
    GOOD = "Bon"
    LOOSE = "Souple"
    STICKY = "Collant"
    FAST = "Rapide"
    VERY_HEAVY = "Très lourd"
    GOOD_LOOSE = "Bon-Souple"
    VERY_FAST = "Très rapide"
    GOOD_LIGHT = "Bon-Léger"
    STANDART = "Standard"


class UnibetBlinkers(Enum):
    UNKNOWN = None
    NO_BLINKERS = "0"  # or ""
    BLINKERS = "1"  # or "X" for runner_info_stats
    AUSTRALIEN_BLINKERS = "2"  # or "A" for runner_info_stats
    FIRST_TIME_BLINKERS = "3"
    FIRST_TIME_AUSTRALIAN_BLINKERS = "4"


class UnibetShoes(Enum):
    SHOD = ""
    FIRST_TIME_FULLY_UNSHOD = "5"
    FULLY_UNSHOD = "4"
    FIRST_TIME_BACK_UNSHOD = "Q"
    BACK_UNSHOD = "P"
    FIST_TIME_FRONT_UNSHOD = "B"
    FRONT_UNSHOD = "A"

# https://www.lexiqueducheval.net/lexique_sommaire.html
class UnibetCoat(Enum):
    UNKNOWN = None
    PALOMINO = "PALOMINO"
    ROUAN = "ROUAN"
    WHITE = "WHITE"
    BLACK = "BLACK"
    GREY = "GREY"
    CHESTNUT = "CHESTNUT"
    DUN = "DUN"
    BAY = "BAY"
    DARK_BAY = "DARK_BAY"
    LIGHT_BAY = "LIGHT_BAY"
    BROWN = "BROWN"
    STRAWBERRY_ROUAN = "STRAWBERRY_ROUAN"
    LIVER_CHESTNUT = "LIVER_CHESTNUT"
    LIGHT_CHESTNUT = "LIGHT_CHESTNUT"
    DARK_CHESTNUT = "DARK_CHESTNUT"
    FLAXEN_CHESTNUT = "FLAXEN_CHESTNUT"
    COPPER_CHESTNUT = "COPPER_CHESTNUT"
    PIEBALD_CHESTNUT = "PIEBALD_CHESTNUT"
    RUBICANO_CHESTNUT = "RUBICANO_CHESTNUT"
    MEDLEY_CHESTNUT = "MEDLEY_CHESTNUT"
    AUBERE = "AUBERE"
    CHERRY_BAY = "CHERRY_BAY"
    RUBICANO_BAY = "RUBICANO_BAY"
    GREY_BAY = "GREY_BAY"
    PIEBALD_BAY = "PIEBALD_BAY"
    BLACK_BROWN = "BLACK_BROWN"
    GREY_BROWN = "GREY_BROWN"
    DARK_BROWN = "DARK_BROWN"
    CHOCOLATE = "CHOCOLATE"
    LIGHT_GREY = "LIGHT_GREY"
    DARK_GREY = "DARK_GREY"
    GREY_BLACK = "GREY_BLACK"
    GREY_CHESTNUT = "GREY_CHESTNUT"
    GREY_ROUAN = "ROUAN_CHESTNUT"
    LOUVET = "LOUVET"
    PANGARE_NOIR = "PANDARE_NOIR"
    SABINO = "SABINO"
    MEDLEY_BAY = "MEDLEY_BAY"
    MEDLEY_SABINO = "MEDLEY_SABINO"