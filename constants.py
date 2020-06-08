import collections
import datetime as dt
import os

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
