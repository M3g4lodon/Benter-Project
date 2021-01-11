import itertools
from typing import Optional

import utils
from constants import UnibetCoat

logger = utils.setup_logger(name=__name__)

dark_suffices = ["f", "fo", "f.", "fo.", "fonc", "fonce", "foncé", "d,", "d", "dark"]
light_suffices = ["c", "c.", "cl", "cl.", "cr", "cr.", "clair"]
bay_prefices = ["b", "b.", "bb", "bb.", "bai", "bay"]
brown_prefices = ["br", "br.", "brown"]
grey_prefices = ["g", "g.", "gr", "gr.", "gray", "grey"]
join_str = ["", " "]

generated_dark_bay_patterns = [
    f"{a}{b}{c}"
    for (a, b, c) in itertools.product(bay_prefices, join_str, dark_suffices)
]

generated_light_bay_patterns = [
    f"{a}{b}{c}"
    for (a, b, c) in itertools.product(bay_prefices, join_str, light_suffices)
]

generated_dark_brown_patterns = [
    f"{a}{b}{c}"
    for (a, b, c) in itertools.product(brown_prefices, join_str, dark_suffices)
]

generated_dark_grey_patterns = [
    f"{a}{b}{c}"
    for (a, b, c) in itertools.product(grey_prefices, join_str, dark_suffices)
]
generated_light_grey_patterns = [
    f"{a}{b}{c}"
    for (a, b, c) in itertools.product(grey_prefices, join_str, light_suffices)
]


def get_coat(coat: Optional[str]) -> UnibetCoat:

    if coat is None:
        return UnibetCoat.UNKNOWN

    if not isinstance(coat, str):
        logger.warning('Can not parse coat "%s" of type "%s"', coat, type(coat))
        return UnibetCoat.UNKNOWN

    lowered_stripped_coat = coat.lower().strip()

    if lowered_stripped_coat == "":
        return UnibetCoat.UNKNOWN

    if lowered_stripped_coat == "palomino":
        return UnibetCoat.PALOMINO

    if lowered_stripped_coat in ["r", "r.", "ro", "ro.", "rouan"]:
        return UnibetCoat.ROUAN

    if lowered_stripped_coat in [
        "bl",
        "bl.",
        "black",
        "no",
        "n",
        "n.",
        "no.",
        "noi.",
        "noi",
        "noir",
    ]:
        return UnibetCoat.BLACK

    if lowered_stripped_coat in ["white", "blanc"]:
        return UnibetCoat.WHITE

    if lowered_stripped_coat in ["g", "g.", "grey", "gray", "gr", "gr.", "gris"]:
        return UnibetCoat.GREY

    if lowered_stripped_coat in [
        "a",
        "a.",
        "al",
        "al.",
        "alz",
        "ale",
        "ch",
        "ch.",
        "chestnut",
        "alezan",
    ]:
        return UnibetCoat.CHESTNUT

    if lowered_stripped_coat == "isabelle":
        return UnibetCoat.DUN

    if lowered_stripped_coat in ["bai", "bai.", "bay", "b", "b."]:
        return UnibetCoat.BAY

    if lowered_stripped_coat in [
        "bb",
        "bb.",
        "br",
        "br.",
        "marron",
        "mo",
        "b. b.",
        "b.b",
        "b.b.",
        "b. b",
        "b b",
        "b/br",
        "bbr",
        "bai b",
        "baibrown",
        "baybrown",
        "brown",
    ]:
        return UnibetCoat.BROWN

    if lowered_stripped_coat in generated_dark_bay_patterns + ["dkb", "dkb/br"]:
        return UnibetCoat.DARK_BAY

    if lowered_stripped_coat in generated_light_bay_patterns:
        return UnibetCoat.LIGHT_BAY

    if lowered_stripped_coat in ["bo", "bo.", "brown", "marron"]:
        return UnibetCoat.BROWN

    if lowered_stripped_coat in [
        "aa",
        "al. a",
        "al. aub",
        "al aub.",
        "al. aub.",
        "al. aubere",
        "al aubere",
        "alezan aubere",
    ]:
        return UnibetCoat.STRAWBERRY_ROUAN

    if lowered_stripped_coat in [
        "aa",
        "alezan brule",
        "alezan brulé",
        "al brule",
        "al. brule",
        "al brulé",
        "al. brule",
        "ab",
        "ab.",
        "al. b",
        "al b",
    ]:
        return UnibetCoat.LIVER_CHESTNUT

    if lowered_stripped_coat in [
        "ac",
        "ac.",
        "al. c",
        "al. c.",
        "al. clair",
        "al clair",
        "alezan clair",
    ]:
        return UnibetCoat.LIGHT_CHESTNUT

    if lowered_stripped_coat in [
        "af",
        "af.",
        "al. f",
        "al f",
        "al. f.",
        "al. fonce",
        "al foncé",
        "al foncé",
        "al. foncé",
        "alezan fonce",
        "alezan foncé",
    ]:
        return UnibetCoat.DARK_CHESTNUT

    if lowered_stripped_coat in ["al. cr.lav", "al. cr. lav.", "alezan clair lavé"]:
        return UnibetCoat.FLAXEN_CHESTNUT

    if lowered_stripped_coat in [
        "al. cu.",
        "al. cu",
        "al cu",
        "al. cuivre",
        "al cuivre",
        "alezan cuivre",
    ]:
        return UnibetCoat.COPPER_CHESTNUT

    if lowered_stripped_coat in [
        "al p",
        "al. p",
        "al pie",
        "al. pie",
        "alezan piebald",
    ]:
        return UnibetCoat.PIEBALD_CHESTNUT

    if lowered_stripped_coat in [
        "ar",
        "al. r",
        "al r",
        "al. rub.",
        "al. rub",
        "alezan rubincano",
    ]:
        return UnibetCoat.RUBICANO_CHESTNUT

    if lowered_stripped_coat in [
        "b.r.",
        "b.r",
        "b. ru",
        "b ru",
        "b. rubican",
        "bay rubicano",
    ]:
        return UnibetCoat.RUBICANO_BAY

    if lowered_stripped_coat in [
        "am",
        "a.m.",
        "a. m",
        "al.mel.",
        "al mel",
        "al.melange",
        "alezan mélangé",
    ]:
        return UnibetCoat.MEDLEY_CHESTNUT

    if lowered_stripped_coat in ["au", "au.", "aub", "aub.", "aubere"]:
        return UnibetCoat.AUBERE

    if lowered_stripped_coat in [
        "b. ce",
        "b ce",
        "bay cerise",
        "bay ce",
        "b. cerise",
        "b cerise",
        "bay cherry",
    ]:
        return UnibetCoat.CHERRY_BAY

    if lowered_stripped_coat in [
        "bg",
        "gb",
        "b/gr",
        "b. gr",
        "bay grey",
        "grey-bay",
        "gray-bai",
    ]:
        return UnibetCoat.GREY_BAY

    if lowered_stripped_coat in ["bai m", "bm", "b. m", "bay m"]:
        return UnibetCoat.MEDLEY_BAY

    if lowered_stripped_coat in [
        "bai p",
        "bay p",
        "b. pie",
        "b pie",
        "b. p",
        "b. p.",
        "bay pie",
        "bai pie",
        "bai piebalc",
    ]:
        return UnibetCoat.PIEBALD_BAY

    if lowered_stripped_coat in ["bl/br", "bl br", "bl. br"]:
        return UnibetCoat.BLACK_BROWN

    if lowered_stripped_coat in [
        "br/gr",
        "br gr",
        "br. gr.",
        "brown grey",
        "browngrey",
    ]:
        return UnibetCoat.GREY_BROWN

    if lowered_stripped_coat in generated_dark_brown_patterns:
        return UnibetCoat.DARK_BROWN

    if lowered_stripped_coat in ["choco", "chocolat", "chocolate"]:
        return UnibetCoat.CHOCOLATE

    if lowered_stripped_coat in generated_light_grey_patterns:
        return UnibetCoat.LIGHT_GREY

    if lowered_stripped_coat in generated_dark_grey_patterns:
        return UnibetCoat.DARK_GREY

    if lowered_stripped_coat in ["s", "s.", "sabino"]:
        return UnibetCoat.SABINO

    if lowered_stripped_coat in ["louve", "louvet"]:
        return UnibetCoat.LOUVET

    if lowered_stripped_coat in [
        "noi.p",
        "noi p",
        "noir pangare",
        "noir p.",
        "noi.pan",
        "noi pan",
    ]:
        return UnibetCoat.PANGARE_NOIR

    if lowered_stripped_coat == "sm":
        return UnibetCoat.MEDLEY_SABINO

    if lowered_stripped_coat in [
        "rg",
        "rg.",
        "greyroan",
        "gray roan",
        "grey roan",
        "grayroan",
    ]:
        return UnibetCoat.GREY_ROUAN

    if lowered_stripped_coat in ["grey-chest", "gc", "gc.", "grey chestnut"]:
        return UnibetCoat.GREY_CHESTNUT

    if lowered_stripped_coat in ["grey-brown", "grey brown", "gray brown", "gr. br"]:
        return UnibetCoat.GREY_BROWN

    if lowered_stripped_coat in ["grey-blk", "gr b", "gr. b", "bgr"]:
        return UnibetCoat.GREY_BLACK

    return UnibetCoat.UNKNOWN
