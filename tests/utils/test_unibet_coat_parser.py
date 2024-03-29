import pytest

from constants import UnibetCoat
from utils import unibet_coat_parser

# No clue for SM and BM
unknown_found_coats = ["4", "AT", "T"]


@pytest.mark.parametrize(
    "coat_str, unibet_coat",
    [
        (1998, UnibetCoat.UNKNOWN),
        ("", UnibetCoat.UNKNOWN),
        (" ", UnibetCoat.UNKNOWN),
        (None, UnibetCoat.UNKNOWN),
        (" Al.", UnibetCoat.CHESTNUT),
        ("ALZ", UnibetCoat.CHESTNUT),
        ("AL", UnibetCoat.CHESTNUT),
        ("Al", UnibetCoat.CHESTNUT),
        ("AL.", UnibetCoat.CHESTNUT),
        ("Al.", UnibetCoat.CHESTNUT),
        ("Palomino", UnibetCoat.PALOMINO),
        ("PALOMINO", UnibetCoat.PALOMINO),
        ("ro", UnibetCoat.ROUAN),
        ("RO", UnibetCoat.ROUAN),
        ("ROUAN", UnibetCoat.ROUAN),
        ("Rouan", UnibetCoat.ROUAN),
        ("White", UnibetCoat.WHITE),
        ("BLANC", UnibetCoat.WHITE),
        ("gr", UnibetCoat.GREY),
        ("Gr", UnibetCoat.GREY),
        ("GR", UnibetCoat.GREY),
        ("GR.", UnibetCoat.GREY),
        ("Gr.", UnibetCoat.GREY),
        ("GRIS", UnibetCoat.GREY),
        ("Gris", UnibetCoat.GREY),
        ("Gray", UnibetCoat.GREY),
        ("GREY", UnibetCoat.GREY),
        ("grey", UnibetCoat.GREY),
        ("G", UnibetCoat.GREY),
        ("G.", UnibetCoat.GREY),
        ("ch", UnibetCoat.CHESTNUT),
        ("Chestnut", UnibetCoat.CHESTNUT),
        ("CHESTNUT", UnibetCoat.CHESTNUT),
        ("chestnut", UnibetCoat.CHESTNUT),
        ("BAI.", UnibetCoat.BAY),
        ("Bay", UnibetCoat.BAY),
        ("BAY", UnibetCoat.BAY),
        ("bay", UnibetCoat.BAY),
        ("BAI", UnibetCoat.BAY),
        ("bai", UnibetCoat.BAY),
        ("Bai", UnibetCoat.BAY),
        ("B", UnibetCoat.BAY),
        ("b", UnibetCoat.BAY),
        ("B.", UnibetCoat.BAY),
        (" B.", UnibetCoat.BAY),
        ("ISABELLE", UnibetCoat.DUN),
        ("B. B.", UnibetCoat.BROWN),
        ("B.b", UnibetCoat.BROWN),
        ("B.b.", UnibetCoat.BROWN),
        ("b/br", UnibetCoat.BROWN),
        ("B/BR", UnibetCoat.BROWN),
        ("BAYBROWN", UnibetCoat.BROWN),
        ("BB", UnibetCoat.BROWN),
        ("bb.", UnibetCoat.BROWN),
        ("bo", UnibetCoat.BROWN),
        ("Bb.", UnibetCoat.BROWN),
        ("BB.", UnibetCoat.BROWN),
        ("bbr", UnibetCoat.BROWN),
        ("BR", UnibetCoat.BROWN),
        ("br", UnibetCoat.BROWN),
        ("Brown", UnibetCoat.BROWN),
        ("brown", UnibetCoat.BROWN),
        ("marron", UnibetCoat.BROWN),
        ("MO", UnibetCoat.BROWN),
        ("BAI B", UnibetCoat.BROWN),
        ("B. Fonc", UnibetCoat.DARK_BAY),
        ("B. Fonce", UnibetCoat.DARK_BAY),
        ("B. FONCE", UnibetCoat.DARK_BAY),
        ("B. Foncé", UnibetCoat.DARK_BAY),
        ("Bb. Fonce", UnibetCoat.DARK_BAY),
        ("BB. FONCE", UnibetCoat.DARK_BAY),
        ("Bb. Foncé", UnibetCoat.DARK_BAY),
        ("B. Fo", UnibetCoat.DARK_BAY),
        ("B. FO", UnibetCoat.DARK_BAY),
        ("BF", UnibetCoat.DARK_BAY),
        ("BBF", UnibetCoat.DARK_BAY),
        ("dkb", UnibetCoat.DARK_BAY),
        ("dkb/br", UnibetCoat.DARK_BAY),
        ("Bf.", UnibetCoat.DARK_BAY),
        ("B.Foncé", UnibetCoat.DARK_BAY),
        ("BAI FONCE", UnibetCoat.DARK_BAY),
        ("BB. F", UnibetCoat.DARK_BAY),
        ("Bb. F", UnibetCoat.DARK_BAY),
        ("Bl", UnibetCoat.BLACK),
        ("Black", UnibetCoat.BLACK),
        ("black", UnibetCoat.BLACK),
        ("N", UnibetCoat.BLACK),
        ("N.", UnibetCoat.BLACK),
        ("NO", UnibetCoat.BLACK),
        ("Noi.", UnibetCoat.BLACK),
        ("NOIR", UnibetCoat.BLACK),
        ("Noir", UnibetCoat.BLACK),
        ("B. Clair", UnibetCoat.LIGHT_BAY),
        ("B. CLAIR", UnibetCoat.LIGHT_BAY),
        ("Bai C", UnibetCoat.LIGHT_BAY),
        ("BAI CLAIR", UnibetCoat.LIGHT_BAY),
        ("BAI CR", UnibetCoat.LIGHT_BAY),
        ("BC", UnibetCoat.LIGHT_BAY),
        ("B. CR", UnibetCoat.LIGHT_BAY),
        ("A.", UnibetCoat.CHESTNUT),
        ("AA", UnibetCoat.STRAWBERRY_ROUAN),
        ("AB", UnibetCoat.LIVER_CHESTNUT),
        ("AC", UnibetCoat.LIGHT_CHESTNUT),
        ("AF", UnibetCoat.DARK_CHESTNUT),
        ("Al. A", UnibetCoat.STRAWBERRY_ROUAN),
        ("Al. Aub.", UnibetCoat.STRAWBERRY_ROUAN),
        ("AL. B", UnibetCoat.LIVER_CHESTNUT),
        ("Al. B", UnibetCoat.LIVER_CHESTNUT),
        ("AL. BRULE", UnibetCoat.LIVER_CHESTNUT),
        ("Al. Brule", UnibetCoat.LIVER_CHESTNUT),
        ("Al. C", UnibetCoat.LIGHT_CHESTNUT),
        ("AL. C", UnibetCoat.LIGHT_CHESTNUT),
        ("Al. Clair", UnibetCoat.LIGHT_CHESTNUT),
        ("Al. Cr.Lav", UnibetCoat.FLAXEN_CHESTNUT),
        ("AL. CR.LAV", UnibetCoat.FLAXEN_CHESTNUT),
        ("AL. CU", UnibetCoat.COPPER_CHESTNUT),
        ("Al. Cuivre", UnibetCoat.COPPER_CHESTNUT),
        ("AL. F", UnibetCoat.DARK_CHESTNUT),
        ("Al. F", UnibetCoat.DARK_CHESTNUT),
        ("AL. FONCE", UnibetCoat.DARK_CHESTNUT),
        ("Al. Fonce", UnibetCoat.DARK_CHESTNUT),
        ("AL. P", UnibetCoat.PIEBALD_CHESTNUT),
        ("AL. PIE", UnibetCoat.PIEBALD_CHESTNUT),
        ("Al. R", UnibetCoat.RUBICANO_CHESTNUT),
        ("Al. Rub.", UnibetCoat.RUBICANO_CHESTNUT),
        ("AL.MEL.", UnibetCoat.MEDLEY_CHESTNUT),
        ("AL.MELANGE", UnibetCoat.MEDLEY_CHESTNUT),
        ("AM", UnibetCoat.MEDLEY_CHESTNUT),
        ("AR", UnibetCoat.RUBICANO_CHESTNUT),
        ("AU", UnibetCoat.AUBERE),
        ("AUB.", UnibetCoat.AUBERE),
        ("Aub.", UnibetCoat.AUBERE),
        ("AUBERE", UnibetCoat.AUBERE),
        ("B. Ce", UnibetCoat.CHERRY_BAY),
        ("B. Cerise", UnibetCoat.CHERRY_BAY),
        ("B. CL", UnibetCoat.LIGHT_BAY),
        ("B. Cl", UnibetCoat.LIGHT_BAY),
        ("B. Rubican", UnibetCoat.RUBICANO_BAY),
        ("B.d", UnibetCoat.DARK_BAY),
        ("B. Ru", UnibetCoat.RUBICANO_BAY),
        ("b/gr", UnibetCoat.GREY_BAY),
        ("Bai M", UnibetCoat.MEDLEY_BAY),
        ("Bai P", UnibetCoat.PIEBALD_BAY),
        ("BAI PIE", UnibetCoat.PIEBALD_BAY),
        ("BG", UnibetCoat.GREY_BAY),
        ("bl/br", UnibetCoat.BLACK_BROWN),
        ("br/gr", UnibetCoat.GREY_BROWN),
        ("BRF", UnibetCoat.DARK_BROWN),
        ("BROWNGREY", UnibetCoat.GREY_BROWN),
        ("CHOCO", UnibetCoat.CHOCOLATE),
        ("Choco", UnibetCoat.CHOCOLATE),
        ("CHOCOLAT", UnibetCoat.CHOCOLATE),
        ("Chocolat", UnibetCoat.CHOCOLATE),
        ("GB", UnibetCoat.GREY_BAY),
        ("GC", UnibetCoat.LIGHT_GREY),
        ("GF", UnibetCoat.DARK_GREY),
        ("GR. F", UnibetCoat.DARK_GREY),
        ("grey-bay", UnibetCoat.GREY_BAY),
        ("grey-blk", UnibetCoat.GREY_BLACK),
        ("grey-brown", UnibetCoat.GREY_BROWN),
        ("grey-chest", UnibetCoat.GREY_CHESTNUT),
        ("GREYROAN", UnibetCoat.GREY_ROUAN),
        ("Louve", UnibetCoat.LOUVET),
        ("Noi.P", UnibetCoat.PANGARE_NOIR),
        ("NOI.P", UnibetCoat.PANGARE_NOIR),
        ("Noi.Pan", UnibetCoat.PANGARE_NOIR),
        ("NOI.PAN", UnibetCoat.PANGARE_NOIR),
        ("R.", UnibetCoat.ROUAN),
        ("RG", UnibetCoat.GREY_ROUAN),
        ("s", UnibetCoat.SABINO),
        ("S", UnibetCoat.SABINO),
        ("BM", UnibetCoat.MEDLEY_BAY),
        ("SM", UnibetCoat.MEDLEY_SABINO),
    ]
    + [(coat_str, UnibetCoat.UNKNOWN) for coat_str in unknown_found_coats],
)
def test_get_coat(coat_str, unibet_coat):
    assert unibet_coat_parser.get_coat(coat=coat_str) == unibet_coat
