from constants import UnibetHorseSex


def test_unibet_sex():
    assert UnibetHorseSex("H") == UnibetHorseSex.GELDING
    assert UnibetHorseSex(None) == UnibetHorseSex.UNKNOWN


def test_unibet_sex_is_born_male():
    assert UnibetHorseSex.MALE.is_born_male
    assert UnibetHorseSex.GELDING.is_born_male
    assert not UnibetHorseSex.FEMALE.is_born_male
    assert UnibetHorseSex.UNKNOWN.is_born_male is None
