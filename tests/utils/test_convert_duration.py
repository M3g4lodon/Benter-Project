import pytest

import utils


@pytest.mark.parametrize(
    "duration_str, duration_value",
    [(None, None), ("", None), ("""3'32"91""", 212.91), ("""3'32''91""", 212.91)],
)
def test_convert_duration_in_sec(duration_str, duration_value):
    assert utils.convert_duration_in_sec(time_str=duration_str) == duration_value
