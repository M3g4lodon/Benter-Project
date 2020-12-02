import datetime as dt
import json
import os
import re
from typing import Generator
from typing import Optional

from constants import PMU_DATA_DIR
from constants import SOURCE_PMU
from constants import SOURCE_Unibet
from constants import SOURCES
from constants import UNIBET_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


def dump_json(data: dict, filename: str) -> None:
    assert filename.endswith(".json")
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


def load_json(filename: str) -> Optional[dict]:
    assert filename.endswith(".json")
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as fp:
        return json.load(fp=fp)


def get_folder_path(source: str, date: dt.date) -> Optional[str]:
    assert source in SOURCES
    if source == SOURCE_PMU:
        return os.path.join(PMU_DATA_DIR, date.isoformat())
    if source == SOURCE_Unibet:
        return os.path.join(UNIBET_DATA_DIR, date.isoformat())
    return None


def date_countdown_generator(
    start_date: dt.date, end_date: Optional[dt.date]
) -> Generator[dt.date, None, None]:
    end_date = end_date or dt.date.today()
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += dt.timedelta(days=1)


def convert_duration_in_sec(time_str: Optional[str]) -> Optional[float]:
    if not time_str:
        return None

    matches = re.match(r"(\d{1,2})'(\d{2})''(\d{2})", time_str)
    matches = (
        re.match(r"""(\d{1,2})'(\d{2})"(\d{2})""", time_str)
        if matches is None
        else matches
    )
    if not matches:
        logger.warning("Could not convert %s in duration", time_str)
        return None

    n_min, n_sec, n_cs = matches.groups()
    return 60 * int(n_min) + int(n_sec) + 0.01 * int(n_cs)
