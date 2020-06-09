import datetime as dt
import json
import os
from typing import Optional

from constants import SOURCES, SOURCE_PMU, PMU_DATA_DIR, SOURCE_Unibet, UNIBET_DATA_DIR


def dump_json(data: dict, filename: str) -> None:
    assert filename.endswith('.json')
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def load_json(filename: str) -> Optional[dict]:
    assert filename.endswith('.json')
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as fp:
        return json.load(fp=fp)


def get_folder_path(source:str, date:dt.date)->str:
    assert source in SOURCES
    if source == SOURCE_PMU:
        return os.path.join(PMU_DATA_DIR, date.isoformat())
    if source == SOURCE_Unibet:
        return os.path.join(UNIBET_DATA_DIR, date.isoformat())