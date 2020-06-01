import json
import os
from typing import Optional


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
