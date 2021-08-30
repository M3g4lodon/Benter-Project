import datetime as dt
import json
import os

import requests

import utils
from constants import PMU_DATA_DIR
from constants import Sources
from constants import UNIBET_DATA_DIR


def execute_get_query(url: str) -> dict:
    res = {"get_url": url, "query_date": dt.datetime.now().isoformat()}
    response = requests.get(url=url)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            data_json = e.response.json()
            assert data_json["label"] == "application", data_json
            assert data_json["code"] == 1, data_json
            res.update({"note": "application error"})
            return res
        except json.decoder.JSONDecodeError as e:
            res.update({"note": f"server error, no json ({e})"})
            return res

    if response.status_code == 204:
        res.update({"note": "no content"})
        return res

    data_json = response.json()
    if isinstance(data_json, list):
        data_json = {"data": data_json}
    assert "get_url" not in data_json
    assert "query_date" not in data_json
    res.update(data_json)
    return res


def create_day_folder(date: dt.date, source: Sources) -> None:
    data_dir = PMU_DATA_DIR if source == Sources.PMU else UNIBET_DATA_DIR
    if date.isoformat() in os.listdir(data_dir):
        return
    os.mkdir(os.path.join(data_dir, date.isoformat()))


def check_query_json(filename: str, url: str):
    data_json = utils.load_json(filename=filename)
    if data_json is None:
        return False

    if "get_url" not in data_json:
        return False

    if data_json["get_url"] != url:
        return False

    return True
