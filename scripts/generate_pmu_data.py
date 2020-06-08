import datetime as dt
import os
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from constants import PMU_DATA_DIR, DATA_DIR
from utils import features

# TODO investigate driverchange column
# TODO investigate 'engagement' column


def get_num_pmu_from_name(
    horse_name: str, participants: Optional[dict]
) -> Optional[int]:
    if not participants:
        return None
    horse_ = [part for part in participants if part["nom"] == horse_name]
    if len(horse_) != 1:
        return None
    horse = horse_[0]
    if "numPmu" not in horse:
        return None
    return horse["numPmu"]


def get_last_race_date_from_performance(
    performance_json: dict, participants=None
) -> Optional[dict]:
    if "participants" not in performance_json:
        return None

    res = {}
    for perf in performance_json["participants"]:
        if "coursesCourues" not in perf:
            continue
        if "numPmu" not in perf:
            if "nomCheval" not in perf:
                continue
            num_pmu = get_num_pmu_from_name(
                horse_name=perf["nomCheval"], participants=participants
            )

        else:
            num_pmu = perf["numPmu"]
        if num_pmu is None:
            continue

        if not perf["coursesCourues"]:
            continue
        dates = [
            course["date"] for course in perf["coursesCourues"] if "date" in course
        ]
        if not dates:
            continue
        max_date = max(dates)

        res[num_pmu] = dt.date.fromtimestamp(max_date / 1000)
    if not res:
        return None
    return res


def get_num_pmu_enjeu_from_citations(
    citations: dict, pari_type: str, num_pmu_partants: Optional[List[int]] = None
) -> Optional[dict]:
    if "listeCitations" not in citations:
        return None
    citations_ = [
        citation
        for citation in citations["listeCitations"]
        if citation["typePari"] == pari_type
    ]
    assert len(citations_) <= 1
    if not citations_:
        return None
    citation = citations_[0]
    if "participants" not in citation:
        return None

    if num_pmu_partants is None:
        return {
            part["numPmu"]: part["citations"][0]["enjeu"]
            for part in citation["participants"]
        }
    return {
        part["numPmu"]: part["citations"][0]["enjeu"]
        for part in citation["participants"]
        if part["numPmu"] in num_pmu_partants
    }


def get_num_pmu_enjeu_from_combinaisons(
    combinaisons: dict, pari_type: str, num_pmu_partants: Optional[List[int]] = None
) -> Optional[dict]:
    if "combinaisons" not in combinaisons:
        return None
    _combs = [
        comb for comb in combinaisons["combinaisons"] if comb["pariType"] == pari_type
    ]
    assert len(_combs) <= 1
    if not _combs:
        return None
    combinaison = _combs[0]
    if "listeCombinaisons" not in combinaison:
        return None
    if not all("combinaison" in comb for comb in combinaison["listeCombinaisons"]):
        return None

    if not all("totalEnjeu" in comb for comb in combinaison["listeCombinaisons"]):
        return None

    if num_pmu_partants is None:
        return {
            comb["combinaison"][0]: comb["totalEnjeu"]
            for comb in combinaison["listeCombinaisons"]
        }
    return {
        comb["combinaison"][0]: comb["totalEnjeu"]
        for comb in combinaison["listeCombinaisons"]
        if comb["combinaison"][0] in num_pmu_partants
    }


def get_num_pmu_enjeu_from_rapport_simple_enjeux(
    rapport_simple_gagnant: dict,
    enjeux: dict,
    pari_type: str,
    num_pmu_partants: Optional[List[int]] = None,
) -> Optional[dict]:
    if "rapportsParticipants" not in rapport_simple_gagnant:
        return None

    if "data" not in enjeux:
        return None

    if not all(
        "numPmu" in rapport and "rapportDirect" in rapport
        for rapport in rapport_simple_gagnant["rapportsParticipant"]
    ):
        return None
    if not all("data" in enjeu and "typesParis" in enjeu for enjeu in enjeux["data"]):
        return None

    mutual_proba = {
        rapport["numPmu"]: 1 / rapport["rapportDirect"]
        for rapport in rapport_simple_gagnant["rapportsParticipant"]
    }
    enjeu_ = [enjeu for enjeu in enjeux["data"] if pari_type in enjeu["typesParis"]]
    if not enjeu_:
        return None
    assert len(enjeu_) == 1
    total_enjeu = enjeu_[0]["totalEnjeu"]

    correction_coefficient = sum(
        proba for num_pmu, proba in mutual_proba.items() if num_pmu in num_pmu_partants
    )

    return {
        num_pmu: total_enjeu * proba / correction_coefficient
        for num_pmu, proba in mutual_proba.items()
        if num_pmu in num_pmu_partants
    }


def get_penetrometer_value(course: dict) -> Optional[float]:
    if "penetrometre" not in course:
        return None
    if "valeurMesure" not in course["penetrometre"]:
        return None
    penetrometer_value: str = course["penetrometre"]["valeurMesure"]
    penetrometer_value = penetrometer_value.replace(",", ".")
    return float(penetrometer_value)


def load_queried_data() -> pd.DataFrame:
    race_records = []

    race_count = 0
    for date in tqdm(
        iterable=os.listdir(PMU_DATA_DIR), desc="Loading races per date", unit="day"
    ):
        if date == ".ipynb_checkpoints":
            continue

        assert re.match(r"\d{4}-\d{2}-\d{2}", date)

        folder_path = os.path.join(PMU_DATA_DIR, date)

        programme_json = utils.load_json(
            filename=os.path.join(folder_path, "programme.json")
        )

        if (
            programme_json is None
            or "programme" not in programme_json
            or "reunions" not in programme_json["programme"]
        ):
            continue

        for reunion in programme_json["programme"]["reunions"]:
            for course in reunion["courses"]:
                if course["statut"] in [
                    "COURSE_ANNULEE",
                    "PROGRAMMEE",
                    "ARRIVEE_PROVISOIRE",
                    "COURSE_ANNULEE",
                    "DEPART_CONFIRME",
                    "DEPART_DANS_TROIS_MINUTES",
                ]:
                    continue
                assert course["statut"] in [
                    "FIN_COURSE",
                    "ARRIVEE_DEFINITIVE",
                    "ARRIVEE_DEFINITIVE_COMPLETE",
                    "COURSE_ARRETEE",
                ], course["statut"]
                r_i = course["numReunion"]
                c_i = course["numOrdre"]

                course_race_datetime = dt.datetime.fromtimestamp(
                    course["heureDepart"] / 1000,
                    tz=dt.timezone(dt.timedelta(milliseconds=course["timezoneOffset"])),
                )
                participants_ = utils.load_json(
                    filename=os.path.join(
                        folder_path, f"R{r_i}_C{c_i}_participants.json"
                    )
                )
                if participants_ is None or "participants" not in participants_:
                    continue
                participants_ = participants_["participants"]
                participants = [
                    {k: v for k, v in part.items() if not isinstance(v, dict)}
                    for part in participants_
                ]
                performance_json = utils.load_json(
                    filename=os.path.join(
                        folder_path, f"R{r_i}_C{c_i}_performance.json"
                    )
                )
                last_race_date = get_last_race_date_from_performance(
                    performance_json=performance_json, participants=participants
                )

                num_pmu_partants = [
                    part["numPmu"]
                    for part in participants
                    if part["statut"] == "PARTANT"
                ]

                pari_type = (
                    "E_SIMPLE_GAGNANT" if course["hasEParis"] else "SIMPLE_GAGNANT"
                )

                citations = utils.load_json(
                    filename=os.path.join(
                        folder_path,
                        f"R{r_i}_C{c_i}_{'CITATIONS_INTERNET'.lower() if course['hasEParis'] else 'CITATIONS'.lower()}.json",
                    )
                )
                num_pmu_enjeu = get_num_pmu_enjeu_from_citations(
                    citations=citations,
                    pari_type=pari_type,
                    num_pmu_partants=num_pmu_partants,
                )
                is_true_total_enjeu = True

                if len(num_pmu_partants) <= 12 and num_pmu_enjeu is None:
                    combinaisons = utils.load_json(
                        filename=os.path.join(
                            folder_path,
                            f"R{r_i}_C{c_i}_{'COMBINAISONS_INTERNET'.lower() if course['hasEParis'] else 'COMBINAISONS'.lower()}.json",
                        )
                    )

                    num_pmu_enjeu = get_num_pmu_enjeu_from_combinaisons(
                        combinaisons=combinaisons,
                        pari_type=pari_type,
                        num_pmu_partants=num_pmu_partants,
                    )

                if num_pmu_enjeu is None:
                    rapport_simple_gagnant = utils.load_json(
                        filename=os.path.join(
                            folder_path,
                            f"R{r_i}_C{c_i}_{pari_type}_{'RAPPORTS'.lower()}.json",
                        )
                    )

                    enjeux = utils.load_json(
                        filename=os.path.join(
                            folder_path,
                            f"R{r_i}_C{c_i}_{'ENJEU_INTERNET'.lower() if course['hasEParis'] else 'ENJEU'.lower()}.json",
                        )
                    )
                    num_pmu_enjeu = get_num_pmu_enjeu_from_rapport_simple_enjeux(
                        rapport_simple_gagnant=rapport_simple_gagnant,
                        enjeux=enjeux,
                        pari_type=pari_type,
                        num_pmu_partants=num_pmu_partants,
                    )
                    is_true_total_enjeu = num_pmu_enjeu is None
                penetrometer_value = get_penetrometer_value(cours=course)
                course_incidents = course["incidents"] if "incidents" in course else []
                incident_nums = {
                    num_part
                    for incident in course_incidents
                    for num_part in incident["numeroParticipants"]
                }
                for part, part_ in zip(participants, participants_):
                    # TODO integrate other dict keys
                    # dict key found {'commentaireApresCourse',
                    #  'dernierRapportDirect',
                    #  'dernierRapportReference',
                    #  'distanceChevalPrecedent',
                    #  'gainsParticipant', # added here
                    #  'robe'}
                    if "gainsParticipant" in part_:
                        part.update(part_["gainsParticipant"])
                    part["n_reunion"] = r_i
                    part["n_course"] = c_i
                    part["date"] = date
                    part["race_datetime"] = course_race_datetime
                    part["in_incident"] = part["numPmu"] in incident_nums
                    part["incident_type"] = (
                        None
                        if part["numPmu"] not in incident_nums
                        else [
                            incident["type"]
                            for incident in course_incidents
                            if part["numPmu"] in incident["numeroParticipants"]
                        ][0]
                    )
                    part["totalEnjeu"] = (
                        None
                        if num_pmu_enjeu is None
                        else num_pmu_enjeu.get(part["numPmu"], None)
                    )

                    part["last_race_date"] = (
                        None
                        if last_race_date is None
                        else last_race_date.get(part["numPmu"], None)
                    )
                    handicap_weight = None
                    if "poidsConditionMonte" in part and part["poidsConditionMonte"]:
                        handicap_weight = part["poidsConditionMonte"]
                    elif "handicapPoids" in part and part["handicapPoids"]:
                        handicap_weight = part["handicapPoids"]

                    part["handicap_weight"] = handicap_weight  # in hectogram
                    part["is_true_total_enjeu"] = is_true_total_enjeu
                    part["reunion_nature"] = reunion["nature"]
                    part["reunion_audience"] = reunion["audience"]
                    part["reunion_pays"] = reunion["pays"]["code"]
                    part["course_statut"] = course["statut"]
                    part["course_discipline"] = course["discipline"]
                    part["course_specialite"] = course["specialite"]
                    part["course_condition_sexe"] = course["conditionSexe"]
                    part["course_condition_age"] = (
                        None if "conditionAge" not in course else course["conditionAge"]
                    )
                    part["course_track_type"] = (
                        None if "typePiste" not in course else course["typePiste"]
                    )

                    part["course_penetrometre"] = penetrometer_value
                    # TODO Deal with ecurie
                    part["course_corde"] = (
                        None if "corde" not in course else course["corde"]
                    )
                    part["course_hippodrome"] = (
                        None
                        if "hippodrome" not in course
                        else course["hippodrome"]["codeHippodrome"]
                    )
                    part["course_parcours"] = (
                        None if "parcours" not in course else course["parcours"]
                    )
                    part["course_distance"] = (
                        None if "distance" not in course else course["distance"]
                    )
                    part["course_distance_unit"] = (
                        None if "distanceUnit" not in course else course["distanceUnit"]
                    )
                    part["course_duration"] = (
                        None if "dureeCourse" not in course else course["dureeCourse"]
                    )
                    part["course_prize_pool"] = (
                        None
                        if "montantTotalOffert" not in course
                        else course["montantTotalOffert"]
                    )  # TODO look at difference with "montantPrix"
                    part["course_winner_prize"] = (
                        None
                        if "montantOffert1er" not in course
                        else course["montantOffert1er"]
                    )
                race_records.extend(participants)
                race_count += 1

    return pd.DataFrame.from_records(data=race_records)


def _create_horse_name_mapper(rh_df: pd.DataFrame) -> Dict[str, int]:
    assert "horse_name" in rh_df
    assert "father_horse_name" in rh_df
    assert "mother_horse_name" in rh_df
    assert "father_mother_horse_name" in rh_df

    horse_names = rh_df["horse_name"].append(
        (
            rh_df["father_horse_name"],
            rh_df["mother_horse_name"],
            rh_df["father_mother_horse_name"],
        )
    )
    horse_names = horse_names.str.upper()
    horse_names.dropna(inplace=True)
    horse_names = pd.Series(horse_names.unique())
    horse_names.name = "horse_name"
    mapper_df = horse_names.reset_index().set_index("horse_name")
    return mapper_df.to_dict()["index"]


def convert_queried_data_to_race_horse_df(
    queried_race_horse_df: pd.DataFrame,
    historical_race_horse_df: Optional[pd.DataFrame],
):
    """Will convert queried race horse df into the right format.

    If `historical_race_horse_df`is provided, the converted race/horses will be formatted as an extension of it """

    race_horse_df = queried_race_horse_df
    for col_name in ["nomPereMere", "ordreArrivee"]:
        if col_name not in race_horse_df:
            race_horse_df[col_name] = np.nan

    columns_renaming = {
        "nom": "horse_name",
        "nomPere": "father_horse_name",
        "nomMere": "mother_horse_name",
        "nomPereMere": "father_mother_horse_name",
        "sexe": "horse_sex",
        "numPmu": "horse_number",
        "driver": "jockey_name",
        "age": "horse_age",
        "race": "horse_race",
        "jumentPleine": "is_pregnant",
        "entraineur": "trainer_name",
        "proprietaire": "owner_name",
        "eleveur": "breeder_name",
        "ordreArrivee": "horse_place",
        "nombreCourses": "n_run_races",
        "nombreVictoires": "n_won_races",
        "nombrePlaces": "n_placed_races",
        "deferre": "unshod",
        "tempsObtenu": "horse_race_duration",
        "gainsAnneeEnCours": "horse_current_year_prize",
        "gainsAnneePrecedente": "horse_last_year_prize",
        "gainsCarriere": "horse_career_prize",
        "oeilleres": "blinkers",
        "handicapDistance": "handicap_distance",
        "handicapValeur": "handicap_value",
    }
    race_horse_df = queried_race_horse_df.rename(columns=columns_renaming)

    race_horse_df[race_horse_df["unshod"] == "REFERRE_ANTERIEURS_POSTERIEURS"][
        "unshod"
    ] = np.nan

    # Compute horse_id
    # TODO check unique mother/father/father_mother

    mapper_dict = {}
    max_old_race_id = -1

    if historical_race_horse_df is not None:
        for col in ["horse", "father_horse", "mother_horse", "father_mother_horse"]:
            col_id = f"{col}_id"
            col_name = f"{col}_name"
            horse_name_df = historical_race_horse_df[
                [col_id, col_name]
            ].drop_duplicates()
            horse_name_df.dropna(inplace=True)
            horse_name_df[col_name] = horse_name_df[col_name].str.upper()
            horse_name_df.drop_duplicates(inplace=True)
            mapper_dict.update(horse_name_df.set_index(col_name)[col_id].to_dict())

        max_old_race_id = max(mapper_dict.values())

    mapper_dict.update(
        {
            horse_name: index + max_old_race_id + 1
            for horse_name, index in _create_horse_name_mapper(
                rh_df=race_horse_df
            ).items()
            if horse_name not in mapper_dict
        }
    )

    race_horse_df["horse_id"] = (
        race_horse_df["horse_name"].str.upper().map(mapper_dict, na_action="ignore")
    )
    race_horse_df["father_horse_id"] = (
        race_horse_df["father_horse_name"]
        .str.upper()
        .map(mapper_dict, na_action="ignore")
    )
    race_horse_df["mother_horse_id"] = (
        race_horse_df["mother_horse_name"]
        .str.upper()
        .map(mapper_dict, na_action="ignore")
    )
    if race_horse_df["father_mother_horse_name"].isna().all():
        race_horse_df["father_mother_horse_id"] = np.nan
    else:
        race_horse_df["father_mother_horse_id"] = (
            race_horse_df["father_mother_horse_name"]
            .str.upper()
            .map(mapper_dict, na_action="ignore")
        )

    # Compute race_id
    mapper_dict = {}
    max_old_race_id = -1
    if historical_race_horse_df is not None:
        horse_name_df = historical_race_horse_df[
            ["date", "n_reunion", "n_course", "race_id"]
        ].drop_duplicates()
        horse_name_df.dropna(inplace=True)
        mapper_dict.update(
            horse_name_df.set_index(["date", "n_reunion", "n_course"])[
                "race_id"
            ].to_dict()
        )

        max_old_race_id = max(mapper_dict.values())

    drc_df = race_horse_df[["date", "n_reunion", "n_course"]]
    mapper_df = drc_df.drop_duplicates().reset_index(drop=True)
    mapper_dict.update(
        {
            (row.date, row.n_reunion, row.n_course): index + max_old_race_id + 1
            for index, row in mapper_df.iterrows()
            if (row.date, row.n_reunion, row.n_course) not in mapper_dict
        }
    )
    race_horse_df["race_id"] = [
        mapper_dict[drc] for drc in drc_df.itertuples(index=False)
    ]
    race_horse_df["n_horses"] = race_horse_df["race_id"].map(
        race_horse_df.groupby("race_id")["statut"].agg(lambda s: (s == "PARTANT").sum())
    )

    # Compute odds and pari_mutuel_proba
    race_horse_df["totalEnjeu"] = race_horse_df["totalEnjeu"]  # in cents
    sum_per_race = race_horse_df.groupby("race_id")["totalEnjeu"].sum()

    race_horse_df["pari_mutuel_proba"] = race_horse_df["totalEnjeu"] / race_horse_df[
        "race_id"
    ].map(sum_per_race)
    race_horse_df["odds"] = 1 / race_horse_df["pari_mutuel_proba"]

    return race_horse_df


def check_df(featured_race_horse_df: pd.DataFrame):
    assert not featured_race_horse_df["odds"].isna().all()


def run():
    queried_race_horse_df = load_queried_data()

    race_horse_df = convert_queried_data_to_race_horse_df(
        queried_race_horse_df=queried_race_horse_df, historical_race_horse_df=None
    )

    # Compute features
    featured_race_horse_df = features.append_features(
        race_horse_df=race_horse_df, historical_race_horse_df=race_horse_df
    )

    featured_race_horse_df["duration_since_last_race"] = (
        pd.to_datetime(featured_race_horse_df["date"]).dt.date
        - pd.to_datetime(featured_race_horse_df["last_race_date"]).dt.date
    )

    check_df(featured_race_horse_df=featured_race_horse_df)

    featured_race_horse_df.to_csv(
        os.path.join(DATA_DIR, "pmu_data_with_features.csv"), index=False
    )


if __name__ == "__main__":
    run()
