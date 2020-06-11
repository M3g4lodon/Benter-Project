import datetime as dt
import os
from typing import List
from typing import Optional
from typing import Union

import utils
from constants import SOURCE_PMU
from utils.scrape import execute_get_query


def get_pmu_api_url(
    url_name: str,
    date: Union[dt.datetime, dt.date],
    r_i: Optional[int] = None,
    c_i: Optional[int] = None,
    code_pari: Optional[str] = None,
) -> str:
    pmu_programme_api_url = (
        f"https://online.turfinfo.api.pmu.fr/rest/client/1/"
        f"programme/{date.strftime('%d%m%Y')}"
    )
    if url_name == "PROGRAMME":
        return pmu_programme_api_url

    assert r_i is not None
    assert c_i is not None

    assert r_i > 0
    assert c_i > 0

    _pmu_programme_course_api_url = f"{pmu_programme_api_url}/R{r_i}/C{c_i}"

    if url_name == "PARTICIPANTS":
        return f"{_pmu_programme_course_api_url}/participants"
    if url_name == "PRONOSTIC":
        return f"{_pmu_programme_course_api_url}/pronostics"
    if url_name == "PRONOSTIC_DETAILLE":
        return f"{_pmu_programme_course_api_url}/pronostics-detailles"
    if url_name == "PERFORMANCE":
        return f"{_pmu_programme_course_api_url}/performances-detaillees/pretty"
    if url_name == "ENJEU":
        return f"{_pmu_programme_course_api_url}/masse-enjeu-v2"
    if url_name == "ENJEU_INTERNET":
        return f"{_pmu_programme_course_api_url}/masse-enjeu-v2?specialisation=INTERNET"
    if url_name == "CITATIONS":
        return f"{_pmu_programme_course_api_url}/citations"
    if url_name == "RAPPORTS_DEF":
        return f"{_pmu_programme_course_api_url}/rapports-definitifs"
    if url_name == "RAPPORTS_DEF_INTERNET":
        return (
            f"{_pmu_programme_course_api_url}/"
            f"rapports-definitifs?specialisation=INTERNET"
        )
    if url_name == "COMBINAISONS":
        return f"{_pmu_programme_course_api_url}/combinaisons"
    if url_name == "COMBINAISONS_INTERNET":
        return f"{_pmu_programme_course_api_url}/combinaisons?specialisation=INTERNET"
    if url_name == "CITATIONS_INTERNET":
        return f"{_pmu_programme_course_api_url}/citations?specialisation=INTERNET"
    # TODO compare w/ or w/o specialisation on combinaisons,
    #  look at E_SIMPLE_GAGNANT vs SIMPLE_GAGNANT

    assert code_pari is not None

    PMU_RAPPORTS_API_URL = f"{_pmu_programme_course_api_url}/rapports/{code_pari}"
    assert url_name == "RAPPORTS"
    return PMU_RAPPORTS_API_URL


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


def get_race_horses_records(
    programme: dict, date: dt.date, r_i: int, c_i: int, should_be_on_disk: bool
) -> Optional[List[dict]]:
    folder_path = utils.get_folder_path(source=SOURCE_PMU, date=date)

    reunions_ = [
        reunion
        for reunion in programme["programme"]["reunions"]
        if reunion["numOfficiel"] == r_i
    ]
    assert len(reunions_) == 1
    reunion = reunions_[0]

    courses_ = [
        course
        for reunion in programme["programme"]["reunions"]
        for course in reunion["courses"]
        if course["numReunion"] == r_i and course["numOrdre"] == c_i
    ]
    assert len(courses_) == 1
    course = courses_[0]

    course_race_datetime = dt.datetime.fromtimestamp(
        course["heureDepart"] / 1000,
        tz=dt.timezone(dt.timedelta(milliseconds=course["timezoneOffset"])),
    )
    if should_be_on_disk:
        participants_ = utils.load_json(
            filename=os.path.join(folder_path, f"R{r_i}_C{c_i}_participants.json")
        )
    else:
        participants_ = execute_get_query(
            url=get_pmu_api_url(url_name="PARTICIPANTS", r_i=r_i, c_i=c_i, date=date)
        )
    if participants_ is None or "participants" not in participants_:
        return None
    participants_ = participants_["participants"]
    race_horses = [
        {k: v for k, v in part.items() if not isinstance(v, dict)}
        for part in participants_
    ]
    if should_be_on_disk:
        performance_json = utils.load_json(
            filename=os.path.join(folder_path, f"R{r_i}_C{c_i}_performance.json")
        )
    else:
        performance_json = execute_get_query(
            url=get_pmu_api_url(url_name="PERFORMANCE", r_i=r_i, c_i=c_i, date=date)
        )

    last_race_date = get_last_race_date_from_performance(
        performance_json=performance_json, participants=race_horses
    )

    num_pmu_partants = [
        part["numPmu"] for part in race_horses if part["statut"] == "PARTANT"
    ]

    pari_type = "E_SIMPLE_GAGNANT" if course["hasEParis"] else "SIMPLE_GAGNANT"

    if should_be_on_disk:
        suffix = (
            "CITATIONS_INTERNET".lower() if course["hasEParis"] else "CITATIONS".lower()
        )
        citations = utils.load_json(
            filename=os.path.join(folder_path, f"R{r_i}_C{c_i}_{suffix}.json")
        )
    else:
        citations = execute_get_query(
            url=get_pmu_api_url(
                url_name="CITATIONS_INTERNET" if course["hasEParis"] else "CITATIONS",
                r_i=r_i,
                c_i=c_i,
                date=date,
            )
        )
    num_pmu_enjeu = get_num_pmu_enjeu_from_citations(
        citations=citations, pari_type=pari_type, num_pmu_partants=num_pmu_partants
    )
    is_true_total_enjeu = True

    if len(num_pmu_partants) <= 12 and num_pmu_enjeu is None:
        if should_be_on_disk:
            suffix = (
                "COMBINAISONS_INTERNET".lower()
                if course["hasEParis"]
                else "COMBINAISONS".lower()
            )
            combinaisons = utils.load_json(
                filename=os.path.join(folder_path, f"R{r_i}_C{c_i}_{suffix}.json")
            )
        else:
            combinaisons = execute_get_query(
                url=get_pmu_api_url(
                    url_name="COMBINAISONS_INTERNET"
                    if course["hasEParis"]
                    else "COMBINAISONS",
                    r_i=r_i,
                    c_i=c_i,
                    date=date,
                )
            )

        num_pmu_enjeu = get_num_pmu_enjeu_from_combinaisons(
            combinaisons=combinaisons,
            pari_type=pari_type,
            num_pmu_partants=num_pmu_partants,
        )

    if num_pmu_enjeu is None:
        if should_be_on_disk:
            rapport_simple_gagnant = utils.load_json(
                filename=os.path.join(
                    folder_path, f"R{r_i}_C{c_i}_{pari_type}_{'RAPPORTS'.lower()}.json"
                )
            )
            suffix = (
                "ENJEU_INTERNET".lower() if course["hasEParis"] else "ENJEU".lower()
            )
            enjeux = utils.load_json(
                filename=os.path.join(folder_path, f"R{r_i}_C{c_i}_{suffix}.json")
            )
        else:
            rapport_simple_gagnant = execute_get_query(
                url=get_pmu_api_url(
                    url_name="RAPPORTS",
                    r_i=r_i,
                    c_i=c_i,
                    date=date,
                    code_pari=pari_type,
                )
            )
            enjeux = execute_get_query(
                url=get_pmu_api_url(
                    url_name="ENJEU_INTERNET" if course["hasEParis"] else "ENJEU",
                    r_i=r_i,
                    c_i=c_i,
                    date=date,
                )
            )

        num_pmu_enjeu = get_num_pmu_enjeu_from_rapport_simple_enjeux(
            rapport_simple_gagnant=rapport_simple_gagnant,
            enjeux=enjeux,
            pari_type=pari_type,
            num_pmu_partants=num_pmu_partants,
        )
        is_true_total_enjeu = num_pmu_enjeu is None
    penetrometer_value = get_penetrometer_value(course=course)
    course_incidents = course["incidents"] if "incidents" in course else []
    incident_nums = {
        num_part
        for incident in course_incidents
        for num_part in incident["numeroParticipants"]
    }
    for race_horse, part_ in zip(race_horses, participants_):
        # TODO integrate other dict keys
        # dict key found {'commentaireApresCourse',
        #  'dernierRapportDirect',
        #  'dernierRapportReference',
        #  'distanceChevalPrecedent',
        #  'gainsParticipant', # added here
        #  'robe'}
        if "gainsParticipant" in part_:
            race_horse.update(part_["gainsParticipant"])
        race_horse["n_reunion"] = r_i
        race_horse["n_course"] = c_i
        race_horse["date"] = date
        race_horse["race_datetime"] = course_race_datetime
        race_horse["in_incident"] = race_horse["numPmu"] in incident_nums
        race_horse["incident_type"] = (
            None
            if race_horse["numPmu"] not in incident_nums
            else [
                incident["type"]
                for incident in course_incidents
                if race_horse["numPmu"] in incident["numeroParticipants"]
            ][0]
        )
        race_horse["totalEnjeu"] = (
            None
            if num_pmu_enjeu is None
            else num_pmu_enjeu.get(race_horse["numPmu"], None)
        )

        race_horse["last_race_date"] = (
            None
            if last_race_date is None
            else last_race_date.get(race_horse["numPmu"], None)
        )
        handicap_weight = None
        if "poidsConditionMonte" in race_horse and race_horse["poidsConditionMonte"]:
            handicap_weight = race_horse["poidsConditionMonte"]
        elif "handicapPoids" in race_horse and race_horse["handicapPoids"]:
            handicap_weight = race_horse["handicapPoids"]

        race_horse["handicap_weight"] = handicap_weight  # in hectogram
        race_horse["is_true_total_enjeu"] = is_true_total_enjeu
        race_horse["reunion_nature"] = reunion["nature"]
        race_horse["reunion_audience"] = reunion["audience"]
        race_horse["reunion_pays"] = reunion["pays"]["code"]
        race_horse["course_statut"] = course["statut"]
        race_horse["course_discipline"] = course["discipline"]
        race_horse["course_specialite"] = course["specialite"]
        race_horse["course_condition_sexe"] = course["conditionSexe"]
        race_horse["course_condition_age"] = (
            None if "conditionAge" not in course else course["conditionAge"]
        )
        race_horse["course_track_type"] = (
            None if "typePiste" not in course else course["typePiste"]
        )

        race_horse["course_penetrometre"] = penetrometer_value
        # TODO Deal with ecurie
        race_horse["course_corde"] = None if "corde" not in course else course["corde"]
        race_horse["course_hippodrome"] = (
            None
            if "hippodrome" not in course
            else course["hippodrome"]["codeHippodrome"]
        )
        race_horse["course_parcours"] = (
            None if "parcours" not in course else course["parcours"]
        )
        race_horse["course_distance"] = (
            None if "distance" not in course else course["distance"]
        )
        race_horse["course_distance_unit"] = (
            None if "distanceUnit" not in course else course["distanceUnit"]
        )
        race_horse["course_duration"] = (
            None if "dureeCourse" not in course else course["dureeCourse"]
        )
        race_horse["course_prize_pool"] = (
            None if "montantTotalOffert" not in course else course["montantTotalOffert"]
        )  # TODO look at difference with "montantPrix"
        race_horse["course_winner_prize"] = (
            None if "montantOffert1er" not in course else course["montantOffert1er"]
        )

    return race_horses
