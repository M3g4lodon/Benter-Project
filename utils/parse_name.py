import dataclasses
import re
from typing import List


@dataclasses.dataclass()
class ParsedName:
    person_names: List[str] = dataclasses.field(default_factory=lambda: [])
    organization_names: List[str] = dataclasses.field(default_factory=lambda: [])


corporation_marker = {
    "Co ",
    "Inc",
    "Ltd",
    "Partner",
    "partner",
    "LLC",
    "Ecurie",
    "ecurie",
    "corp",
    "Corp.",
    "corp.",
    "Llc",
    "Gmbh",
    "Limited",
    "Syndicate",
    "Team",
    "team",
    "Racing",
    "Horseracing",
    "Developments",
    "Family",
    "family",
    "Plc",
    "plc",
    "ec.",
}
regex_separator_marker = (
    r"(?<!\b\w)(?<!\b\w\s)(?<!^)(?<!^\s)(?<!Ch|mr)"
    r"(?<!(?:Ch|mr)\s)"
    r"(?:[&\-\/&,]|\sAnd\s|\set\s)(?!\s?\w\b)"
    r"(?!\s?Inc)(?!\s?Partner)(?!\s?Co)"
)
forbidden_beginnings = [r"\b\w", "^", "Ch", "Mr", "M"]
forbidden_endings = [r"\w\b", "Inc", "Partner", "Co", "Corp", "Jr", "$"]
separators = r"[&-\/&,]|\sAnd\s|\set\s"
regex_separator = "".join(rf"(?<!{b})(?<!{b}\s)" for b in forbidden_beginnings)
regex_separator += rf"(?:{separators})"
regex_separator += rf"(?!\s?(?:{'|'.join(forbidden_endings)}))"


def parse_name(name: str) -> ParsedName:
    person_names, organization_names = [], []

    entities = re.split(regex_separator, name, flags=re.IGNORECASE)

    for entity in entities:
        if entity is None:
            continue
        if entity.strip() == "":
            continue
        if re.search("|".join(corporation_marker), entity, flags=re.IGNORECASE):
            organization_names.append(entity.strip())
        else:
            person_names.append(entity.strip())

    return ParsedName(person_names=person_names, organization_names=organization_names)
