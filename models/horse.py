import logging
import re
from typing import List
from typing import Optional
from typing import Tuple

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.orm import validates

from database.setup import SQLAlchemySession
from models import Race
from models.base import Base

logger = logging.getLogger(__file__)


class Horse(Base):
    __tablename__ = "horses"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)
    country_code = sa.Column(sa.String, nullable=True, index=True)
    is_born_male = sa.Column(sa.Boolean, nullable=True, index=True)
    runners = relationship("Runner", backref="horse")
    father_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    father = relationship(
        "Horse", remote_side=id, backref="father_children", foreign_keys=father_id
    )
    mother_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    mother = relationship(
        "Horse", remote_side=id, backref="mother_children", foreign_keys=mother_id
    )

    @validates("name")
    def validate_name(self, key, value: str) -> str:
        assert value.upper() == value
        return value

    @property
    def children(self) -> List["Horse"]:
        if self.is_born_male:
            assert not self.mother_children
            return self.father_children
        assert not self.father_children
        return self.mother_children

    @staticmethod
    def _update_check_horse(
        found_horse: "Horse",
        is_born_male: Optional[bool],
        country_code: Optional[str],
        father: Optional["Horse"],
        mother: Optional["Horse"],
        db_session: SQLAlchemySession,
    ) -> "Horse":
        if found_horse.country_code is None and country_code:
            found_horse.country_code = country_code
        if found_horse.father_id is None and father:
            found_horse.father_id = father.id
        if found_horse.mother_id is None and mother:
            found_horse.mother_id = mother.id
        if found_horse.is_born_male is None and is_born_male is not None:
            found_horse.is_born_male = is_born_male

        assert found_horse.name
        if country_code:
            assert found_horse.country_code == country_code
        if father:
            assert found_horse.father_id == father.id
        if mother:
            assert found_horse.mother_id == mother.id
        if is_born_male is not None:
            assert found_horse.is_born_male == is_born_male
        db_session.commit()
        return found_horse

    @classmethod
    def upsert(
        cls,
        name: str,
        is_born_male: Optional[bool],
        country_code: Optional[str],
        father: Optional["Horse"],
        mother: Optional["Horse"],
        db_session: SQLAlchemySession,
    ) -> Optional["Horse"]:
        potential_horses = db_session.query(Horse).filter(Horse.name == name).all()

        # TODO filter on age according to previous races

        if is_born_male is not None:
            potential_horses = [
                horse
                for horse in potential_horses
                if horse.is_born_male is is_born_male or horse.is_born_male is None
            ]

        if country_code:
            potential_horses = [
                horse
                for horse in potential_horses
                if horse.country_code == country_code or horse.country_code is None
            ]

        if father:
            potential_horses = [
                horse
                for horse in potential_horses
                if horse.father_id == father.id or horse.father_id is None
            ]

        if mother:
            potential_horses = [
                horse
                for horse in potential_horses
                if horse.mother_id == mother.id or horse.mother_id is None
            ]

        if len(potential_horses) > 1:
            logger.debug("Too many horses found!")
            return None
        if len(potential_horses) == 1:
            found_horse = potential_horses[0]
            return cls._update_check_horse(
                found_horse=found_horse,
                is_born_male=is_born_male,
                country_code=country_code,
                father=father,
                mother=mother,
                db_session=db_session,
            )

        horse = Horse(
            name=name,
            country_code=country_code,
            father_id=father.id if father else None,
            mother_id=mother.id if mother else None,
            is_born_male=is_born_male,
        )
        db_session.add(horse)
        db_session.commit()
        assert horse.id
        return horse

    @classmethod
    def upsert_father_mother(  # pylint:disable=too-many-branches
        cls,
        current_horse_age: Optional[int],
        race: Race,
        father_mother_names: Optional[str],
        db_session: SQLAlchemySession,
    ) -> Tuple[Optional["Horse"], Optional["Horse"]]:
        if not father_mother_names:
            return None, None
        father_mother_names = re.split("[-/]", father_mother_names)
        if len(father_mother_names) != 2:
            logger.debug(
                "Could not find father mother names in origins: %s", father_mother_names
            )
            return None, None

        father_name, mother_name = father_mother_names
        father_name = father_name.upper()
        mother_name = mother_name.upper()

        potential_fathers = (
            db_session.query(Horse)
            .filter(Horse.name == father_name, Horse.is_born_male.is_(True))
            .all()
        )

        # TODO don't be too strict on is_born_male, update it the found horse
        # Fathers needs to be older than their children
        if current_horse_age:
            potential_fathers = [
                potential_father
                for potential_father in potential_fathers
                if all(
                    f_rn.race.date.year - f_rn.age <= race.date.year - current_horse_age
                    for f_rn in potential_father.runners
                )
            ]

        if not potential_fathers:
            father: Optional["Horse"] = Horse(name=father_name, is_born_male=True)
            db_session.add(father)
            db_session.commit()
        elif len(potential_fathers) == 1:
            father = potential_fathers[0]
        else:
            assert len(potential_fathers) > 1
            logger.debug("Too many fathers found!")
            father = None

        potential_mothers = (
            db_session.query(Horse)
            .filter(Horse.name == mother_name, Horse.is_born_male.is_(False))
            .all()
        )
        # TODO don't be too strict on is_born_male, update it the found horse
        # Mothers needs to be older than their children
        if current_horse_age:
            potential_mothers = [
                potential_mother
                for potential_mother in potential_mothers
                if all(
                    m_rn.race.date.year - m_rn.age <= race.date.year - current_horse_age
                    for m_rn in potential_mother.runners
                )
            ]

        if not potential_mothers:
            mother: Optional["Horse"] = Horse(name=mother_name, is_born_male=False)
            db_session.add(mother)
            db_session.commit()
        elif len(potential_mothers) == 1:
            mother = potential_mothers[0]
        else:
            assert len(potential_mothers) > 1
            logger.debug("Too many mothers found!")
            mother = None
        return father, mother


sa.Index(
    "horse_name_parents_index",
    Horse.name,
    Horse.father_id,
    Horse.mother_id,
    unique=True,
)
