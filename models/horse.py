import logging
from typing import List
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.orm import validates

from database.setup import SQLAlchemySession
from models.base import Base

logger = logging.getLogger(__file__)


class Horse(Base):
    __tablename__ = "horses"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)
    country_code = sa.Column(sa.String, nullable=True, index=True)
    is_born_male = sa.Column(sa.Boolean, nullable=True, index=True)
    runners = relationship("Runner", backref="horse")
    first_found_origins = sa.Column(sa.String, nullable=True, index=False)
    birth_year = sa.Column(sa.Integer, nullable=True, index=True)
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
        assert value
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
        origins: Optional[str],
        birth_year: Optional[int],
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
        if found_horse.first_found_origins is None and origins:
            found_horse.first_found_origins = origins
        if found_horse.birth_year is None and birth_year:
            found_horse.birth_year = birth_year

        assert found_horse.name
        if country_code:
            assert found_horse.country_code == country_code
        if father:
            assert found_horse.father_id == father.id
        if mother:
            assert found_horse.mother_id == mother.id
        if is_born_male is not None:
            assert found_horse.is_born_male == is_born_male
        # No checks on first_found_origins
        if birth_year:
            assert found_horse.birth_year == birth_year

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
        origins: Optional[str],
        birth_year: Optional[int],
        db_session: SQLAlchemySession,
    ) -> Optional["Horse"]:
        potential_horses = db_session.query(Horse).filter(Horse.name == name).all()

        # TODO filter on age according to previous races (approximate birth year), same origins

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
        if len(potential_horses) > 1 and origins:
            potential_horses = [
                horse
                for horse in potential_horses
                if horse.first_found_origins == origins
            ]

        if len(potential_horses) > 1:
            logger.warning("Too many horses found!")
            return None
        if len(potential_horses) == 1:
            found_horse = potential_horses[0]
            return cls._update_check_horse(
                found_horse=found_horse,
                is_born_male=is_born_male,
                country_code=country_code,
                father=father,
                mother=mother,
                origins=origins,
                birth_year=birth_year,
                db_session=db_session,
            )

        horse = Horse(
            name=name,
            country_code=country_code,
            father_id=father.id if father else None,
            mother_id=mother.id if mother else None,
            first_found_origins=origins,
            birth_year=birth_year,
            is_born_male=is_born_male,
        )
        db_session.add(horse)
        db_session.commit()
        assert horse.id
        return horse


sa.Index(
    "horse_name_parents_index",
    Horse.name,
    Horse.father_id,
    Horse.mother_id,
    unique=True,
)
