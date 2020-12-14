from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models.base import Base
from utils import setup_logger

logger = setup_logger(name=__name__)


class Person(Base):
    __tablename__ = "persons"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)

    runners_as_jockey = relationship(
        "Runner", backref="jockey", foreign_keys="Runner.jockey_id"
    )
    runners_as_owner = relationship(
        "Runner", backref="owner", foreign_keys="Runner.owner_id"
    )
    runners_as_trainer = relationship(
        "Runner", backref="trainer", foreign_keys="Runner.trainer_id"
    )

    @classmethod
    def upsert(
        cls,
        person_id: Optional[int],
        name: Optional[str],
        db_session: SQLAlchemySession,
    ) -> Optional["Person"]:

        if person_id is not None:
            assert name
            name = name.strip()
            assert name
            found_person = db_session.query(Person).filter(Person.id == person_id).one()
            assert found_person.name == name
            return found_person

        if not name:
            return None

        if not isinstance(name, str):
            logger.warning("Can not upsert person with name: %s", name)
            return None

        name = name.strip()
        if not name:
            return None
        found_person = (
            db_session.query(Person).filter(Person.name == name).one_or_none()
        )

        if found_person is not None:
            assert found_person.id
            return found_person

        person = Person(name=name)
        db_session.add(person)
        db_session.commit()
        assert person.id
        return person
