from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models.base import Base
from utils import setup_logger

logger = setup_logger(name=__name__)


class EntityPerson(Base):
    __tablename__ = "entity_person"
    entity_id = sa.Column(sa.Integer, sa.ForeignKey("entities.id"), primary_key=True)
    person_id = sa.Column(sa.Integer, sa.ForeignKey("persons.id"), primary_key=True)
    person = relationship("Person", back_populates="entities")
    entity = relationship("Entity", back_populates="persons")

    @classmethod
    def upsert(
        cls,
        person: "Person",
        entity: "Entity",
        db_session: SQLAlchemySession,
    ) -> "EntityPerson":
        found_association = (
            db_session.query(EntityPerson)
            .filter(EntityPerson.entity_id == entity.id)
            .filter(EntityPerson.person_id == person.id)
            .one_or_none()
        )

        if found_association is not None:
            return found_association

        association = EntityPerson(entity_id=entity.id, person_id=person.id)
        db_session.add(association)
        db_session.commit()
        return association


class Person(Base):
    __tablename__ = "persons"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)

    entities = relationship("EntityPerson", back_populates="person")

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
