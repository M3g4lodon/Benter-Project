from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models import EntityOrganization
from models import EntityPerson
from models import Organization
from models import Person
from models.base import Base
from utils import parse_name
from utils import setup_logger

logger = setup_logger(name=__name__)


class Entity(Base):
    __tablename__ = "entities"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)

    runners_as_jockey = relationship(
        "Runner", backref="jockey_entity", foreign_keys="Runner.jockey_id"
    )
    runners_as_owner = relationship(
        "Runner", backref="owner_entity", foreign_keys="Runner.owner_id"
    )
    runners_as_trainer = relationship(
        "Runner", backref="trainer_entity", foreign_keys="Runner.trainer_id"
    )

    persons = relationship("EntityPerson", back_populates="entity")
    organizations = relationship("EntityOrganization", back_populates="entity")

    @classmethod
    def upsert(
        cls,
        entity_id: Optional[int],
        name: Optional[str],
        db_session: SQLAlchemySession,
    ) -> Optional["Entity"]:

        if entity_id is not None:
            assert name
            name = name.strip()
            assert name
            found_entity = db_session.query(Entity).filter(Entity.id == entity_id).one()
            assert found_entity.name == name
            return found_entity

        if not name:
            return None

        if not isinstance(name, str):
            logger.warning("Can not upsert entity with name: %s", name)
            return None

        name = name.strip()
        if not name:
            return None
        found_entity = (
            db_session.query(Entity).filter(Entity.name == name).one_or_none()
        )

        if found_entity is not None:
            assert found_entity.id
            return found_entity

        entity = Entity(name=name)
        db_session.add(entity)
        db_session.commit()

        parsed_names = parse_name.parse_name(name=name)
        for p_name in parsed_names.person_names:
            person = Person.upsert(person_id=None, name=p_name, db_session=db_session)
            _ = EntityPerson.upsert(person=person, entity=entity, db_session=db_session)

        for o_name in parsed_names.organization_names:
            organization = Organization.upsert(
                organization_id=None, name=o_name, db_session=db_session
            )
            _ = EntityOrganization.upsert(
                organization=organization, entity=entity, db_session=db_session
            )

        db_session.commit()
        assert entity.id
        return entity
