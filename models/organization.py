from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models.base import Base
from utils import setup_logger

logger = setup_logger(name=__name__)


class EntityOrganization(Base):
    __tablename__ = "entity_organization"
    entity_id = sa.Column(sa.Integer, sa.ForeignKey("entities.id"), primary_key=True)
    organization_id = sa.Column(
        sa.Integer, sa.ForeignKey("organizations.id"), primary_key=True
    )
    organization = relationship("Organization", back_populates="entities")
    entity = relationship("Entity", back_populates="organizations")

    @classmethod
    def upsert(
        cls,
        organization: "Organization",
        entity: "Entity",
        db_session: SQLAlchemySession,
    ) -> "EntityOrganization":
        found_association = (
            db_session.query(EntityOrganization)
            .filter(EntityOrganization.entity_id == entity.id)
            .filter(EntityOrganization.organization_id == organization.id)
            .one_or_none()
        )

        if found_association is not None:
            return found_association

        association = EntityOrganization(
            entity_id=entity.id, organization_id=organization.id
        )
        db_session.add(association)
        db_session.commit()
        return organization


class Organization(Base):
    __tablename__ = "organizations"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)

    entities = relationship("EntityOrganization", back_populates="organization")

    @classmethod
    def upsert(
        cls,
        organization_id: Optional[int],
        name: Optional[str],
        db_session: SQLAlchemySession,
    ) -> Optional["Organization"]:

        if organization_id is not None:
            assert name
            name = name.strip()
            assert name
            found_organization = (
                db_session.query(Organization)
                .filter(Organization.id == organization_id)
                .one()
            )
            assert found_organization.name == name
            return found_organization

        if not name:
            return None

        if not isinstance(name, str):
            logger.warning("Can not upsert organization with name: %s", name)
            return None

        name = name.strip()
        if not name:
            return None
        found_organization = (
            db_session.query(Organization)
            .filter(Organization.name == name)
            .one_or_none()
        )

        if found_organization is not None:
            assert found_organization.id
            return found_organization

        organization = Organization(name=name)
        db_session.add(organization)
        db_session.commit()
        assert organization.id
        return organization
