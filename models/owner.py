from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models.base import Base


class Owner(Base):
    __tablename__ = "owners"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)

    runners = relationship("Runner", backref="owner")

    @classmethod
    def upsert(
        cls, name: Optional[str], db_session: SQLAlchemySession
    ) -> Optional["Owner"]:
        if not name:
            return None
        if not name.strip():
            return None
        found_instance = (
            db_session.query(Owner).filter(Owner.name == name).one_or_none()
        )

        if found_instance is not None:
            assert found_instance.id
            return found_instance

        instance = Owner(name=name)
        db_session.add(instance)
        db_session.commit()
        assert instance.id
        return instance
