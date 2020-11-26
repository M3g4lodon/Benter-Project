from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models.base import Base


class Jockey(Base):
    __tablename__ = "jockeys"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)

    runners = relationship("Runner", backref="jockey")

    @classmethod
    def upsert(
        cls, name: Optional[str], db_session: SQLAlchemySession
    ) -> Optional["Jockey"]:
        if not name:
            return None
        if not name.strip():
            return None
        found_instance = (
            db_session.query(Jockey).filter(Jockey.name == name).one_or_none()
        )

        if found_instance is not None:
            assert found_instance.id
            return found_instance

        instance = Jockey(name=name)
        db_session.add(instance)
        db_session.commit()
        assert instance.id
        return instance
