import logging
from typing import List

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.orm import validates

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
