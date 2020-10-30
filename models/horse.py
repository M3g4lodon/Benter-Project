import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base


class Horse(Base):
    __tablename__ = "horses"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    unibet_id = sa.Column(sa.Integer, unique=True, nullable=False, index=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)
    horse_race = sa.Column(sa.String, unique=False, nullable=True, index=True)
    father_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    father = orm.relationship(
        "Horse", remote_side=id, backref=orm.backref("father_children")
    )
    mother_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    mother = orm.relationship(
        "Horse", remote_side=id, backref=orm.backref("mother_children")
    )
    father_mother_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    father_mother = orm.relationship(
        "Horse", remote_side=id, backref=orm.backref("father_mother_children")
    )
