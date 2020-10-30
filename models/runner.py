import sqlalchemy as sa
from sqlalchemy.orm import relationship

from models.base import Base


class Runner(Base):
    __tablename__ = "runners"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    unibet_id = sa.Column(sa.Integer, unique=True, nullable=False, index=True)
    race_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("races.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    race = relationship("Race", back_populates="runners")

    jockey_weight = sa.Column(sa.Float, nullable=True, index=True)
    unibet_n = sa.Column(sa.Integer, nullable=False, index=True)
    team = sa.Column(sa.Integer, nullable=True, index=True)
    draw = sa.Column(sa.Integer, nullable=True, index=True)
    blinkers = sa.Column(sa.String, nullable=True, index=True)
    shoes = sa.Column(sa.String, nullable=True, index=True)
    silk = sa.Column(sa.String, nullable=True, index=True)
    bet_counter = sa.Column(sa.Integer, nullable=True, index=True)
    stakes = sa.Column(sa.Integer, nullable=True, index=True)
    music = sa.Column(sa.String, nullable=True, index=True)
    sex = sa.Column(sa.String, nullable=True, index=True)
    age = sa.Column(sa.Integer, nullable=True, index=True)
    coat = sa.Column(sa.String, nullable=True, index=True)
    origins = sa.Column(sa.String, nullable=True, index=True)
    comment = sa.Column(sa.String, nullable=True, index=True)

    owner_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("owners.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner = relationship("Owner", back_populates="runners")
    trainer_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("trainers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trainer = relationship("Trainer", back_populates="trainees")

    position = sa.Column(sa.String, nullable=True, index=True)
    race_duration_sec = sa.Column(sa.Integer, nullable=True)
    length = sa.Column(sa.String, nullable=True)
