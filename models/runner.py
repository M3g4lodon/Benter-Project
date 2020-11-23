import sqlalchemy as sa

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
    weight = sa.Column(sa.Float, nullable=True, index=True)
    unibet_n = sa.Column(sa.Integer, nullable=False, index=True)
    draw = sa.Column(sa.Integer, nullable=True, index=True)
    blinkers = sa.Column(sa.String, nullable=True, index=True)
    shoes = sa.Column(sa.String, nullable=True, index=True)
    silk = sa.Column(sa.String, nullable=True, index=True)
    stakes = sa.Column(sa.Integer, nullable=True, index=True)
    music = sa.Column(sa.String, nullable=True, index=True)
    sex = sa.Column(sa.String, nullable=True, index=True)
    age = sa.Column(sa.String, nullable=True, index=True)
    coat = sa.Column(sa.String, nullable=True, index=True)
    origins = sa.Column(sa.String, nullable=True, index=True)
    comment = sa.Column(sa.String, nullable=True, index=True)
    length = sa.Column(sa.String, nullable=True, index=True)

    owner_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("owners.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    trainer_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("trainers.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    jockey_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("jockeys.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    horse_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position = sa.Column(sa.String, nullable=True, index=True)
    race_duration_sec = sa.Column(sa.Float, nullable=True)
    morning_odds = sa.Column(sa.Float, nullable=True, index=True)
    final_odds = sa.Column(sa.Float, nullable=True, index=True)

    @property
    def date(self):
        return self.race.date
