import sqlalchemy as sa

from models.base import Base


class Race(Base):
    __tablename__ = "races"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    unibet_id = sa.Column(sa.Integer, unique=True, nullable=False, index=True)
    name = sa.Column(sa.String, nullable=True, index=False)
    start_at = sa.Column(sa.DateTime, nullable=False)
    date = sa.Column(sa.Date, nullable=False, index=True)
    unibet_n = sa.Column(sa.Integer, nullable=False, index=True)
    type = sa.Column(sa.String, nullable=True)
    conditions = sa.Column(sa.String, nullable=True)
    stake = sa.Column(sa.Integer, nullable=True)
    arjel_level = sa.Column(sa.String, nullable=True)
    distance = sa.Column(sa.Integer, nullable=False)
    friendly_URL = sa.Column(sa.String, nullable=True)
    pronostic = sa.Column(sa.String, nullable=True)
    horse_show_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horse_shows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )


sa.Index("race_code_index", Race.date, Race.unibet_n)
