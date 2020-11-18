import sqlalchemy as sa

from models.base import Base


class HorseShow(Base):
    __tablename__ = "horse_shows"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    unibet_id = sa.Column(sa.Integer, unique=True, nullable=False, index=True)
    datetime = sa.Column(sa.DateTime, nullable=False, index=True)
    unibet_n = sa.Column(sa.Integer, nullable=False, index=True)
    ground = sa.Column(sa.String, nullable=True)
    race_track_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("race_tracks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
