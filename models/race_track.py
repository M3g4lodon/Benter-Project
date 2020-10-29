import sqlalchemy as sa

from models.base import Base


class RaceTrack(Base):
    __tablename__ = "race_tracks"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    race_track_name = sa.column(sa.String, nullable=False)
    country_name = sa.column(sa.String, nullable=False)
