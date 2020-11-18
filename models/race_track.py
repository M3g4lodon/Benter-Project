import sqlalchemy as sa
from sqlalchemy import Index
from sqlalchemy.orm import relationship

from models.base import Base


class RaceTrack(Base):
    __tablename__ = "race_tracks"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    race_track_name = sa.Column(sa.String, nullable=False, index=True)
    country_name = sa.Column(sa.String, nullable=False, index=True)
    horse_shows = relationship("HorseShow", backref="race_track")


Index(
    "index_track_name_country",
    RaceTrack.race_track_name,
    RaceTrack.country_name,
    unique=True,
)
