import sqlalchemy as sa
from sqlalchemy.orm import relationship

from models.base import Base


class RaceTrack(Base):
    __tablename__ = "race_tracks"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    race_track_name = sa.Column(sa.String, nullable=False, index=True, unique=True)
    country_name = sa.Column(sa.String, nullable=False, index=True)
    horse_shows = relationship("HorseShow", backref="race_track")
