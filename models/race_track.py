import sqlalchemy as sa
from sqlalchemy import Index
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models.base import Base


class RaceTrack(Base):
    __tablename__ = "race_tracks"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    race_track_name = sa.Column(sa.String, nullable=False, index=True)
    country_name = sa.Column(sa.String, nullable=False, index=True)
    horse_shows = relationship("HorseShow", backref="race_track")

    @classmethod
    def upsert(
        cls, race_track_name: str, country_name: str, db_session: SQLAlchemySession
    ) -> "RaceTrack":
        found_race_track = (
            db_session.query(RaceTrack)
            .filter(
                RaceTrack.race_track_name == race_track_name,
                RaceTrack.country_name == country_name,
            )
            .one_or_none()
        )
        if found_race_track is not None:
            assert found_race_track.id
            return found_race_track

        race_track = RaceTrack(
            race_track_name=race_track_name, country_name=country_name
        )
        db_session.add(race_track)
        db_session.commit()

        assert race_track.id
        return race_track


Index(
    "index_track_name_country",
    RaceTrack.race_track_name,
    RaceTrack.country_name,
    unique=True,
)
