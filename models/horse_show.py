import datetime as dt

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database.setup import SQLAlchemySession
from models import RaceTrack
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
    races = relationship("Race", backref="horse_show")

    @classmethod
    def upsert(
        cls,
        horse_show_unibet_id: int,
        horse_show_ground: str,
        horse_show_unibet_n: int,
        horse_show_datetime: dt.datetime,
        race_track: RaceTrack,
        db_session: SQLAlchemySession,
    ):
        found_horse_show = (
            db_session.query(HorseShow)
            .filter(
                HorseShow.unibet_n == horse_show_unibet_n,
                sa.func.date(HorseShow.datetime) == horse_show_datetime.date(),
            )
            .one_or_none()
        )

        if found_horse_show is not None:
            assert found_horse_show.unibet_id == horse_show_unibet_id
            assert found_horse_show.ground == horse_show_ground
            assert found_horse_show.race_track_id == race_track.id
            assert found_horse_show.id
            return found_horse_show

        horse_show = HorseShow(
            unibet_id=horse_show_unibet_id,
            datetime=horse_show_datetime,
            unibet_n=horse_show_unibet_n,
            ground=horse_show_ground,
            race_track_id=race_track.id,
        )
        db_session.add(horse_show)
        db_session.commit()
        assert horse_show.id
        return horse_show
