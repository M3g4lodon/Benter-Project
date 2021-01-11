import datetime as dt
from typing import Tuple

import sqlalchemy as sa
from sqlalchemy.orm import relationship

from constants import UnibetRaceType
from database.setup import SQLAlchemySession
from models import HorseShow
from models.base import Base


class Race(Base):
    __tablename__ = "races"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    unibet_id = sa.Column(sa.Integer, unique=True, nullable=False, index=True)
    name = sa.Column(sa.String, nullable=True, index=False)
    start_at = sa.Column(sa.DateTime, nullable=False)
    date = sa.Column(sa.Date, nullable=False, index=True)
    unibet_n = sa.Column(sa.Integer, nullable=False, index=True)
    type = sa.Column(sa.Enum(UnibetRaceType), nullable=True)
    conditions = sa.Column(sa.String, nullable=True)
    stake = sa.Column(sa.Integer, nullable=True)
    distance = sa.Column(sa.Integer, nullable=False)
    friendly_URL = sa.Column(sa.String, nullable=True)
    pronostic = sa.Column(sa.String, nullable=True)
    horse_show_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("horse_shows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    runners = relationship("Runner", backref="race")

    @property
    def horse_show_unibet_n(self) -> int:
        return self.horse_show.unibet_n

    @property
    def unibet_code(self) -> Tuple[dt.date, int, int]:
        return self.date, self.horse_show_unibet_n, self.unibet_n

    @classmethod
    def upsert(
        cls,
        race_unibet_id: int,
        race_unibet_n: int,
        race_name: str,
        race_start_at: dt.datetime,
        race_date: dt.date,
        race_type: UnibetRaceType,
        race_conditions: str,
        race_stake: int,
        race_arjel_level: str,
        race_distance: int,
        race_friendly_url: str,
        race_pronostic: str,
        horse_show: HorseShow,
        db_session: SQLAlchemySession,
    ) -> "Race":

        assert race_arjel_level == "2"

        found_race = (
            db_session.query(Race)
            .filter(Race.unibet_id == race_unibet_id)
            .one_or_none()
        )
        if found_race is not None:
            assert found_race.name == race_name
            assert found_race.start_at == race_start_at
            assert found_race.date == race_date
            assert found_race.unibet_n == race_unibet_n
            assert found_race.type == race_type
            assert found_race.conditions == race_conditions
            assert found_race.stake == race_stake
            assert found_race.distance == race_distance
            assert found_race.friendly_URL == race_friendly_url
            assert found_race.pronostic == race_pronostic
            assert found_race.horse_show_id == horse_show.id
            assert found_race.id
            return found_race

        race = Race(
            unibet_id=race_unibet_id,
            name=race_name,
            start_at=race_start_at,
            date=race_date,
            unibet_n=race_unibet_n,
            type=race_type,
            conditions=race_conditions,
            stake=race_stake,
            distance=race_distance,
            friendly_URL=race_friendly_url,
            pronostic=race_pronostic,
            horse_show_id=horse_show.id,
        )
        db_session.add(race)
        db_session.commit()

        assert race.id
        return race


sa.Index("race_code_index", Race.date, Race.unibet_n)
