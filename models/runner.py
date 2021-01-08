import datetime as dt
from typing import Optional
from typing import Tuple

import sqlalchemy as sa

from constants import UnibetBlinkers
from constants import UnibetCoat
from constants import UnibetHorseSex
from constants import UnibetShoes
from database.setup import SQLAlchemySession
from models import Horse
from models import Person
from models import Race
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
    rope_n = sa.Column(
        sa.Integer, nullable=True, index=True
    )  # in French "Corde", is the real draw number (double checked with PMU data)
    draw = sa.Column(
        sa.Integer, nullable=True, index=True
    )  # Might be different from `rope_n`
    blinkers = sa.Column(sa.Enum(UnibetBlinkers), nullable=True, index=True)
    shoes = sa.Column(sa.Enum(UnibetShoes), nullable=True, index=True)
    silk = sa.Column(
        sa.String, nullable=True, index=True
    )  # image name of the jockey jersey
    stakes = sa.Column(
        sa.Integer, nullable=True, index=True
    )  # from runner dict, probably year to year stakes (not sure)
    music = sa.Column(sa.String, nullable=True, index=True)
    sex = sa.Column(sa.Enum(UnibetHorseSex), nullable=True, index=True)
    age = sa.Column(sa.Integer, nullable=True, index=True)
    team = sa.Column(
        sa.Integer, nullable=False, index=True
    )  # team number in race 0 means no team, 1 first team and so on...
    coat = sa.Column(
        sa.Enum(UnibetCoat), nullable=True, index=True
    )  # in French "robe", equine color
    origins = sa.Column(sa.String, nullable=True, index=False)
    comment = sa.Column(sa.String, nullable=True, index=False)
    length = sa.Column(sa.String, nullable=True, index=True)

    owner_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    trainer_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    jockey_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("persons.id", ondelete="CASCADE"),
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
    def date(self) -> dt.date:
        return self.race.date

    @property
    def horse_show_unibet_n(self) -> int:
        return self.race.horse_show.unibet_n

    @property
    def race_unibet_n(self) -> int:
        return self.race.unibet_n

    @property
    def unibet_code(self) -> Tuple[dt.date, int, int, int]:
        return self.date, self.horse_show_unibet_n, self.race_unibet_n, self.unibet_n

    @classmethod
    def upsert(
        cls,
        unibet_id: int,
        race: Race,
        weight: Optional[int],
        unibet_n: int,
        draw: int,
        blinkers: UnibetBlinkers,
        shoes: UnibetShoes,
        silk: str,
        stakes: Optional[int],
        music: Optional[str],
        sex: UnibetHorseSex,
        age: Optional[int],
        team: int,
        coat: UnibetCoat,
        origins: Optional[str],
        comment: Optional[str],
        length: str,
        rope_n: Optional[int],
        owner: Optional[Person],
        trainer: Optional[Person],
        jockey: Optional[Person],
        horse: Optional[Horse],
        position: Optional[int],
        race_duration_sec: Optional[float],
        morning_odds: Optional[float],
        final_odds: Optional[float],
        db_session: SQLAlchemySession,
    ) -> "Runner":
        assert race

        found_runner: Optional[Runner] = (
            db_session.query(Runner).filter(Runner.unibet_id == unibet_id).one_or_none()
        )

        if found_runner is not None:

            assert found_runner.race_id == race.id
            assert found_runner.weight == weight
            assert found_runner.unibet_n == unibet_n
            assert found_runner.draw == draw
            assert (
                found_runner.blinkers == blinkers
            ), f"{found_runner.blinkers} != {blinkers}"
            assert found_runner.shoes == shoes, f"{found_runner.shoes} != {shoes}"
            assert found_runner.silk == silk
            assert found_runner.stakes == stakes
            assert found_runner.music == music
            assert found_runner.sex == sex
            assert found_runner.age == age, f"{found_runner.age} != {age}"
            assert found_runner.coat == coat, f"{found_runner.coat} != {coat}"
            assert found_runner.origins == origins
            assert found_runner.comment == comment
            assert found_runner.length == length
            assert found_runner.team == team, f"{found_runner.team} != {team}"
            assert found_runner.owner_id == (owner.id if owner else None)
            assert found_runner.trainer_id == (trainer.id if trainer else None)
            assert found_runner.jockey_id == (jockey.id if jockey else None)
            assert found_runner.horse_id == (
                horse.id if horse else None
            ), f"{found_runner.horse_id} != {horse.id if horse else None}"
            assert found_runner.position == (str(position) if position else None)
            assert (
                found_runner.race_duration_sec == race_duration_sec
            ), f"{found_runner.race_duration_sec} != {race_duration_sec}"
            assert found_runner.morning_odds == morning_odds
            assert found_runner.final_odds == final_odds
            assert found_runner.rope_n == rope_n
            assert found_runner.id
            return found_runner

        runner = Runner(
            unibet_id=unibet_id,
            race_id=race.id,
            weight=weight,
            unibet_n=unibet_n,
            draw=draw,
            blinkers=blinkers,
            shoes=shoes,
            silk=silk,
            stakes=stakes,
            music=music,
            sex=UnibetHorseSex(sex),
            age=age,
            team=team,
            coat=coat,
            origins=origins,
            comment=comment,
            length=length,
            rope_n=rope_n,
            owner_id=owner.id if owner else None,
            trainer_id=trainer.id if trainer else None,
            jockey_id=jockey.id if jockey else None,
            horse_id=horse.id if horse else None,
            position=position,
            race_duration_sec=race_duration_sec,
            morning_odds=morning_odds,
            final_odds=final_odds,
        )
        db_session.add(runner)
        db_session.commit()
        assert runner.id
        return runner
