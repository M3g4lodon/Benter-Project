from database.setup import create_sqlalchemy_session
from models.horse import Horse
from models.horse_show import HorseShow
from models.person import Person
from models.race import Race
from models.race_track import RaceTrack
from models.runner import Runner

if __name__ == "__main__":
    with create_sqlalchemy_session() as db_session:
        db_session.query(HorseShow).delete()
        db_session.query(RaceTrack).delete()
        db_session.query(Person).delete()
        db_session.query(Race).delete()
        db_session.query(Horse).delete()
        db_session.query(Runner).delete()
        db_session.commit()
    print("All records are deleted!")
