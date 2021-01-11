from database.setup import create_sqlalchemy_session
from models.horse import Horse
from models.horse_show import HorseShow
from models.person import Person
from models.race import Race
from models.race_track import RaceTrack
from models.runner import Runner

if __name__ == "__main__":
    with create_sqlalchemy_session() as db_session:
        print(f"{db_session.query(HorseShow).delete()} horse shows deleted")
        print(f"{db_session.query(RaceTrack).delete()} race tracks deleted")
        print(f"{db_session.query(Person).delete()} persons deleted")
        print(f"{db_session.query(Race).delete()} races deleted")
        print(f"{db_session.query(Horse).delete()} horses deleted")
        print(f"{db_session.query(Runner).delete()} runners deleted")
        db_session.commit()
    print("All records are deleted!")
