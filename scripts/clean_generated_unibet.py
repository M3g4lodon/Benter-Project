from database.setup import create_sqlalchemy_session
from models import Horse
from models import Runner
from scripts.generate_unibet import _extract_name_country


def run():  # pylint:disable=too-many-branches
    with create_sqlalchemy_session() as db_session:
        db_session.query(Horse).filter(Horse.country_code == "<HTML>").update(
            {"country_code": None}
        )
        db_session.query(Horse).filter(Horse.country_code == "1005750").update(
            {"country_code": None}
        )
        db_session.query(Runner).filter(Runner.weight > 200).update({"weight": None})

        db_session.commit()

        nc_horses = (
            db_session.query(Horse)
            .filter(Horse.name.op("~")(".*[\(\{](.*)[\)\}]"))
            .all()
        )
        for horse in nc_horses:
            name, country_code = _extract_name_country(name_country=horse.name)
            if country_code:
                horse.name = name
                horse.country_code = country_code

        db_session.commit()


if __name__ == "__main__":
    run()
