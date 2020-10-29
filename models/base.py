from sqlalchemy.ext.declarative import declarative_base

from database.setup import SQLAlchemySession

Base = declarative_base()  # pylint: disable=invalid-name
Base.query = SQLAlchemySession.query_property()
