# pylint: disable=invalid-name
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from config import DB_HOST
from config import DB_NAME
from config import DB_PASSWORD
from config import DB_POOL_SIZE
from config import DB_USER

log = logging.getLogger(__name__)

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    pool_size=DB_POOL_SIZE,
    convert_unicode=True,
    pool_pre_ping=True,
    connect_args={"options": "-c statement_timeout=1000000"},
)

# expire_on_commit is needed to keep in memory objects for Dataloaders

SQLAlchemySession = scoped_session(
    sessionmaker(bind=engine, autoflush=True, expire_on_commit=False)
)


@contextmanager
def create_sqlalchemy_session():
    """Provide a transactional scope around a series of operations"""
    sqlalchemy_session = SQLAlchemySession()
    try:
        yield sqlalchemy_session
    except Exception as err:
        log.error(err)
        sqlalchemy_session.rollback()
        raise
    else:
        sqlalchemy_session.commit()
    finally:
        sqlalchemy_session.close()
