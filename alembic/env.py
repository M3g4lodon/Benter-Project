from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from config import DB_HOST
from config import DB_NAME
from config import DB_PASSWORD
from config import DB_USER

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.

config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# pylint:disable=wrong-import-position
from models.base import Base
from models.horse import Horse  # pylint:disable=unused-import
from models.horse_show import HorseShow  # pylint:disable=unused-import
from models.person import Person  # pylint:disable=unused-import
from models.race import Race  # pylint:disable=unused-import
from models.race_track import RaceTrack  # pylint:disable=unused-import
from models.runner import Runner  # pylint:disable=unused-import

# pylint:enable=wrong-import-position
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    return f"postgres+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


config.set_main_option("sqlalchemy.url", get_url())
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
