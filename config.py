import getpass
import os

ENVIRONMENT = os.environ["ENVIRONMENT"]
assert ENVIRONMENT in ("development", "production")

if ENVIRONMENT == "development":
    DB_HOST = "localhost:5435"
    DB_NAME = "benter"
    DB_PASSWORD = "mathieu"
    DB_USER = "mathieu"
    DB_POOL_SIZE = 10
else:
    # ENVIRONMENT == "production"
    DB_HOST = "benter-db.ce5eo53kxw1f.eu-west-3.rds.amazonaws.com:5432"
    DB_NAME = "postgres"
    DB_PASSWORD = getpass.getpass("What is production DB password?")
    DB_USER = "postgres"
    DB_POOL_SIZE = 10
