import sqlalchemy as sa
from sqlalchemy.orm import relationship

from models.base import Base


class Stable(Base):
    __tablename__ = "stables"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)