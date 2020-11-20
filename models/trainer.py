import sqlalchemy as sa
from sqlalchemy.orm import relationship

from models.base import Base


class Trainer(Base):
    __tablename__ = "trainers"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, unique=False, nullable=False, index=True)

    runners = relationship("Runner", backref="trainer")
