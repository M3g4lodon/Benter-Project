"""Create first tables

Revision ID: 354c2b3d4331
Revises:
Create Date: 2020-10-29 08:35:48.007831

"""
import sqlalchemy as sa

from alembic import op


# revision identifiers, used by Alembic.
revision = "1"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "race_tracks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("race_track_name", sa.String, nullable=False),
        sa.Column("country_name", sa.String, nullable=True),
    )


def downgrade():
    op.drop_table("race_tracks")
