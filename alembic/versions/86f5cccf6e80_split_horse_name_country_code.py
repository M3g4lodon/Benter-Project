"""split_horse_name_country_code

Revision ID: 86f5cccf6e80
Revises: a000bb907119
Create Date: 2020-11-25 08:35:56.709201

"""
import sqlalchemy as sa

from alembic import op


# revision identifiers, used by Alembic.
revision = "86f5cccf6e80"
down_revision = "a000bb907119"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("horses", sa.Column("country_code", sa.String(), nullable=True))
    op.execute(
        """
        UPDATE horses
        SET country_code = regexp_replace(regexp_replace("name" , '.*\(', ''), '\)', '');
        """
    )
    op.execute(
        """
        UPDATE horses
        SET name = regexp_replace("name" , '\s\(.*\)', '');
        """
    )
    op.create_index(
        op.f("ix_horses_country_code"), "horses", ["country_code"], unique=False
    )


def downgrade():
    op.drop_index(op.f("ix_horses_country_code"), table_name="horses")
    op.drop_column("horses", "country_code")
