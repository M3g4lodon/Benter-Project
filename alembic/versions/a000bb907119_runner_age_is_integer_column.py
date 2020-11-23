"""runner.age is Integer column

Revision ID: a000bb907119
Revises: f1c86649e545
Create Date: 2020-11-23 17:33:59.501129

"""
import sqlalchemy as sa

from alembic import op


# revision identifiers, used by Alembic.
revision = "a000bb907119"
down_revision = "f1c86649e545"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        UPDATE runners
        SET age = null
        WHERE age='';
        """
    )
    op.alter_column("runners", "age", type_=sa.Integer, postgresql_using="age::bigint")
    op.execute(
        """
        UPDATE runners
        SET age = null
        WHERE age>100;
        """
    )


def downgrade():
    op.alter_column(
        "runners", "age", type_=sa.String, postgresql_using="age::varchar(255)"
    )
