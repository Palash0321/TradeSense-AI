"""create_position_fill_allocations

Revision ID: 45a973de430c
Revises: c9e8f7d6a5b4
Create Date: 2026-09-21 22:43:42.848469

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '45a973de430c'
down_revision: Union[str, Sequence[str], None] = 'c9e8f7d6a5b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "position_fill_allocations",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "position_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "fill_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "execution_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "fill_sequence",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "fill_quantity",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "fill_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "applied_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["position_id"],
            ["positions.position_id"],
        ),
        sa.ForeignKeyConstraint(
            ["fill_id"],
            ["execution_fills.fill_id"],
        ),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            ["execution_records.execution_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "fill_id",
            name="uq_position_fill_allocations_fill_id",
        ),
    )

    op.create_index(
        "ix_position_fill_allocations_id",
        "position_fill_allocations",
        ["id"],
        unique=False,
    )

    op.create_index(
        "ix_position_fill_allocations_position_id",
        "position_fill_allocations",
        ["position_id"],
        unique=False,
    )

    op.create_index(
        "ix_position_fill_allocations_fill_id",
        "position_fill_allocations",
        ["fill_id"],
        unique=True,
    )

    op.create_index(
        "ix_position_fill_allocations_execution_id",
        "position_fill_allocations",
        ["execution_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_position_fill_allocations_execution_id",
        table_name="position_fill_allocations",
    )

    op.drop_index(
        "ix_position_fill_allocations_fill_id",
        table_name="position_fill_allocations",
    )

    op.drop_index(
        "ix_position_fill_allocations_position_id",
        table_name="position_fill_allocations",
    )

    op.drop_index(
        "ix_position_fill_allocations_id",
        table_name="position_fill_allocations",
    )

    op.drop_table("position_fill_allocations")
