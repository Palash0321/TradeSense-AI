"""create execution fills

Revision ID: c9e8f7d6a5b4
Revises: b7c6d5e4f3a2
Create Date: 2026-09-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c9e8f7d6a5b4"
down_revision: Union[str, Sequence[str], None] = "b7c6d5e4f3a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "execution_fills",
        sa.Column("id", sa.Integer(), nullable=False),
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
            "broker_order_id",
            sa.String(length=100),
            nullable=True,
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
            "filled_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            ["execution_records.execution_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_execution_fills_id"),
        "execution_fills",
        ["id"],
        unique=False,
    )

    op.create_index(
        op.f("ix_execution_fills_fill_id"),
        "execution_fills",
        ["fill_id"],
        unique=True,
    )

    op.create_index(
        op.f("ix_execution_fills_execution_id"),
        "execution_fills",
        ["execution_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_execution_fills_execution_id"),
        table_name="execution_fills",
    )

    op.drop_index(
        op.f("ix_execution_fills_fill_id"),
        table_name="execution_fills",
    )

    op.drop_index(
        op.f("ix_execution_fills_id"),
        table_name="execution_fills",
    )

    op.drop_table("execution_fills")