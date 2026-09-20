"""create positions

Revision ID: b7c6d5e4f3a2
Revises: 13c05006d297
Create Date: 2026-09-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7c6d5e4f3a2"
down_revision: Union[str, Sequence[str], None] = "13c05006d297"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the positions table."""
    op.create_table(
        "positions",
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
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "symbol",
            sa.String(length=20),
            nullable=False,
        ),
        sa.Column(
            "direction",
            sa.String(length=10),
            nullable=False,
        ),
        sa.Column(
            "quantity",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "average_entry_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "current_price",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "stop_loss",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "target1",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "target2",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "target3",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "state",
            sa.String(length=30),
            nullable=False,
        ),
        sa.Column(
            "opened_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "closed_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_positions_id"),
        "positions",
        ["id"],
        unique=False,
    )

    op.create_index(
        op.f("ix_positions_position_id"),
        "positions",
        ["position_id"],
        unique=True,
    )

    op.create_index(
        op.f("ix_positions_user_id"),
        "positions",
        ["user_id"],
        unique=False,
    )

    op.create_index(
        op.f("ix_positions_symbol"),
        "positions",
        ["symbol"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the positions table."""
    op.drop_index(
        op.f("ix_positions_symbol"),
        table_name="positions",
    )

    op.drop_index(
        op.f("ix_positions_user_id"),
        table_name="positions",
    )

    op.drop_index(
        op.f("ix_positions_position_id"),
        table_name="positions",
    )

    op.drop_index(
        op.f("ix_positions_id"),
        table_name="positions",
    )

    op.drop_table("positions")