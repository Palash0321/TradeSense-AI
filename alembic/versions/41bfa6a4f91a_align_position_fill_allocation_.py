"""align_position_fill_allocation_uniqueness

Revision ID: 41bfa6a4f91a
Revises: 45a973de430c
Create Date: 2026-09-21 22:49:19.020434

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '41bfa6a4f91a'
down_revision: Union[str, Sequence[str], None] = '45a973de430c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_position_fill_allocations_fill_id",
        "position_fill_allocations",
        type_="unique",
    )


def downgrade() -> None:
    op.create_unique_constraint(
        "uq_position_fill_allocations_fill_id",
        "position_fill_allocations",
        ["fill_id"],
    )
