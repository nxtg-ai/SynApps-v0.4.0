"""add completed_applets to workflow_runs

Revision ID: 8e8ca3c65593
Revises: fd07f894a915
Create Date: 2026-02-17 16:24:57.037073

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '8e8ca3c65593'
down_revision: str | None = 'fd07f894a915'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('workflow_runs', sa.Column('completed_applets', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('workflow_runs', 'completed_applets')
