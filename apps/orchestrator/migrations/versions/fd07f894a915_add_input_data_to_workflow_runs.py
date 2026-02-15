"""add_input_data_to_workflow_runs

Revision ID: fd07f894a915
Revises: 3201bb1d2a40
Create Date: 2025-06-25 14:46:19.883580

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fd07f894a915'
down_revision: Union[str, None] = '3201bb1d2a40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add input_data column to the workflow_runs table
    op.add_column('workflow_runs', sa.Column('input_data', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove input_data column from the workflow_runs table
    op.drop_column('workflow_runs', 'input_data')
