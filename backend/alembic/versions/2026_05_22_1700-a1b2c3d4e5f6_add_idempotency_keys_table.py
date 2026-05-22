"""add idempotency_keys table

Revision ID: a1b2c3d4e5f6
Revises: 4e0e08a83d4d
Create Date: 2026-05-22 17:00:00.000000+00:00

Stripe-style idempotency cache used by POST /api/v1/transactions to detect
replays. See docs/adr/PHASE_3_DESIGN.md "Idempotency design".
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '4e0e08a83d4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply the migration."""
    op.create_table(
        'idempotency_keys',
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('request_hash', sa.String(length=64), nullable=False),
        sa.Column('transaction_id', sa.String(length=36), nullable=False),
        sa.Column('response_body', sa.Text(), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('expires_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.CheckConstraint(
            'expires_at > created_at',
            name='ck_idempotency_keys_expires_after_created',
        ),
        sa.ForeignKeyConstraint(
            ['transaction_id'], ['transactions.id'], ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('key'),
    )
    with op.batch_alter_table('idempotency_keys', schema=None) as batch_op:
        batch_op.create_index(
            'ix_idempotency_keys_request_hash',
            ['request_hash'],
            unique=False,
        )
        batch_op.create_index(
            'ix_idempotency_keys_expires_at',
            ['expires_at'],
            unique=False,
        )


def downgrade() -> None:
    """Revert the migration."""
    with op.batch_alter_table('idempotency_keys', schema=None) as batch_op:
        batch_op.drop_index('ix_idempotency_keys_expires_at')
        batch_op.drop_index('ix_idempotency_keys_request_hash')
    op.drop_table('idempotency_keys')
