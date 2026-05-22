"""extend ck_transactions_fraud_decision to allow PENDING

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-22 17:01:00.000000+00:00

Phase 3B persists a stub Transaction row before scoring runs, so
fraud_decision needs a PENDING value alongside APPROVE / REVIEW /
DECLINE. Phase 3C will overwrite PENDING with the real decision on
every scored transaction; the value stays in the allowed set because
it's also useful as a transient state for rows that error mid-scoring.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply the migration."""
    with op.batch_alter_table('transactions', schema=None) as batch_op:
        batch_op.drop_constraint(
            'ck_transactions_fraud_decision', type_='check',
        )
        batch_op.create_check_constraint(
            'ck_transactions_fraud_decision',
            "fraud_decision IS NULL OR fraud_decision IN "
            "('APPROVE', 'REVIEW', 'DECLINE', 'PENDING')",
        )


def downgrade() -> None:
    """Revert the migration."""
    with op.batch_alter_table('transactions', schema=None) as batch_op:
        batch_op.drop_constraint(
            'ck_transactions_fraud_decision', type_='check',
        )
        batch_op.create_check_constraint(
            'ck_transactions_fraud_decision',
            "fraud_decision IS NULL OR fraud_decision IN "
            "('APPROVE', 'REVIEW', 'DECLINE')",
        )
