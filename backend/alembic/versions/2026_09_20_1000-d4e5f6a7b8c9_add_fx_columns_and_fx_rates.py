"""add FX enrichment columns and the fx_rates cache table

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-09-20 10:00:00.000000+00:00

Phase 5F. Four additive, nullable columns on `transactions` carrying the
derived reporting-currency figure and its provenance, plus the local
cache of historical reference rates that produces them.

Additive throughout: `amount` and `currency` are untouched, every new
column is nullable, and rows written before this migration stay valid —
they read as "not enriched", which is exactly what they are. See
docs/FX_CONTRACT.md.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# The six values of `transactions.fx_source`, in resolution order:
# identity (same currency, no lookup), cache, live, stale, then the two
# no-rate outcomes — unsupported (permanent) and unavailable (today).
_FX_SOURCE_CHECK = (
    "fx_source IS NULL OR fx_source IN "
    "('identity', 'live', 'cache', 'stale', 'unsupported', 'unavailable')"
)

# A half-converted row must not exist: either we have both the derived
# amount and the rate that produced it, or we have neither.
_FX_PAIRING_CHECK = (
    "(amount_base IS NULL AND fx_rate IS NULL) OR "
    "(amount_base IS NOT NULL AND fx_rate IS NOT NULL)"
)


def upgrade() -> None:
    """Apply the migration."""
    op.create_table(
        'fx_rates',
        sa.Column('base', sa.String(length=3), nullable=False),
        sa.Column('quote', sa.String(length=3), nullable=False),
        sa.Column('rate_date', sa.Date(), nullable=False),
        sa.Column('rate', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('fetched_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.CheckConstraint('rate > 0', name='ck_fx_rates_rate_positive'),
        sa.CheckConstraint('base <> quote', name='ck_fx_rates_distinct_pair'),
        sa.PrimaryKeyConstraint('base', 'quote', 'rate_date'),
    )

    with op.batch_alter_table('transactions', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'amount_base', sa.Numeric(precision=19, scale=4), nullable=True
            )
        )
        batch_op.add_column(
            sa.Column(
                'fx_rate', sa.Numeric(precision=18, scale=8), nullable=True
            )
        )
        batch_op.add_column(sa.Column('fx_rate_date', sa.Date(), nullable=True))
        batch_op.add_column(
            sa.Column('fx_source', sa.String(length=16), nullable=True)
        )
        batch_op.create_check_constraint(
            'ck_transactions_fx_source', _FX_SOURCE_CHECK,
        )
        batch_op.create_check_constraint(
            'ck_transactions_fx_rate_positive',
            'fx_rate IS NULL OR fx_rate > 0',
        )
        batch_op.create_check_constraint(
            'ck_transactions_fx_amount_pairing', _FX_PAIRING_CHECK,
        )


def downgrade() -> None:
    """Revert the migration."""
    with op.batch_alter_table('transactions', schema=None) as batch_op:
        batch_op.drop_constraint(
            'ck_transactions_fx_amount_pairing', type_='check',
        )
        batch_op.drop_constraint(
            'ck_transactions_fx_rate_positive', type_='check',
        )
        batch_op.drop_constraint('ck_transactions_fx_source', type_='check')
        batch_op.drop_column('fx_source')
        batch_op.drop_column('fx_rate_date')
        batch_op.drop_column('fx_rate')
        batch_op.drop_column('amount_base')

    op.drop_table('fx_rates')
