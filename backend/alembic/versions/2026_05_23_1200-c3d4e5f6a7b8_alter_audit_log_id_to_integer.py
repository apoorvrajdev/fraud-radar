"""alter audit_log.id from BigInteger to Integer for SQLite autoincrement

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-23 12:00:00.000000+00:00

SQLite only treats PRIMARY KEY columns of declared type INTEGER as
auto-incrementing ROWID aliases. BIGINT PRIMARY KEY is stored but never
auto-incremented, so INSERT without an explicit `id` fails with a NOT
NULL constraint violation — `autoincrement=True` is a no-op on SQLite
when the underlying type isn't exactly INTEGER.

The audit_log table was created in 4e0e08a83d4d (Phase 2A) with
BigInteger and never written to during Phase 2. Phase 3C-2's scoring
service is the first writer, which surfaced the bug. Switching to
Integer fixes it on SQLite (dev) and is fine on Postgres (production
target): 4-byte signed ints give ~2.1B rows, plenty of headroom for
any audit log this project will ever hold.

The table is empty at the time of this migration, so the column-type
change is non-destructive.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply the migration."""
    with op.batch_alter_table('audit_log', schema=None) as batch_op:
        batch_op.alter_column(
            'id',
            existing_type=sa.BigInteger(),
            type_=sa.Integer(),
            existing_nullable=False,
            existing_server_default=None,
            autoincrement=True,
        )


def downgrade() -> None:
    """Revert the migration."""
    with op.batch_alter_table('audit_log', schema=None) as batch_op:
        batch_op.alter_column(
            'id',
            existing_type=sa.Integer(),
            type_=sa.BigInteger(),
            existing_nullable=False,
            existing_server_default=None,
            autoincrement=True,
        )
