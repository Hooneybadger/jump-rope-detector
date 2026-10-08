"""Add the optional custom body profile and a profile photo."""

import sqlalchemy as sa
from alembic import op

revision = "0004_custom_profile"
down_revision = "0003_remove_result_delivery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Accounts that already exist have logged in before, so they are not asked again on login.
    # New accounts get "pending" from the ORM default.
    op.add_column("users", sa.Column("profile_status", sa.String(length=20), nullable=False, server_default="skipped"))
    op.add_column("users", sa.Column("sex", sa.String(length=10), nullable=True))
    op.add_column("users", sa.Column("age", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("height_cm", sa.Float(), nullable=True))
    op.add_column("users", sa.Column("weight_kg", sa.Float(), nullable=True))
    op.add_column("users", sa.Column("avatar", sa.LargeBinary(), nullable=True))
    op.add_column("users", sa.Column("avatar_type", sa.String(length=20), nullable=True))
    op.add_column("users", sa.Column("avatar_updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    for column in ("avatar_updated_at", "avatar_type", "avatar", "weight_kg", "height_cm", "age", "sex", "profile_status"):
        op.drop_column("users", column)
