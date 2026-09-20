"""Add member email, password resets, timed workouts, and result delivery."""

import sqlalchemy as sa
from alembic import op

revision = "0002_accounts_and_deliveries"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("email", sa.String(length=254), nullable=True))
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.add_column("workouts", sa.Column("target_duration_seconds", sa.Integer(), nullable=True))
    op.create_table(
        "password_reset_tokens",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_password_reset_tokens_token_hash", "password_reset_tokens", ["token_hash"], unique=True)
    op.create_index("ix_password_reset_tokens_user_id", "password_reset_tokens", ["user_id"])
    op.create_index("ix_password_reset_tokens_expires_at", "password_reset_tokens", ["expires_at"])
    op.create_table(
        "result_deliveries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("workout_id", sa.Integer(), sa.ForeignKey("workouts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("channel", sa.String(length=20), nullable=False),
        sa.Column("destination", sa.String(length=254), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("provider_message_id", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_result_deliveries_workout_id", "result_deliveries", ["workout_id"])
    op.create_index("ix_result_deliveries_user_id", "result_deliveries", ["user_id"])
    op.create_index("ix_result_deliveries_channel", "result_deliveries", ["channel"])
    op.create_index("ix_result_deliveries_status", "result_deliveries", ["status"])
    op.create_index("ix_result_delivery_workout_created", "result_deliveries", ["workout_id", "created_at"])


def downgrade() -> None:
    op.drop_table("result_deliveries")
    op.drop_table("password_reset_tokens")
    op.drop_column("workouts", "target_duration_seconds")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_column("users", "email")
