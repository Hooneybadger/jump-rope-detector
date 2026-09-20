"""Remove result delivery now that users manage stored workout records."""

import sqlalchemy as sa
from alembic import op

revision = "0003_remove_result_delivery"
down_revision = "0002_accounts_and_deliveries"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("result_deliveries")


def downgrade() -> None:
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
