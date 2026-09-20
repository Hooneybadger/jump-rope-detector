from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.database import Base, database_url
from app import models  # noqa: F401

config = context.config
rendered_url = database_url.render_as_string(hide_password=False) if hasattr(database_url, "render_as_string") else str(database_url)
# Alembic stores this value in ConfigParser, where percent signs trigger
# interpolation. URL-encoded passwords (for example, ``!`` -> ``%21``) must
# therefore escape percent signs before being assigned to sqlalchemy.url.
config.set_main_option("sqlalchemy.url", rendered_url.replace("%", "%%"))
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_offline() if context.is_offline_mode() else run_migrations_online()
