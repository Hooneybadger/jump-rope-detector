from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Jump Rope Lab"
    environment: str = "development"
    database_url: str = ""
    db_host: str = ""
    db_port: int = 5432
    db_name: str = "jump_rope"
    db_user: str = "jump_rope"
    db_password: str = ""
    admin_username: str = "admin"
    admin_password: str = ""
    admin_display_name: str = "관리자"
    session_hours: int = Field(default=12, ge=1, le=168)
    cookie_secure: bool = False
    allowed_hosts: str = "localhost,127.0.0.1,testserver"
    allowed_origins: str = "http://localhost:8080,http://127.0.0.1:8080,http://testserver"
    max_frame_bytes: int = Field(default=2_000_000, ge=100_000, le=8_000_000)
    max_concurrent_streams: int = Field(default=4, ge=1, le=32)

    @field_validator("admin_password")
    @classmethod
    def validate_admin_password(cls, value: str) -> str:
        if value and len(value) < 12:
            raise ValueError("ADMIN_PASSWORD must be at least 12 characters")
        return value

    @property
    def hosts(self) -> list[str]:
        return [item.strip() for item in self.allowed_hosts.split(",") if item.strip()]

    @property
    def origins(self) -> set[str]:
        return {item.strip().rstrip("/") for item in self.allowed_origins.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
