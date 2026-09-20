from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginInput(StrictModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=200)


class UserCreate(StrictModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,50}$")
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=200)
    role: str = "member"
    can_basic: bool = True
    can_alternating: bool = True
    can_double: bool = False
    can_view_history: bool = True

    @field_validator("role")
    @classmethod
    def role_is_valid(cls, value: str) -> str:
        if value not in {"admin", "member"}:
            raise ValueError("role must be admin or member")
        return value


class UserUpdate(StrictModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    password: str | None = Field(default=None, min_length=12, max_length=200)
    role: str | None = None
    is_active: bool | None = None
    can_basic: bool | None = None
    can_alternating: bool | None = None
    can_double: bool | None = None
    can_view_history: bool | None = None

    @field_validator("role")
    @classmethod
    def role_is_valid(cls, value: str | None) -> str | None:
        if value is not None and value not in {"admin", "member"}:
            raise ValueError("role must be admin or member")
        return value
