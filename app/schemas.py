from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginInput(StrictModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=200)


class SignupInput(StrictModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,50}$")
    email: str = Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$", max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=200)


class PasswordResetRequest(StrictModel):
    email: str = Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$", max_length=254)


class PasswordResetConfirm(StrictModel):
    token: str = Field(min_length=32, max_length=200)
    password: str = Field(min_length=12, max_length=200)


class ProfileUpdate(StrictModel):
    display_name: str = Field(min_length=1, max_length=80)
    email: str | None = Field(default=None, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$", max_length=254)
    current_password: str | None = Field(default=None, min_length=1, max_length=200)
    new_password: str | None = Field(default=None, min_length=12, max_length=200)


class BulkDeleteInput(StrictModel):
    ids: list[int] = Field(min_length=1, max_length=500)

    @field_validator("ids")
    @classmethod
    def ids_are_positive_and_unique(cls, value: list[int]) -> list[int]:
        if any(item < 1 for item in value):
            raise ValueError("ids must contain positive integers")
        return list(dict.fromkeys(value))


class UserCreate(StrictModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,50}$")
    email: str | None = Field(default=None, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$", max_length=254)
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
