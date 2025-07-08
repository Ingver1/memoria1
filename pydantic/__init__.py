from __future__ import annotations

import json
from typing import Any, Callable, TypeVar

__all__ = [
    "BaseModel",
    "BaseSettings",
    "ValidationError",
    "PositiveInt",
    "SecretStr",
    "Field",
    "field_validator",
    "ValidationInfo",
]

T = TypeVar("T")

class ValidationError(Exception):
    pass

class ValidationInfo:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or {}

class BaseModel:
    model_config: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls: type[T], data: Any) -> T:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(**data)
        raise ValidationError("Invalid data")

    def model_dump(self, *, mode: str | None = None) -> dict[str, Any]:
        return dict(self.__dict__)

    def model_dump_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.__dict__, indent=indent)

class BaseSettings(BaseModel):
    pass

class SecretStr(str):
    def get_secret_value(self) -> str:
        return str(self)

PositiveInt = int

F = TypeVar("F", bound=Callable[..., Any])

def Field(default: Any = ..., **_: Any) -> Any:
    return default

def field_validator(*_: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        return func
    return decorator
