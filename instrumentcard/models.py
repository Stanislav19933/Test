"""Input and output models with conservative validation."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    name: str = Field(default="", max_length=300)
    manufacturer: str = Field(default="", max_length=200)
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = Field(default="", max_length=20000)
    expected_specs: list[str] = Field(default_factory=list, max_length=25)
    source_url: str = Field(default="", max_length=2048)

    @field_validator("name", "manufacturer", "description", "source_url", mode="before")
    @classmethod
    def strings_only(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("Значение должно быть строкой")
        return value.strip()

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        if not value:
            return ""
        parsed = urlsplit(value)
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise ValueError("source_url должен быть абсолютным http(s)-адресом")
        return value

    @field_validator("specs", mode="before")
    @classmethod
    def normalize_specs(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Поле specs должно быть объектом JSON")
        if len(value) > 80:
            raise ValueError("Максимум 80 характеристик")
        result: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or len(key.strip()) > 150:
                raise ValueError("Названия характеристик должны быть непустыми строками до 150 символов")
            if item is not None and not isinstance(item, (str, int, float)):
                raise ValueError(f"Характеристика «{key}» должна быть строкой, числом или null")
            result[key.strip()] = "" if item is None else str(item).strip()[:1001]
            if len(result[key.strip()]) > 1000:
                raise ValueError(f"Характеристика «{key}» слишком длинная")
        return result

    @field_validator("expected_specs", mode="before")
    @classmethod
    def normalize_expected(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list) or len(value) > 25:
            raise ValueError("expected_specs должен быть списком не более 25 названий")
        if any(not isinstance(item, str) or not item.strip() or len(item.strip()) > 150 for item in value):
            raise ValueError("expected_specs содержит некорректное название")
        return list(dict.fromkeys(item.strip() for item in value))


class GeneratedText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial_block: str


class CardOutput(BaseModel):
    product_name: str
    manufacturer: str
    seo_title: str
    seo_description: str
    characteristics: dict[str, str]
    search_queries: list[str]
    commercial_block: str
    missing_fields: list[str]
    review_warnings: list[str]
    source_url: str = ""
    automated_checks_passed: bool = False
