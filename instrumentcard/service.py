"""Card pipeline combining trusted source fields and generated editorial copy."""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from .groq_client import GenerationError, generate_text
from .models import CardOutput, GeneratedText, ProductInput
from .quality import characteristic_list, missing_fields, validate_draft


class InputError(Exception):
    """User-facing invalid input error."""


def parse_product(text: str) -> ProductInput:
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"Некорректный JSON: строка {exc.lineno}, колонка {exc.colno}.") from exc
    if not isinstance(data, dict):
        raise InputError("На верхнем уровне JSON должен быть объектом товара.")
    try:
        return ProductInput.model_validate(data)
    except ValidationError as exc:
        field_errors = [".".join(map(str, err["loc"])) + ": " + err["msg"] for err in exc.errors()]
        raise InputError("Некорректные данные: " + "; ".join(field_errors)) from exc


def generate_card(
    product: ProductInput,
    api_key: str,
    generator: Callable[[ProductInput, str, list[str] | None], GeneratedText] | None = None,
) -> CardOutput:
    generator = generator or (lambda p, key, corrections: generate_text(p, key, corrections))
    if not product.name:
        raise GenerationError("Необходимо заполнить название товара.")
    if not product.description and not product.specs:
        raise GenerationError("Укажите хотя бы одну характеристику или описание товара.")
    draft = generator(product, api_key, None)
    problems = validate_draft(product, draft)
    if problems:
        revised = generator(product, api_key, problems)
        # Prefer the revision when it fixes more validation issues. If still
        # insufficient, show the draft with explicit warnings; never label ready.
        revised_problems = validate_draft(product, revised)
        if len(revised_problems) <= len(problems):
            draft, problems = revised, revised_problems

    missing = missing_fields(product)
    if len(product.description) < 100 and len(product.specs) < 3:
        problems.append("Недостаточно исходных данных для гарантированного описания 800–1200 символов.")
    if missing:
        problems.append("Перед публикацией необходимо проверить неполные исходные данные.")
    # Structured output is validated; semantic correctness still needs human review.
    problems.append("Перед публикацией контент-менеджер должен проверить фактические утверждения.")
    return CardOutput(
        product_name=product.name,
        manufacturer=product.manufacturer or "данных нет",
        seo_title=draft.seo_title,
        seo_description=draft.seo_description,
        characteristics=characteristic_list(product),
        search_queries=draft.search_queries,
        commercial_block=draft.commercial_block,
        missing_fields=missing,
        review_warnings=problems,
        source_url=product.source_url,
        automated_checks_passed=not bool(missing or validate_draft(product, draft)),
    )
