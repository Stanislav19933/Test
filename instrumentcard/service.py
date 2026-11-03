"""Card pipeline combining trusted source fields and generated editorial copy."""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from .zai_client import GenerationError, generate_text
from .models import CardOutput, GeneratedText, ProductInput
from .quality import characteristic_list, missing_fields, validate_draft


class InputError(Exception):
    """User-facing invalid input error."""


class IncompleteCardError(GenerationError):
    """Nonconforming card to show, but never cache as completed."""

    def __init__(self, card: CardOutput) -> None:
        super().__init__("Карточка требует доработки: автоматические проверки выявили замечания.")
        self.card = card


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


def _draft_score(product: ProductInput, draft: GeneratedText) -> tuple[int, int, int]:
    issues = validate_draft(product, draft)
    unsafe = any("Числовые значения отсутствуют" in item for item in issues)
    n = len(draft.seo_description)
    distance = max(0, 800 - n, n - 1200)
    return (int(unsafe), len(issues), distance)


def _revision_instructions(draft: GeneratedText, problems: list[str]) -> list[str]:
    result = [*problems]
    if not 800 <= len(draft.seo_description) <= 1200:
        result.append(
            f"Предыдущее SEO-описание имело {len(draft.seo_description)} символов. "
            "Полностью перепиши его объёмом 900–1050 символов, разделив на 2–3 "
            "связных абзаца. Используй исключительно подтверждённые факты "
            "из исходного JSON; запрещено придумывать новые свойства, "
            "повторять фразы и растягивать текст пустыми словами."
        )
    return result


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
        try:
            revised = generator(product, api_key, _revision_instructions(draft, problems))
        except GenerationError:
            # Preserve a useful first response if a corrective API call gets 429.
            problems.append("Повторная генерация временно недоступна; показан первый черновик с замечаниями.")
        else:
            revised_problems = validate_draft(product, revised)
            if _draft_score(product, revised) < _draft_score(product, draft):
                draft, problems = revised, revised_problems

    missing = missing_fields(product)
    if len(product.description) < 100 and len(product.specs) < 3:
        problems.append("Исходных сведений мало: для длинного SEO-текста нужна редакторская проверка фактов.")
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
        automated_checks_passed=not bool(missing or validate_draft(product, draft))
        and not (len(product.description) < 100 and len(product.specs) < 3),
    )
