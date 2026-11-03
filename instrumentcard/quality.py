"""Deterministic checks; numeric claims are only a heuristic, not fact verification."""

from __future__ import annotations

import re

from .models import GeneratedText, ProductInput

# A product model code, range limit or number in generated prose must be present
# in the original source. This does NOT prove factual correctness.
NUMERIC_RE = re.compile(r"(?<!\d)\d+(?:[,.]\d+)?(?:/\d+)?")


def normalized_numbers(text: str) -> set[str]:
    return {match.group().replace(",", ".") for match in NUMERIC_RE.finditer(text)}


def numeric_claim_warnings(product: ProductInput, draft: GeneratedText) -> list[str]:
    source = " ".join([product.name, product.manufacturer, product.description, *product.specs.keys(), *product.specs.values()])
    allowed = normalized_numbers(source)
    output = " ".join([draft.seo_title, draft.seo_description, draft.commercial_block, *draft.search_queries])
    unknown = sorted(normalized_numbers(output) - allowed)
    if unknown:
        return ["Числовые значения отсутствуют в источнике: " + ", ".join(unknown) + ". Требуется проверка."]
    return []


def validate_draft(product: ProductInput, draft: GeneratedText) -> list[str]:
    problems: list[str] = []
    if not 1 <= len(draft.seo_title.strip()) <= 70:
        problems.append(f"SEO-заголовок: {len(draft.seo_title)} символов вместо 1–70")
    # A short SEO draft is always reported as nonconforming, including sparse inputs.
    if not 800 <= len(draft.seo_description) <= 1200:
        problems.append(f"SEO-описание: {len(draft.seo_description)} символов вместо 800–1200")
    if not 3 <= len(draft.search_queries) <= 5:
        problems.append("Нужно 3–5 поисковых запросов")
    if any(not query.strip() for query in draft.search_queries):
        problems.append("Среди поисковых запросов есть пустой")
    if not draft.commercial_block.strip():
        problems.append("Коммерческий блок пустой")
    problems.extend(numeric_claim_warnings(product, draft))
    return problems


def characteristic_list(product: ProductInput) -> dict[str, str]:
    """Preserve source values verbatim. Only show requested absent keys as no data."""
    result = {key: value if value else "данных нет" for key, value in product.specs.items()}
    for expected in product.expected_specs:
        if expected.casefold() not in {key.casefold() for key in result}:
            result[expected] = "данных нет"
    return result


def missing_fields(product: ProductInput) -> list[str]:
    missing: list[str] = []
    if not product.name:
        missing.append("Название товара")
    if not product.manufacturer:
        missing.append("Производитель")
    if not product.specs:
        missing.append("Ключевые характеристики")
    if not product.description:
        missing.append("Описание / фрагмент паспорта")
    for key, value in characteristic_list(product).items():
        if value == "данных нет":
            missing.append(f"Характеристика: {key}")
    return missing
