"""Copy-friendly Markdown export."""

from __future__ import annotations

from .models import CardOutput


def to_markdown(card: CardOutput) -> str:
    lines = [
        f"# {card.seo_title}", "", "## SEO-описание", "", card.seo_description,
        "", "## Ключевые характеристики", "",
    ]
    lines.extend(f"- **{key}:** {value}" for key, value in card.characteristics.items())
    lines.extend(["", "## Поисковые запросы", ""])
    lines.extend(f"- {query}" for query in card.search_queries)
    lines.extend(["", "## Для менеджера", "", card.commercial_block, ""])
    if card.missing_fields:
        lines.extend(["## Недостающие данные", ""])
        lines.extend(f"- {field}" for field in card.missing_fields)
        lines.append("")
    lines.extend(["## Проверка", ""])
    lines.extend(f"- {message}" for message in card.review_warnings)
    if card.source_url:
        lines.extend(["", f"Источник: {card.source_url}"])
    return "\n".join(lines) + "\n"
