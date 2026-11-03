"""Streamlit entry point for InstrumentCard AI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from instrumentcard.formatting import to_markdown
from instrumentcard.groq_client import GenerationError
from instrumentcard.models import CardOutput, GeneratedText
from instrumentcard.quality import characteristic_list, missing_fields
from instrumentcard.service import InputError, generate_card, parse_product

ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "Термогигрометр CEM DT-83": ROOT / "samples" / "dt83.json",
    "Манометр РОСМА ТМ-510": ROOT / "samples" / "tm510.json",
}


def api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", ""))
    except (FileNotFoundError, KeyError):
        return ""


def main() -> None:
    st.set_page_config(page_title="InstrumentCard AI", page_icon="📐", layout="wide")
    st.markdown("""<style>
    .block-container {padding-top:2rem;max-width:1200px}
    h1 {letter-spacing:-.04em}
    [data-testid='stMetricValue'] {font-size:1.6rem}
    </style>""", unsafe_allow_html=True)
    st.title("InstrumentCard AI")
    st.caption("B2B-карточки измерительных приборов · проверка данных · Qwen 3.8 через Groq")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("01 / Исходные данные")
        example_name = st.selectbox("Пример для проверки", list(EXAMPLES))
        if st.button("Загрузить пример", use_container_width=True):
            st.session_state["json_input"] = EXAMPLES[example_name].read_text(encoding="utf-8")
        if "json_input" not in st.session_state:
            st.session_state["json_input"] = EXAMPLES[example_name].read_text(encoding="utf-8")
        uploaded = st.file_uploader("Или загрузите JSON-файл", type=["json"])
        if uploaded is not None:
            if uploaded.size > 128 * 1024:
                st.error("JSON-файл должен быть не больше 128 КБ.")
            else:
                # Streamlit reruns often; only load newly changed content once.
                raw_data = uploaded.getvalue()
                identity = (uploaded.name, len(raw_data), hash(raw_data))
                if st.session_state.get("last_upload") != identity:
                    try:
                        st.session_state["json_input"] = raw_data.decode("utf-8-sig")
                        st.session_state["last_upload"] = identity
                    except UnicodeDecodeError:
                        st.error("Файл должен иметь кодировку UTF-8.")
        st.text_area("JSON товара", key="json_input", height=365, help="Вставьте объект с name, manufacturer, specs, description. Дополнительно: expected_specs, source_url.")
        product = None
        try:
            product = parse_product(st.session_state["json_input"])
            if not product.name:
                st.warning("Не заполнено название товара — генерация недоступна.")
            fields = missing_fields(product)
            if fields:
                with st.expander(f"Требует уточнения: {len(fields)} полей"):
                    for field in fields:
                        st.write("• " + field)
            else:
                st.success("Обязательные исходные данные заполнены.")
            with st.expander("Характеристики из JSON"):
                st.json(characteristic_list(product), expanded=False)
        except InputError as exc:
            st.error(str(exc))
        enabled = bool(product and product.name and (product.specs or product.description))
        if not api_key():
            st.info("Для генерации нужен GROQ_API_KEY. Данные-примеры и их проверку можно открыть без ключа.")
        if st.button("Показать редакторский пример (без API)", use_container_width=True):
            sample_path = EXAMPLES[example_name]
            example_path = sample_path.with_name(sample_path.stem + "_editorial_example.json")
            example_product = parse_product(sample_path.read_text(encoding="utf-8"))
            example_text = GeneratedText.model_validate_json(example_path.read_text(encoding="utf-8"))
            # An illustrative manually written result, clearly marked as such.
            from instrumentcard.quality import validate_draft
            st.session_state["card"] = CardOutput(
                product_name=example_product.name,
                manufacturer=example_product.manufacturer,
                seo_title=example_text.seo_title,
                seo_description=example_text.seo_description,
                characteristics=characteristic_list(example_product),
                search_queries=example_text.search_queries,
                commercial_block=example_text.commercial_block,
                missing_fields=missing_fields(example_product),
                review_warnings=["РЕДАКТОРСКИЙ ПРИМЕР: составлен вручную без вызова API.", *validate_draft(example_product, example_text), "Перед публикацией необходима проверка контент-менеджером."],
                source_url=example_product.source_url,
                automated_checks_passed=False,
            ).model_dump()
        if st.button("Сгенерировать карточку", type="primary", use_container_width=True, disabled=not enabled):
            if not api_key():
                st.error("API-ключ отсутствует. Укажите GROQ_API_KEY и перезапустите приложение.")
            else:
                try:
                    with st.spinner("Генерируем и проверяем факты…"):
                        st.session_state["card"] = generate_card(product, api_key()).model_dump()
                except GenerationError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Непредвиденная ошибка генерации. Проверьте журнал сервиса.")
        st.caption("Примеры составлены по публичным карточкам souz-pribor.ru. Принадлежность товаров к исходному YML-фиду отдельно не подтверждена.")

    with right:
        st.subheader("02 / Результат")
        if "card" not in st.session_state:
            st.info("Загрузите пример или вставьте JSON, затем нажмите «Сгенерировать карточку».")
            st.markdown("**Как проверяется результат**")
            st.write("1. Исходные характеристики копируются дословно, без генерации.")
            st.write("2. Модель возвращает четыре текстовых поля по строгой JSON-схеме.")
            st.write("3. Длина, количество запросов и новые числовые значения проверяются кодом.")
            st.write("4. Замечания остаются видимыми до проверки контент-менеджером.")
            st.stop()
        card = CardOutput.model_validate(st.session_state["card"])
        if product and product.name != card.product_name:
            st.warning("Показан результат для предыдущего товара. Нажмите генерацию для новых данных.")
        m1, m2, m3 = st.columns(3)
        m1.metric("SEO-заголовок", f"{len(card.seo_title)}/70")
        m2.metric("SEO-описание", f"{len(card.seo_description)} симв.")
        m3.metric("Поисковые запросы", len(card.search_queries))
        preview, audit = st.tabs(["Карточка", "Контроль качества"])
        with preview:
            st.markdown("**SEO-заголовок**")
            st.code(card.seo_title, language=None)
            st.markdown("**SEO-описание**")
            st.text_area("Текст для копирования", card.seo_description, height=215)
            st.markdown("**Технические характеристики — из входного JSON**")
            st.table([{"Характеристика": k, "Значение": v} for k, v in card.characteristics.items()])
            st.markdown("**Поисковые запросы**")
            st.code("\n".join(card.search_queries), language=None)
            st.markdown("**Коммерческий блок**")
            st.code(card.commercial_block, language=None)
        with audit:
            st.markdown("**Недостающие сведения**")
            if card.missing_fields:
                for item in card.missing_fields:
                    st.warning(item)
            else:
                st.success("Не обнаружены отсутствующие обязательные поля и ожидаемые характеристики.")
            st.markdown("**Ограничения и замечания**")
            for item in card.review_warnings:
                st.info(item)
            if card.source_url:
                st.markdown(f"[Источник товара]({card.source_url})")
            st.caption("Проверка чисел не обнаруживает все возможные смысловые ошибки. Требуется экспертная проверка.")
        col1, col2 = st.columns(2)
        col1.download_button("Скачать JSON", json.dumps(card.model_dump(), ensure_ascii=False, indent=2), "product_card.json", "application/json", use_container_width=True)
        col2.download_button("Скачать Markdown", to_markdown(card), "product_card.md", "text/markdown", use_container_width=True)


if __name__ == "__main__":
    main()
