"""Smoke checks for the actual Streamlit entrypoint (without API/network)."""

import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class TestStreamlitApp(unittest.TestCase):
    def test_initial_page_loads_without_exceptions(self):
        app = AppTest.from_file(APP_FILE, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("InstrumentCard AI" in item.value for item in app.title))

    def test_editorial_example_is_explicitly_labelled(self):
        app = AppTest.from_file(APP_FILE, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        button = next(item for item in app.button if item.label == "Показать редакторский пример (без API)")
        button.click().run()
        self.assertEqual(len(app.exception), 0)
        card = app.session_state["card"]
        self.assertFalse(card["automated_checks_passed"])
        self.assertTrue(any("РЕДАКТОРСКИЙ ПРИМЕР" in item for item in card["review_warnings"]))
        self.assertTrue(800 <= len(card["seo_description"]) <= 1200)

    def test_zai_outage_does_not_fake_result(self):
        from unittest.mock import patch
        from instrumentcard.zai_client import GenerationError

        app = AppTest.from_file(APP_FILE, default_timeout=20)
        app.secrets["ZAI_API_KEY"] = "fake-key-for-test"
        with patch("instrumentcard.zai_client._api_call", side_effect=GenerationError("Временный лимит 429")):
            app.run()
            self.assertEqual(len(app.exception), 0)
            button = next(item for item in app.button if item.label == "Сгенерировать карточку")
            button.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("card", app.session_state)
        self.assertTrue(any("Временный лимит 429" in item.value for item in app.error))

    def test_short_ai_response_displays_real_incomplete_draft(self):
        import json
        from unittest.mock import patch

        draft = {
            "seo_title": "Термогигрометр CEM DT-83",
            "seo_description": "Прибор предназначен для измерения температуры и влажности.",
            "search_queries": ["DT-83 CEM", "термогигрометр", "измерение температуры"],
            "commercial_block": "Для измерения температуры и влажности воздуха.",
        }
        response = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(draft, ensure_ascii=False),
        }}]}
        app = AppTest.from_file(APP_FILE, default_timeout=20)
        app.secrets["ZAI_API_KEY"] = "fake-key-for-test"
        with patch("instrumentcard.zai_client._api_call", return_value=response):
            app.run()
            self.assertEqual(len(app.exception), 0)
            button = next(item for item in app.button if item.label == "Сгенерировать карточку")
            button.click().run()
        self.assertEqual(len(app.exception), 0)
        card = app.session_state["card"]
        self.assertFalse(any("РЕДАКТОРСКИЙ ПРИМЕР" in item for item in card["review_warnings"]))
        self.assertTrue(len(card["seo_description"]) < 800)
        self.assertFalse(card["automated_checks_passed"])
        self.assertTrue(any("не прошла проверку" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
