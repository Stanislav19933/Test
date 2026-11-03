import json
import unittest
from unittest.mock import patch

from instrumentcard.zai_client import GenerationError, _api_call, generate_text
from instrumentcard.models import GeneratedText, ProductInput
from instrumentcard.quality import characteristic_list, missing_fields, numeric_claim_warnings, validate_draft
from instrumentcard.service import InputError, generate_card, parse_product


class TestInput(unittest.TestCase):
    def test_invalid_json(self):
        with self.assertRaises(InputError):
            parse_product('{bad')

    def test_non_object_json(self):
        with self.assertRaisesRegex(InputError, 'объектом'):
            parse_product('[]')

    def test_bad_specs_type(self):
        with self.assertRaises(InputError):
            parse_product('{"name":"X", "specs":[1]}')

    def test_missing_fields_visible(self):
        product = parse_product('{"name":"Прибор","specs":{"Класс точности":""},"expected_specs":["Класс точности","Давление"]}')
        self.assertEqual(characteristic_list(product), {"Класс точности": "данных нет", "Давление": "данных нет"})
        self.assertIn('Производитель', missing_fields(product))
        self.assertIn('Характеристика: Давление', missing_fields(product))

    def test_specs_values_not_modified(self):
        product = ProductInput(name='X', specs={'Класс':'1,5', 'Резьба':'G1/2'})
        self.assertEqual(characteristic_list(product)['Резьба'], 'G1/2')


class TestQuality(unittest.TestCase):
    def setUp(self):
        self.product = ProductInput(name="Прибор Т-10", manufacturer="Фирма", specs={"Диапазон": "0–10 МПа"}, description="Датчик давления измеряет давление в трубопроводе. " * 5)
        self.draft = GeneratedText(seo_title="Прибор Т-10", seo_description="Измерение 0–10 МПа. " * 10, search_queries=['прибор Т-10', 'датчик 10 МПа', 'купить Т-10'], commercial_block="Для промышленных измерений.")

    def test_new_numbers_are_flagged(self):
        draft = self.draft.model_copy(update={"seo_description": "Новая точность 0,02 МПа. " * 40})
        self.assertTrue(any('0.02' in msg for msg in numeric_claim_warnings(self.product, draft)))

    def test_valid_original_numbers_allowed(self):
        self.assertEqual(numeric_claim_warnings(self.product, self.draft), [])

    def test_length_validation(self):
        errors = validate_draft(self.product, self.draft)
        self.assertTrue(any('SEO-описание' in err for err in errors))

    def test_no_unsupported_claims_in_specs(self):
        self.assertNotIn('Точность', characteristic_list(self.product))


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.p = ProductInput(name="Термометр 20", manufacturer="М", specs={"Диапазон": "0–20 °C", "Питание": "9 В", "Корпус": "пластик"}, description="Прибор измеряет температуру. " * 10)
        self.good = GeneratedText(seo_title='Термометр 20', seo_description='Термометр 20 измеряет температуру 0–20 °C. ' * 21, search_queries=['термометр 20', 'термометр купить', 'измерение температуры'], commercial_block='Для измерения температуры.')
        # Deliberately trim to the required window without introducing new numerical data.
        self.good = self.good.model_copy(update={'seo_description': (self.good.seo_description * 2)[:1000]})

    def test_api_key_not_present(self):
        with self.assertRaisesRegex(GenerationError, 'API-ключ'):
            generate_text(self.p, '')

    def test_retries_and_warns_on_hallucination(self):
        bad = self.good.model_copy(update={'seo_title':'Супер ' * 20, 'commercial_block':'Работает при 999 °C.'})
        calls = []
        def generate(product, key, corrections):
            calls.append(corrections)
            return bad if len(calls) == 1 else self.good
        card = generate_card(self.p, 'fake-key', generator=generate)
        self.assertEqual(len(calls), 2)
        self.assertEqual(card.seo_title, self.good.seo_title)
        self.assertEqual(card.characteristics['Диапазон'], '0–20 °C')

    def test_explicit_missing_data(self):
        self.p.expected_specs = ['Диапазон', 'Класс точности']
        card = generate_card(self.p, 'fake', generator=lambda p, k, e: self.good)
        self.assertEqual(card.characteristics['Класс точности'], 'данных нет')
        self.assertFalse(card.automated_checks_passed)

    def test_missing_essential_input_short_circuits(self):
        with self.assertRaises(GenerationError):
            generate_card(ProductInput(), 'fake', generator=lambda p, k, e: self.good)


if __name__ == '__main__':
    unittest.main()

class TestSampleExamples(unittest.TestCase):
    def test_editorial_examples_match_contract(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / "samples"
        for slug in ("dt83", "tm510"):
            with self.subTest(slug=slug):
                product = parse_product((root / f"{slug}.json").read_text(encoding="utf-8"))
                draft = GeneratedText.model_validate_json((root / f"{slug}_editorial_example.json").read_text(encoding="utf-8"))
                self.assertEqual(validate_draft(product, draft), [])

    def test_zai_request_uses_json_object_mode(self):
        product = ProductInput(name="Датчик 10", specs={"Диапазон": "0–10 МПа"})
        response = {"choices": [{"message": {"content": json.dumps({
            "seo_title": "Датчик 10", "seo_description": "Диапазон 0–10 МПа.",
            "search_queries": ["Датчик 10", "датчик давления", "датчик 0–10 МПа"],
            "commercial_block": "Измеряет давление."
        }, ensure_ascii=False)}}]}
        with patch("instrumentcard.zai_client._api_call", return_value=response) as mocked:
            result = generate_text(product, "fake-key")
        self.assertEqual(result.seo_title, "Датчик 10")
        payload = mocked.call_args.args[0]
        self.assertEqual(payload["model"], "glm-4.7-flash")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})

class TestResponseValidation(unittest.TestCase):
    def test_truncated_model_response_is_rejected(self):
        product = ProductInput(name="Датчик", specs={"Диапазон": "0–10 МПа"})
        response = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
        with patch("instrumentcard.zai_client._api_call", return_value=response):
            with self.assertRaisesRegex(GenerationError, "схеме"):
                generate_text(product, "fake-key")


class TestApiErrors(unittest.TestCase):
    def test_auth_error_is_user_friendly(self):
        from urllib.error import HTTPError
        err = HTTPError('https://example.org', 401, 'Unauthorized', {}, None)
        with patch('instrumentcard.zai_client.urllib.request.urlopen', side_effect=err):
            with self.assertRaisesRegex(GenerationError, 'авторизации'):
                _api_call({'model': 'demo'}, 'fake')

    def test_rate_limit_is_user_friendly(self):
        from urllib.error import HTTPError
        err = HTTPError('https://example.org', 429, 'Too many requests', {}, None)
        with patch('instrumentcard.zai_client.urllib.request.urlopen', side_effect=err):
            with self.assertRaisesRegex(GenerationError, 'лимит'):
                _api_call({'model': 'demo'}, 'fake')

    def test_rejects_unsafe_source_url(self):
        with self.assertRaises(InputError):
            parse_product('{"name":"X","source_url":"javascript:alert(1)"}')



class TestTemporaryZaiRecovery(unittest.TestCase):
    def setUp(self):
        from instrumentcard.zai_client import TemporaryAPIError
        self.temp_error = TemporaryAPIError
        self.product = ProductInput(
            name="Манометр", specs={"Диапазон": "0–10 МПа"},
            description="Прибор предназначен для измерения давления. " * 5,
        )
        self.answer = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "seo_title": "Манометр 0–10 МПа",
            "seo_description": "Измерение давления 0–10 МПа.",
            "search_queries": ["манометр", "манометр давления", "манометр 0–10 МПа"],
            "commercial_block": "Измеряет давление.",
        }, ensure_ascii=False)}}]}

    @staticmethod
    def http_error(status, code, retry_after=None):
        from io import BytesIO
        from urllib.error import HTTPError
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        body = BytesIO(json.dumps({"error": {"code": str(code), "message": ""}}).encode())
        return HTTPError("https://api.z.ai/api/paas/v4/chat/completions", status, "error", headers, body)

    def test_1302_has_retry_after(self):
        with patch("instrumentcard.zai_client.urllib.request.urlopen", side_effect=self.http_error(429, 1302, 6)):
            with self.assertRaises(self.temp_error) as caught:
                _api_call({"model": "glm-4.7-flash"}, "fake")
        self.assertEqual(caught.exception.retry_after, 6)

    def test_1305_is_temporary(self):
        with patch("instrumentcard.zai_client.urllib.request.urlopen", side_effect=self.http_error(429, 1305)):
            with self.assertRaises(self.temp_error):
                _api_call({"model": "glm-4.7-flash"}, "fake")

    def test_balance_and_quota_are_not_retried(self):
        for code, part in ((1113, "тарифу"), (1310, "квота")):
            with self.subTest(code=code):
                with patch("instrumentcard.zai_client.urllib.request.urlopen", side_effect=self.http_error(429, code)):
                    with self.assertRaisesRegex(GenerationError, part):
                        _api_call({"model": "glm-4.7-flash"}, "fake")

    def test_successful_retry(self):
        models = []
        def call(payload, key):
            models.append(payload["model"])
            if len(models) == 1:
                raise self.temp_error("overloaded")
            return self.answer
        with patch("instrumentcard.zai_client._api_call", side_effect=call), patch("instrumentcard.zai_client.time.sleep") as sleep:
            text = generate_text(self.product, "fake")
        self.assertEqual(text.seo_title, "Манометр 0–10 МПа")
        self.assertEqual(models, ["glm-4.7-flash", "glm-4.7-flash"])
        sleep.assert_called_once()

    def test_fallback_to_other_free_model(self):
        models = []
        def call(payload, key):
            models.append(payload["model"])
            if payload["model"] == "glm-4.7-flash":
                raise self.temp_error("overloaded")
            return self.answer
        with patch("instrumentcard.zai_client._api_call", side_effect=call), patch("instrumentcard.zai_client.time.sleep"):
            text = generate_text(self.product, "fake")
        self.assertEqual(text.seo_title, "Манометр 0–10 МПа")
        self.assertEqual(models, ["glm-4.7-flash", "glm-4.7-flash", "glm-4.5-flash"])

    def test_both_unavailable_fail_honestly(self):
        with patch("instrumentcard.zai_client._api_call", side_effect=self.temp_error("overloaded")), patch("instrumentcard.zai_client.time.sleep"):
            with self.assertRaisesRegex(GenerationError, "Обе бесплатные модели"):
                generate_text(self.product, "fake")

    def test_keep_first_draft_if_correction_fails(self):
        draft = GeneratedText(
            seo_title="Манометр", seo_description="Измерение 0–10 МПа.",
            search_queries=["манометр", "манометр купить", "манометр 0–10 МПа"],
            commercial_block="Для измерения давления.",
        )
        attempts = []
        def generator(product, key, corrections):
            attempts.append(corrections)
            if corrections:
                raise GenerationError("limit")
            return draft
        card = generate_card(self.product, "fake", generator=generator)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(card.seo_title, "Манометр")
        self.assertFalse(card.automated_checks_passed)
        self.assertTrue(any("первый черновик" in w for w in card.review_warnings))


class TestRobustZaiOutput(unittest.TestCase):
    def setUp(self):
        self.answer = {
            "seo_title": "Манометр",
            "seo_description": "Измерение давления 0–10 МПа.",
            "search_queries": ["манометр", "измерение давления", "манометр 0–10 МПа"],
            "commercial_block": "Для измерения давления.",
        }

    def test_complete_json_even_when_length_finish_is_accepted(self):
        from instrumentcard.zai_client import _parse_answer
        data = {"model": "glm-4.7-flash", "choices": [{"finish_reason": "length",
            "message": {"content": json.dumps(self.answer, ensure_ascii=False)}}]}
        self.assertEqual(_parse_answer(data).seo_title, "Манометр")

    def test_markdown_fenced_json_is_parsed(self):
        from instrumentcard.zai_client import _parse_answer
        data = {"choices": [{"finish_reason": "stop", "message": {"content":
            "```json\n" + json.dumps(self.answer, ensure_ascii=False) + "\n```"}}]}
        self.assertEqual(_parse_answer(data).seo_title, "Манометр")

    def test_extra_model_metadata_is_ignored(self):
        from instrumentcard.zai_client import _parse_answer
        data = {"choices": [{"message": {"content": json.dumps({**self.answer, "characteristics": {"Точность": "999"}}, ensure_ascii=False)}}]}
        self.assertEqual(_parse_answer(data).seo_title, "Манометр")

    def test_missing_output_fields_reported(self):
        from instrumentcard.zai_client import _parse_answer, ModelOutputError
        data = {"choices": [{"message": {"content": '{"seo_title":"Манометр"}'}}]}
        with self.assertRaisesRegex(ModelOutputError, "не хватает полей"):
            _parse_answer(data)

    def test_invalid_json_is_format_error(self):
        from instrumentcard.zai_client import _parse_answer, ModelOutputError
        data = {"choices": [{"message": {"content": "Некорректно"}}]}
        with self.assertRaisesRegex(ModelOutputError, "невалидный JSON"):
            _parse_answer(data)

    def test_format_error_retries_and_then_uses_other_model(self):
        from instrumentcard.zai_client import ModelOutputError
        product = ProductInput(name="Манометр", specs={"Диапазон": "0–10 МПа"})
        calls = []
        def fake(payload, key):
            calls.append(payload["model"])
            if payload["model"] == "glm-4.7-flash":
                raise ModelOutputError("невалидный JSON", model=payload["model"])
            return {"choices": [{"message": {"content": json.dumps(self.answer, ensure_ascii=False)}}]}
        with patch("instrumentcard.zai_client._api_call", side_effect=fake), patch("instrumentcard.zai_client.time.sleep"):
            result = generate_text(product, "fake-key")
        self.assertEqual(result.seo_title, "Манометр")
        self.assertEqual(calls, ["glm-4.7-flash", "glm-4.7-flash", "glm-4.5-flash"])
