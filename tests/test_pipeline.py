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
