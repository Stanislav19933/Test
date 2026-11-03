"""Regression tests: the exact SEO contract must be enforced."""

import unittest

from instrumentcard.models import GeneratedText, ProductInput
from instrumentcard.quality import validate_draft
from instrumentcard.service import IncompleteCardError, generate_card


class TestStrictSeoContract(unittest.TestCase):
    def setUp(self):
        self.product = ProductInput(
            name="Датчик модели 20",
            manufacturer="Завод",
            specs={"Диапазон": "0–20 МПа", "Питание": "9 В", "Корпус": "сталь"},
            description="Прибор предназначен для измерения давления. " * 8,
        )
        self.short = GeneratedText(
            seo_title="Датчик модели 20",
            seo_description="Датчик измеряет давление 0–20 МПа.",
            search_queries=["датчик 20", "датчик давления", "датчик 0–20 МПа"],
            commercial_block="Для измерения давления.",
        )
        self.good = self.short.model_copy(update={
            "seo_description": ("Датчик модели 20 измеряет давление 0–20 МПа. " * 25)[:950]
        })

    def test_short_description_cannot_pass(self):
        result = generate_card(self.product, "fake", generator=lambda p, key, corrections: self.short)
        self.assertFalse(result.automated_checks_passed)
        self.assertTrue(any("SEO-описание" in issue for issue in result.review_warnings))

    def test_repair_selects_full_description(self):
        calls = []
        def generator(product, key, corrections):
            calls.append(corrections)
            return self.good if corrections else self.short
        result = generate_card(self.product, "fake", generator=generator)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("900–1050" in instruction for instruction in calls[1]))
        self.assertTrue(800 <= len(result.seo_description) <= 1200)
        self.assertTrue(result.automated_checks_passed)

    def test_partly_improved_description_is_still_incomplete(self):
        partial = self.good.model_copy(update={"seo_description": self.good.seo_description[:640]})
        result = generate_card(self.product, "fake",
            generator=lambda p, key, corrections: partial if corrections else self.short)
        self.assertEqual(len(result.seo_description), 640)
        self.assertFalse(result.automated_checks_passed)

    def test_new_unsupported_number_is_rejected_even_if_longer(self):
        suspect = self.good.model_copy(update={"commercial_block": "Точность 999 МПа."})
        result = generate_card(self.product, "fake",
            generator=lambda p, key, corrections: suspect if corrections else self.short)
        self.assertEqual(result.seo_description, self.short.seo_description)

    def test_sparse_input_still_reports_length_error(self):
        issues = validate_draft(ProductInput(name="Датчик", description="Мало."), self.short)
        self.assertTrue(any("800–1200" in issue for issue in issues))

    def test_incomplete_error_retains_draft(self):
        card = generate_card(self.product, "fake", generator=lambda p, key, corrections: self.short)
        self.assertEqual(IncompleteCardError(card).card.seo_title, self.short.seo_title)


if __name__ == "__main__":
    unittest.main()
