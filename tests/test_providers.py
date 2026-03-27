import unittest

from config import AVAILABLE_GEMINI_MODELS, DEFAULT_GEMINI_MODEL
from providers import GeminiProvider, get_provider


class ProviderTests(unittest.TestCase):
    def test_only_new_gemini_model_is_available(self) -> None:
        provider = GeminiProvider()

        self.assertEqual(provider.current_model(), DEFAULT_GEMINI_MODEL)
        self.assertEqual(provider.available_models(), [DEFAULT_GEMINI_MODEL])
        self.assertEqual(AVAILABLE_GEMINI_MODELS, [DEFAULT_GEMINI_MODEL])

    def test_rejects_old_models(self) -> None:
        with self.assertRaises(ValueError):
            get_provider(model="gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
