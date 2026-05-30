import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llm_groq


class ModelListingTests(unittest.TestCase):
    def test_get_model_details_uses_groq_api_key_env_var(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(llm_groq.llm, "user_dir", return_value=Path(tmpdir)),
                patch.object(llm_groq.llm, "get_key", return_value="test-key") as get_key,
                patch.object(
                    llm_groq, "refresh_models", return_value=[{"id": "llama-3.1-8b-instant"}]
                ),
            ):
                details = llm_groq.get_model_details()

        self.assertEqual(details, [{"id": "llama-3.1-8b-instant"}])
        get_key.assert_called_once_with("", "groq", "GROQ_API_KEY")


if __name__ == "__main__":
    unittest.main()
