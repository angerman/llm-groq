import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import llm_groq


class GroqTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        os.environ["LLM_USER_PATH"] = self.tmp.name
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("LLM_GROQ_KEY", None)

    def test_key_fallback_order(self):
        os.environ["LLM_GROQ_KEY"] = "legacy"
        self.assertEqual(llm_groq.get_groq_api_key(), "legacy")
        os.environ["GROQ_API_KEY"] = "primary"
        self.assertEqual(llm_groq.get_groq_api_key(), "primary")
        (Path(self.tmp.name) / "keys.json").write_text(json.dumps({"groq": "stored"}))
        self.assertEqual(llm_groq.get_groq_api_key(), "stored")

    def test_vision_detection(self):
        vision = llm_groq.model_supports_vision
        self.assertTrue(vision({"id": "scout", "input_modalities": ["text", "image"]}))
        self.assertTrue(vision({"id": "llama-3.2-11b-vision-preview"}))
        self.assertTrue(vision({"id": "x-vision", "input_modalities": None}))
        self.assertFalse(vision({"id": "foo-vision", "input_modalities": ["text"]}))
        self.assertFalse(vision({"id": "foo-vision", "input_modalities": []}))


if __name__ == "__main__":
    unittest.main()
