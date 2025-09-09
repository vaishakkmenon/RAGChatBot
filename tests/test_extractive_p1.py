import os
import importlib
import types
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Load the app module
app_module = importlib.import_module("app.main")
app = app_module.app

# Pull the configured API key from settings, with fallbacks
API_KEY = getattr(app_module, "settings", None)
API_KEY = getattr(API_KEY, "api_key", None) or os.getenv("API_KEY") or "my-dev-key-1"

HEADERS = {"X-API-Key": API_KEY}

def _fake_search_with_gold(q, k=5, max_distance=0.8):
    text = (
        '# Normans\nThe Normans ... '
        'They were descended from Norse ("Norman" comes from "Norseman") '
        'raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, ...'
    )
    # Match the app's expectations: dicts with distance
    return [{
        "id": "/doc/Normans.md:0",
        "text": text,
        "distance": 0.10,   # within threshold
    }]

class TestExtractiveA1(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.search", side_effect=_fake_search_with_gold, autospec=True)
    @patch("app.main.generate_extractive", autospec=True)
    def test_extractive_returns_verbatim_span(self, mock_gen, _mock_search):
        mock_gen.return_value = (
            "Norse raiders and pirates from Denmark, Iceland and Norway",
            {"mock": True},
        )
        r = self.client.post(
            "/chat?extractive=1",
            headers=HEADERS,
            json={"question": "Who were the Normans descended from?", "top_k": 5},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["answer"].startswith("Norse raiders"))
        self.assertTrue(data.get("raw_generation", "").startswith("Norse raiders"))
        self.assertTrue(data["sources"] and data["sources"][0]["id"].endswith("Normans.md:0"))

    @patch("app.main.search", side_effect=_fake_search_with_gold, autospec=True)
    @patch("app.main.generate_extractive", autospec=True)
    def test_extractive_not_in_context_maps_to_idk(self, mock_gen, _mock_search):
        mock_gen.return_value = ("NOT_IN_CONTEXT", {"mock": True})
        r = self.client.post(
            "/chat?extractive=1",
            headers=HEADERS,
            json={"question": "Who were the Normans descended from?", "top_k": 5},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["answer"], "I don't know.")
        self.assertEqual(data.get("raw_generation"), "NOT_IN_CONTEXT")

    @patch("app.main._CLIENT", autospec=True)
    @patch("app.main.search", side_effect=_fake_search_with_gold, autospec=True)
    def test_generative_branch_still_works(self, _mock_search, mock_client):
        mock_client.chat.return_value = {"message": {"content": "hello from generative"}}
        r = self.client.post(
            "/chat?grounded_only=false",
            headers=HEADERS,
            json={"question": "ping", "top_k": 1},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["answer"], "hello from generative")

    @patch("app.main.search", side_effect=_fake_search_with_gold, autospec=True)
    @patch("app.main.generate_extractive", side_effect=RuntimeError("boom"), autospec=True)
    def test_no_unboundlocalerror_on_extractive_error(self, _mock_gen, _mock_search):
        r = self.client.post(
            "/chat?extractive=1",
            headers=HEADERS,
            json={"question": "ping", "top_k": 1},
        )
        # Expect a generic 500 from our handler, NOT an UnboundLocalError leak
        self.assertEqual(r.status_code, 500, r.text)
        self.assertIn("detail", r.json())

if __name__ == "__main__":
    unittest.main()