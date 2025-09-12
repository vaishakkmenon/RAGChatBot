import os
import json
import unittest
from typing import List, Dict
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class TestExtractiveP2(unittest.TestCase):
    def setUp(self) -> None:
        from app import main as main_mod
        self.main_mod = main_mod
        self.client = TestClient(main_mod.app)

    def _mock_chunk(self) -> Dict:
        return {
            "id": "/workspace/data/docs/squad-v2/dev_md/Normans-dd5c8526091c.md:0",
            "text": (
                "The Normans were descended from Norse raiders and pirates from Denmark, "
                "Iceland, and Norway."
            ),
            "distance": 0.12,
        }

    def test_extractive_returns_evidence_span(self):
        chunk = self._mock_chunk()
        gold = chunk["text"]  # generator copies verbatim from retrieved text

        with patch.object(self.main_mod, "search", return_value=[chunk]), \
            patch.object(
                self.main_mod,
                "generate_extractive",
                new=AsyncMock(return_value=(gold, {"meta": "test"})),
            ):

            payload = {"question": "Who were the Normans descended from?", "top_k": 5}
            r = self.client.post(
                "/chat?extractive=1",
                json=payload,
                headers={"X-API-Key": os.getenv("API_KEY", "my-dev-key-1")},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()

            # Basic shape
            self.assertEqual(body.get("answer"), gold)
            self.assertIsInstance(body.get("sources"), list)
            self.assertEqual(len(body["sources"]), 1)

            ev = body.get("evidence_span")
            self.assertIsNotNone(ev, "evidence_span must be present for extractive answers")

            # Correct doc and indices
            src = body["sources"][0]
            self.assertEqual(ev["doc_id"], src["id"])
            start, end = ev["start"], ev["end"]
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)
            self.assertLessEqual(end, len(src["text"]))

            # Slice equality
            sliced = src["text"][start:end]
            self.assertEqual(sliced, ev["text"])
            self.assertEqual(ev["text"], body["answer"])
            

    def test_extractive_abstains_has_no_evidence_span(self):
        # When the model abstains (NOT_IN_CONTEXT), we must return "I don't know." and no evidence span.
        from app import main as main_mod
        client = TestClient(main_mod.app)

        chunk = self._mock_chunk()
        with patch.object(main_mod, "search", return_value=[chunk]), \
            patch.object(
                main_mod,
                "generate_extractive",
                new=AsyncMock(return_value=("NOT_IN_CONTEXT", {"meta": "test"})),
            ):
            payload = {"question": "Who were the Normans descended from?", "top_k": 5}
            r = client.post(
                "/chat?extractive=1",
                json=payload,
                headers={"X-API-Key": os.getenv("API_KEY", "my-dev-key-1")},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body.get("answer"), "I don't know.")
            self.assertIsNone(body.get("evidence_span"))

    def test_extractive_case_insensitive_match(self):
        # Case-insensitive substring should still yield a valid contiguous span.
        from app import main as main_mod
        client = TestClient(main_mod.app)

        chunk = self._mock_chunk()
        gold_ci = chunk["text"].upper()  # simulate model copying with different casing

        with patch.object(main_mod, "search", return_value=[chunk]), \
             patch.object(
                 main_mod,
                 "generate_extractive",
                 new=AsyncMock(return_value=(gold_ci, {"meta": "test"})),
             ):
            payload = {"question": "Who were the Normans descended from?", "top_k": 5}
            r = client.post(
                "/chat?extractive=1",
                json=payload,
                headers={"X-API-Key": os.getenv("API_KEY", "my-dev-key-1")},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body.get("answer"), gold_ci)
            ev = body.get("evidence_span")
            self.assertIsNotNone(ev)
            # Evidence text should equal source slice (original casing), but CI-equal to the answer
            self.assertEqual(
                ev["text"].casefold(),
                body["answer"].casefold(),
            )

    def test_extractive_subsequence_fallback_span(self):
        # Ordered token subsequence should snap to a single contiguous span from first→last token.
        from app import main as main_mod
        client = TestClient(main_mod.app)

        chunk = self._mock_chunk()
        # Model answer drops commas and some stopwords; tokens remain in order:
        # pick a middle subsequence so the snapped span is inside the sentence.
        answer_subseq = "Norse raiders pirates Denmark Iceland Norway"

        with patch.object(main_mod, "search", return_value=[chunk]), \
             patch.object(
                 main_mod,
                 "generate_extractive",
                 new=AsyncMock(return_value=(answer_subseq, {"meta": "test"})),
             ):
            payload = {"question": "Who were the Normans descended from?", "top_k": 5}
            r = client.post(
                "/chat?extractive=1",
                json=payload,
                headers={"X-API-Key": os.getenv("API_KEY", "my-dev-key-1")},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()
            ev = body.get("evidence_span")
            self.assertIsNotNone(ev)
            src_slice = ev["text"].casefold()

            # All tokens must appear in order inside the contiguous slice
            tokens = answer_subseq.casefold().split()
            pos = 0
            for tok in tokens:
                found = src_slice.find(tok, pos)
                self.assertNotEqual(found, -1, f"token '{tok}' not found in order")
                pos = found + len(tok)

            # Indices are sane
            src = body["sources"][0]
            start, end = ev["start"], ev["end"]
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(src["text"]))
            self.assertGreater(end, start)