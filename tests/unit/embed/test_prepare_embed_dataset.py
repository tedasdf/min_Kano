import json
import tempfile
import unittest
from pathlib import Path

from mini_kanon3.data.prepare_embed import prepare


class PreparationTest(unittest.TestCase):
    def test_deduplication_multi_positive_and_document_isolation(self):
        rows = [
            {"question": "Same question?", "answer": "A", "source": {"version_id": "d1", "text": "Passage one."}},
            {"question": "Same question?", "answer": "A", "source": {"version_id": "d1", "text": "Passage one."}},
            {"question": "Same question?", "answer": "B", "source": {"version_id": "d2", "text": "Passage two."}},
            {"question": "Other?", "answer": "C", "source": {"version_id": "d3", "text": "Passage three."}},
            {"question": " ", "answer": "D", "source": {"version_id": "d4", "text": "Dropped."}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            report = prepare(source, root / "data", "test", (1 / 3, 1 / 3, 1 / 3), [2])
            self.assertEqual(report["rows"]["valid"], 4)
            self.assertEqual(report["unique"]["queries"], 2)
            self.assertEqual(report["unique"]["passages"], 3)
            self.assertEqual(report["unique"]["qrels"], 3)
            self.assertEqual(report["multi_positive_queries"], 1)
            self.assertEqual(report["duplicates"]["pair_extra_rows"], 1)
            split_docs = []
            for split in ("train", "validation", "test"):
                docs = {
                    json.loads(line)["document_id"]
                    for line in (root / "data" / "processed" / "embed" / split / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
                }
                split_docs.append(docs)
            self.assertFalse(split_docs[0] & split_docs[1])
            self.assertFalse(split_docs[0] & split_docs[2])
            self.assertFalse(split_docs[1] & split_docs[2])


if __name__ == "__main__":
    unittest.main()
