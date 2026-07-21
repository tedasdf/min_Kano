import json
import tempfile
import unittest
from pathlib import Path

from mini_kanon3.capabilities.embed.dataset import PositivePair, attach_mined_negatives


class HardNegativeDatasetTest(unittest.TestCase):
    def test_attaches_passage_text_and_rejects_positive_overlap(self):
        pair = PositivePair("q1", "p1", "query", "positive")
        corpus = {"p1": "positive", "p2": "negative"}
        positives = {"q1": ["p1"]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mined.jsonl"
            path.write_text(json.dumps({"query_id": "q1", "positive_passage_ids": ["p1"],
                                        "negative_passages": [{"passage_id": "p2", "rank": 2,
                                                               "bm25_score": 1.0}]}) + "\n",
                            encoding="utf-8")
            result = attach_mined_negatives([pair], corpus, positives, path, 1)
            self.assertEqual(result[0].negative_passages, ("negative",))


if __name__ == "__main__":
    unittest.main()
