import json
import tempfile
import unittest
from pathlib import Path

from mini_kanon3.capabilities.embed.mining import mine_bm25_hard_negatives


class BM25MiningTest(unittest.TestCase):
    def test_known_positives_and_same_document_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "train"
            split.mkdir()
            self._jsonl(split / "queries.jsonl", [{"query_id": "q1", "text": "bail court"}])
            self._jsonl(split / "corpus.jsonl", [
                {"passage_id": "p1", "document_id": "d1", "text": "bail court positive"},
                {"passage_id": "p2", "document_id": "d1", "text": "bail court same document"},
                {"passage_id": "p3", "document_id": "d2", "text": "bail court hard negative"},
            ])
            (split / "qrels.tsv").write_text(
                "query_id\tpassage_id\trelevance\nq1\tp1\t1\n", encoding="utf-8")
            output, report = root / "negatives.jsonl", root / "report.json"
            mine_bm25_hard_negatives({"input_split": str(split), "output_path": str(output),
                                      "report_path": str(report), "num_negatives": 1,
                                      "candidate_depth": 3, "exclude_same_document": True})
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([item["passage_id"] for item in row["negative_passages"]], ["p3"])

    @staticmethod
    def _jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
