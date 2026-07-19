import unittest

from mini_kanon3.capabilities.embed.metrics import evaluate_rankings


class MetricsTest(unittest.TestCase):
    def test_perfect_ranking(self):
        metrics = evaluate_rankings({"q1": ["p1", "p2"]}, {"q1": {"p1"}})
        self.assertEqual(metrics["ndcg_at_10"], 1.0)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_multiple_positives_uses_recall_fraction(self):
        metrics = evaluate_rankings({"q1": ["p1", "x", "p2"]}, {"q1": {"p1", "p2"}})
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
