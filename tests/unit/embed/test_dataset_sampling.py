import unittest

from mini_kanon3.capabilities.embed.dataset import arrange_no_duplicate_batches, sample_one_positive_per_query


class PositiveSamplingTest(unittest.TestCase):
    def test_one_pair_per_query_and_deterministic(self):
        queries = {"q1": "one", "q2": "two"}
        corpus = {"p1": "a", "p2": "b", "p3": "c"}
        positives = {"q1": ["p1", "p2"], "q2": ["p3"]}
        first = sample_one_positive_per_query(queries, corpus, positives, 42, 0)
        second = sample_one_positive_per_query(queries, corpus, positives, 42, 0)
        self.assertEqual(first, second)
        self.assertEqual({pair.query_id for pair in first}, {"q1", "q2"})
        self.assertEqual(len(first), 2)

    def test_duplicate_passages_are_separated(self):
        queries = {"q1": "one", "q2": "two", "q3": "three"}
        corpus = {"p1": "shared", "p2": "other"}
        positives = {"q1": ["p1"], "q2": ["p1"], "q3": ["p2"]}
        pairs = sample_one_positive_per_query(queries, corpus, positives, 42, 0)
        arranged = arrange_no_duplicate_batches(pairs, 2)
        for index in range(0, len(arranged), 2):
            batch = arranged[index:index + 2]
            self.assertEqual(len({pair.passage for pair in batch}), len(batch))


if __name__ == "__main__":
    unittest.main()
