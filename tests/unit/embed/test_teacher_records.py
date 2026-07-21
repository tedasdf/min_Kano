import unittest

from mini_kanon3.capabilities.embed.distillation import _validate_teacher_records


class TeacherRecordValidationTest(unittest.TestCase):
    def test_positive_must_be_first(self):
        records = [{"query_id": "q1", "candidates": [
            {"passage_id": "p2", "teacher_score": 1.0, "is_positive": False},
            {"passage_id": "p1", "teacher_score": 2.0, "is_positive": True},
        ]}]
        with self.assertRaises(ValueError):
            _validate_teacher_records(records, {"q1": "query"}, {"p1": "one", "p2": "two"})


if __name__ == "__main__": unittest.main()
