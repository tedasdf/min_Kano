import unittest


class DeviceTransferContractTest(unittest.TestCase):
    def test_trainer_transfers_each_sentence_feature(self):
        """Regression guard for CPU token IDs with a CUDA-resident model."""
        from pathlib import Path

        source = Path("src/mini_kanon3/capabilities/embed/trainer.py").read_text(encoding="utf-8")
        self.assertIn("util.batch_to_device(feature, model.device)", source)


if __name__ == "__main__":
    unittest.main()
