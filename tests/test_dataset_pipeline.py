import csv
import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DatasetPipelineTest(unittest.TestCase):
    def test_split_counts_and_disjoint_paths(self) -> None:
        expected_per_identity = {"train": 400, "val": 50, "test": 50}
        paths_by_split: dict[str, set[str]] = {}
        for split, count in expected_per_identity.items():
            with (ROOT / "dataset" / "splits" / f"{split}.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(Counter(row["identity"] for row in rows), {
                "you": count,
                "teammate": count,
                "professor": count,
            })
            paths_by_split[split] = {row["path"] for row in rows}

        self.assertFalse(paths_by_split["train"] & paths_by_split["val"])
        self.assertFalse(paths_by_split["train"] & paths_by_split["test"])
        self.assertFalse(paths_by_split["val"] & paths_by_split["test"])
        self.assertEqual(len(set.union(*paths_by_split.values())), 1500)

    def test_train_only_references_are_normalized(self) -> None:
        for identity in ("you", "teammate", "professor"):
            vector = np.load(ROOT / "reference_embeddings" / f"{identity}_ref.npy")
            self.assertEqual(vector.size, 512)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_recorded_test_metrics_are_complete(self) -> None:
        metrics = json.loads((ROOT / "report" / "results" / "metrics.json").read_text())
        self.assertEqual(metrics["detected_faces"], {"train": 1200, "val": 150, "test": 150})
        test = metrics["splits"]["test"]
        self.assertEqual(test["evaluated"], 150)
        self.assertEqual(test["target_trigger"]["false_positive"], 0)
        self.assertEqual(test["target_trigger"]["false_negative"], 0)


if __name__ == "__main__":
    unittest.main()
