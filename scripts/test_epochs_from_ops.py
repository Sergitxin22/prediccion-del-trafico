import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from epochs_from_ops import (build_rows, epoch_count, load_budgets, main, operation_count,
                             parse_count, training_factors)
from run_experiments import DEFAULTS


def write_dataset(root, cluster, adjacency, nodes, window, train, horizon=1):
    folder = Path(root) / cluster / adjacency
    folder.mkdir(parents=True)
    np.savez(folder / "series.npz", values=np.zeros((window + horizon + train, nodes), np.float32),
             window=np.array(window), horizon=np.array(horizon), train_starts=np.arange(train, dtype=np.int32))
    np.savez(folder / "graph.npz", nodes_idx=np.arange(nodes, dtype=np.int32))
    (folder / "metadata.json").write_text(json.dumps({
        "format_version": 1, "index": "pfm", "cluster": cluster, "adjacency": adjacency,
        "n_nodes": nodes, "window": window, "horizon": horizon,
        "split_samples": {"train": train, "valid": 1, "test": 1},
    }), encoding="utf-8")
    return folder


class EpochBudgetTests(unittest.TestCase):
    def test_spatial_block_runs_once_per_window_step(self):
        spatial, temporal, n_ops = operation_count(nodes=5, layers=2, time_steps=4, graphs=7, x=10, y=3)
        self.assertEqual((spatial, temporal, n_ops), (2800, 420, 3220))
        self.assertEqual(epoch_count(10000, n_ops), 3)
        self.assertEqual(epoch_count(1119, n_ops), 0)
        self.assertEqual(parse_count("1e12", "MAX_OPS"), 10**12)

    def test_factors_come_from_the_saved_training_split(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = write_dataset(directory, "proximity_50", "closeness", nodes=11, window=12, train=40)
            self.assertEqual(training_factors(folder), {
                "dataset": str(folder), "cluster": "proximity_50", "adjacency": "closeness",
                "n": 11, "T": 12, "k": 40,
            })
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            metadata["n_nodes"] = 10
            (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "número de nodos"):
                training_factors(folder)

    def test_each_dataset_and_model_gets_its_own_epoch_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, "base", "closeness", nodes=5, window=4, train=7)
            write_dataset(root, "proximity_500", "correlation", nodes=2, window=4, train=7)
            budgets = load_budgets_from(root / "budgets.json", {
                "gcn_lstm": {"x": 10, "y": 3, "max_ops": 10000},
                "gat_gru": {"x": 8, "y": 1, "max_ops": 5000},
            })
            rows = build_rows([root / "base" / "closeness", root / "proximity_500" / "correlation"],
                              budgets, [DEFAULTS["layers"]])
            by_key = {(row["cluster"], row["model"]): row["n_epochs"] for row in rows}
            self.assertEqual(by_key[("base", "gcn_lstm")], 3)
            self.assertEqual(by_key[("proximity_500", "gcn_lstm")], 7)
            self.assertEqual(by_key[("base", "gat_gru")], 2)
            self.assertEqual(by_key[("proximity_500", "gat_gru")], 5)
            output = root / "epocas.csv"
            written = main(["--source", str(root), "--budgets", str(root / "budgets.json"),
                            "--layers", "2", "--output", str(output)])
            self.assertEqual(len(written), 4)
            with output.open(encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual([row["n_epochs"] for row in saved], ["3", "2", "7", "5"])

    def test_unknown_model_and_duplicate_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, "base", "closeness", nodes=2, window=2, train=2)
            with self.assertRaisesRegex(ValueError, "desconocido"):
                main(["--source", str(root), "--budget", "cnn_lstm", "1", "1", "10"])
            with self.assertRaisesRegex(ValueError, "una vez"):
                main(["--source", str(root), "--budget", "gcn_lstm", "1", "1", "10",
                      "--budget", "gcn_lstm", "2", "2", "20"])


def load_budgets_from(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return load_budgets(path)


if __name__ == "__main__":
    unittest.main()
