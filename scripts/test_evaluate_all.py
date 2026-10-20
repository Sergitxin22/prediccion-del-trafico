import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_all import evaluate_one, output_paths, score, valid_output
from train import dataset_signature


class EvaluateAllTests(unittest.TestCase):
    def test_scores_count_only_observed_nodes_and_horizons(self):
        square = np.array([[1., 0.], [4., 9.]], np.float64)
        absolute = np.sqrt(square)
        count = np.array([[1, 0], [1, 1]])
        result = score(square, absolute, count)
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["mse"], 14 / 3)
        self.assertEqual(result["per_horizon"][0]["count"], 1)

    def test_saved_checkpoint_evaluates_every_node_without_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset"
            source.mkdir()
            times = np.arange("2025-01-01T00:00", "2025-01-01T08:00", dtype="datetime64[15m]").astype("datetime64[ns]")
            values = np.tile(np.arange(len(times), dtype=np.float32)[:, None] / len(times), (1, 2))
            observed = np.ones_like(values, bool)
            observed[27, 1] = False
            np.savez(source / "series.npz", values=values, observed=observed, times=times,
                     node_labels=np.arange(2), window=np.array(3), horizon=np.array(4),
                     train_starts=np.arange(3, 13, dtype=np.int32), valid_starts=np.arange(16, 20, dtype=np.int32),
                     test_starts=np.arange(23, 28, dtype=np.int32))
            np.savez(source / "graph.npz", nodes_idx=np.arange(2), node_labels=np.arange(2),
                     senders=np.array([0, 1]), receivers=np.array([1, 0]), edges=np.ones((2, 1), np.float32))
            (source / "metadata.json").write_text(json.dumps({"format_version": 1, "index": "pfm", "frequency": "15min"}))
            (source / "nodes.csv").write_text("node_index,name\n0,a\n1,b\n")
            program = Path(__file__).resolve().parent / "train.py"
            subprocess.run([sys.executable, "-B", str(program), "--source", str(source), "--target", str(root / "train"),
                            "--device", "cpu", "--epochs", "1", "--batch-size", "2", "--micro-batch-size", "1",
                            "--hidden-feats", "4", "--temporal-hidden", "4", "--layers", "1", "--time-alpha", ".2",
                            "--defer-test"], cwd=program.parents[1], env=dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg"),
                           check=True, capture_output=True, text=True)
            experiment, = (root / "train").glob("exp_*")
            target = root / "curves"
            for name in ("predictions", "metrics", "per_node"):
                (target / name).mkdir(parents=True)
            job = {"id": "tiny_case", "result_dir": str(experiment), "architecture": "pfm",
                   "layers": 1, "hidden_feats": 4, "time_alpha": .2}
            dataset = {"path": str(source), "signature": dataset_signature(source), "n_nodes": 2}
            evaluate_one(root, target, job, dataset, 2, device_kind="cpu")
            self.assertTrue(valid_output(target, job, dataset))
            forecast, metrics_path, nodes_path = output_paths(target, job["id"])
            with np.load(nodes_path, allow_pickle=False) as nodes:
                self.assertEqual(nodes["count"].shape, (4, 2))
                self.assertLess(nodes["count"].sum(), 5 * 4 * 2)
            prediction = np.load(forecast, mmap_mode="r")
            self.assertEqual(prediction.shape, (5, 4, 2))
            self.assertTrue(np.isfinite(prediction).all())
            metrics = json.loads(metrics_path.read_text())
            self.assertEqual(metrics["test"]["model"]["count"], metrics["test"]["persistence"]["count"])
            self.assertEqual(len(metrics["test"]["model"]["per_horizon"]), 4)
            self.assertFalse(list(target.rglob("*.png")))
            evaluate_one(root, target, job, dataset, 2, device_kind="cpu")


if __name__ == "__main__":
    unittest.main()
