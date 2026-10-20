import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_group_graph import CASES, compare, grouped_stats, squared_error


class CompareGroupGraphTests(unittest.TestCase):
    def test_group_projection_uses_only_observed_individual_targets(self):
        values = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.], [4., 5.]], dtype=np.float32)
        observed = np.ones_like(values, bool)
        observed[1, 1] = False
        stats = grouped_stats(values, observed, np.array([0]), np.array([0, 1]), np.array([0, 0]), 1)
        self.assertEqual(stats[0][0, 1, 0], 1)
        self.assertAlmostEqual(squared_error(np.zeros((1, 4, 1), dtype=np.float32), stats),
                               (1 + 1 + 4 + 9 + 9 + 16) / 7)

    def test_ten_cases_share_targets_and_omit_missing_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            metrics = run / "curves_precomputed_v1" / "metrics"
            predictions = run / "curves_precomputed_v1" / "predictions"
            metrics.mkdir(parents=True)
            predictions.mkdir()
            times = np.arange(12, dtype="int64").astype("datetime64[h]")
            values = np.stack([np.arange(12) / 12, np.arange(12) / 12 + 10,
                               np.arange(12) / 12 + .5], axis=1).astype(np.float32)
            observed = np.ones_like(values, dtype=bool)
            observed[7, 2] = False
            starts = np.array([5, 6], dtype=np.int32)
            datasets, jobs = {}, []
            for label, group, graph in CASES:
                source = root / label
                source.mkdir()
                grouped = group != "base"
                with (source / "node_map.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("sensor_id", "node_index"))
                    writer.writerows(((100, 0), (101, 0 if grouped else 2)))
                    if not grouped:
                        writer.writerow((102, 1))
                np.savez(source / "series.npz", times=times, test_starts=starts,
                         values=values[:, [0, 2]].mean(axis=1, keepdims=True) if grouped else values, observed=observed)
                datasets[f"{group}/{graph}"] = {"path": str(source), "signature": label,
                                                 "signal_signature": group, "n_nodes": 1 if grouped else 3}
                job_id = label + "_candidate"
                jobs.append({"id": job_id, "kind": "candidate", "cluster": group, "adjacency": graph,
                             "architecture": "pfm", "model": "gcn_lstm", "time_alpha": 1,
                             "layers": 2, "hidden_feats": 32, "status": "completed"})
                true = np.stack([values[starts + h] for h in range(4)], axis=1)
                prediction = true[:, :, [0, 2]].mean(axis=2, keepdims=True) if grouped else true
                np.save(predictions / f"{job_id}.npy", prediction.astype(np.float32))
                (metrics / f"{job_id}.json").write_text(json.dumps({"job_id": job_id, "dataset_signature": label}))
            (run / "manifest.json").write_text(json.dumps({"state": "completed", "datasets": datasets}))
            (run / "summary.json").write_text(json.dumps({"results": jobs}))
            (run / "curves_precomputed_v1" / "manifest.json").write_text(json.dumps({
                "state": "completed", "completed": [job["id"] for job in jobs], "failed": {}}))
            count, sensors = compare(run, root / "analysis")
            self.assertEqual((count, sensors), (1, 2))
            with (root / "analysis" / "paired_mse_ratio.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["Ba1"]), 0)
            self.assertGreater(float(row["c1a1"]), 0)
            with (root / "analysis" / "paired_effects.csv").open(newline="") as stream:
                effects = list(csv.DictReader(stream))
            self.assertEqual(next(effect for effect in effects if effect["case"] == "c1a1" and
                                  effect["versus"] == "Ba1" and effect["architecture"] == "all")["improved"], "0")
            with (root / "analysis" / "paired_baselines.csv").open(newline="") as stream:
                baselines = list(csv.DictReader(stream))
            self.assertEqual(len(baselines), 10)
            self.assertAlmostEqual(float(baselines[0]["ratio_to_individual_persistence"]), 1)
            self.assertEqual(json.loads((root / "analysis" / "metadata.json").read_text())["evaluated_targets"], 14)
            one, _ = compare(run, root / "h1", horizon="1")
            four, _ = compare(run, root / "h4", horizon="4")
            self.assertEqual((one, four), (1, 1))
            h1 = json.loads((root / "h1" / "metadata.json").read_text())
            h4 = json.loads((root / "h4" / "metadata.json").read_text())
            self.assertEqual((h1["horizons"], h1["evaluated_targets"]), ([1], 4))
            self.assertEqual((h4["horizons"], h4["evaluated_targets"]), ([4], 4))
            with (root / "h1" / "paired_mse_ratio.csv").open(newline="") as stream:
                h1_row = next(csv.DictReader(stream))
            with (root / "h4" / "paired_mse_ratio.csv").open(newline="") as stream:
                h4_row = next(csv.DictReader(stream))
            self.assertEqual(float(h1_row["Ba1"]), 0)
            self.assertGreater(float(h1_row["c1a1"]), float(h4_row["c1a1"]))

    def test_invalid_horizon_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with self.assertRaises(ValueError):
                compare(Path(directory), output, horizon="8")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
