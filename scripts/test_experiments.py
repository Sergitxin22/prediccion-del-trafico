import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_experiments import (
    ADJACENCIES, CLUSTERINGS, DEFAULTS, MODELS, atomic_json, inspect_datasets,
    plan_jobs, recover_jobs, signal_signature, summarize, training_command, validate_result,
)
from search_hyperparams import ALPHAS, ARCHITECTURES, LAYERS, WIDTHS, candidate, exhaustive_stage, first_stage, next_stage, select
from launch_search import main as launch_main


class ExperimentTests(unittest.TestCase):
    def test_matrix_has_forty_unique_conditions(self):
        jobs = plan_jobs(CLUSTERINGS, ADJACENCIES, MODELS)
        self.assertEqual(len(jobs), 40)
        self.assertEqual(len({job["id"] for job in jobs}), 40)
        self.assertTrue(all(job["status"] == "pending" for job in jobs))
        self.assertEqual(sum(job["model"] == "gat_gru" for job in jobs), 10)

    def test_command_preserves_protocol_and_selects_model(self):
        job = plan_jobs(["base"], ["correlation"], ["gat_gru"])[0]
        command = training_command(job, {"path": "/tmp/data with spaces"}, DEFAULTS, Path("/tmp/output with spaces"))
        self.assertEqual(command[command.index("--spatial")+1], "gat")
        self.assertEqual(command[command.index("--temporal")+1], "gru")
        self.assertEqual(command[command.index("--source")+1], "/tmp/data with spaces")
        self.assertEqual(command[command.index("--epochs")+1], "25")
        self.assertEqual(command[command.index("--batch-size")+1], "32")
        self.assertNotIn("--shuffle", command)

    def test_signal_fingerprint_ignores_archive_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a", root / "b"
            a.mkdir()
            b.mkdir()
            values = {"values": np.arange(6).reshape(3, 2), "observed": np.ones((3, 2), bool)}
            np.savez(a / "series.npz", **values)
            np.savez_compressed(b / "series.npz", **values)
            self.assertEqual(signal_signature(a), signal_signature(b))
            values["observed"][1, 0] = False
            np.savez_compressed(b / "series.npz", **values)
            self.assertNotEqual(signal_signature(a), signal_signature(b))

    def test_mismatched_masks_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for adjacency in ADJACENCIES:
                folder = root / "base" / adjacency
                folder.mkdir(parents=True)
                metadata = {"format_version": 1, "index": "pfm", "cluster": "base", "adjacency": adjacency, "n_nodes": 2, "horizon": 1}
                atomic_json(folder / "metadata.json", metadata)
                mask = np.ones((3, 2), bool)
                if adjacency == "correlation":
                    mask[0, 0] = False
                np.savez(folder / "series.npz", values=np.arange(6).reshape(3, 2), observed=mask)
                np.savez(folder / "graph.npz", nodes_idx=np.arange(2))
            with self.assertRaisesRegex(ValueError, "Unmatched"):
                inspect_datasets(root, ["base"], ADJACENCIES)

    def test_atomic_json_keeps_previous_value_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_json(path, {"value": 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_resume_refuses_an_active_incomplete_process(self):
        with tempfile.TemporaryDirectory() as directory:
            job = plan_jobs(["base"], ["closeness"], ["gcn_lstm"])[0]
            job.update(status="running", attempts=[{"folder": directory, "pid": 123, "log": str(Path(directory) / "train.log")}])
            manifest = {"jobs": [job], "datasets": {job["dataset"]: {}}, "training": DEFAULTS}
            with patch("run_experiments.pid_exists", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "still be active"):
                    recover_jobs(manifest, False)

    def test_resume_preserves_old_attempt_and_requeues_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "original_output"
            sentinel.write_text("keep")
            job = plan_jobs(["base"], ["closeness"], ["gcn_lstm"])[0]
            job.update(status="running", attempts=[{"folder": directory, "pid": 123, "log": str(Path(directory) / "train.log")}])
            manifest = {"jobs": [job], "datasets": {job["dataset"]: {}}, "training": DEFAULTS}
            with patch("run_experiments.pid_exists", return_value=False):
                recover_jobs(manifest, False)
            self.assertEqual(job["status"], "pending")
            self.assertEqual(job["attempts"][0]["status"], "interrupted")
            self.assertEqual(sentinel.read_text(), "keep")

    def test_completed_jobs_are_not_restarted_and_retries_are_explicit(self):
        jobs = plan_jobs(["base"], ["closeness"], ["gcn_lstm", "gat_lstm"])
        jobs[0]["status"] = "completed"
        jobs[1]["status"] = "failed"
        manifest = {"jobs": jobs}
        original = copy.deepcopy(manifest)
        recover_jobs(manifest, False)
        self.assertEqual(manifest, original)
        recover_jobs(manifest, True)
        self.assertEqual(jobs[0]["status"], "completed")
        self.assertEqual(jobs[1]["status"], "pending")

    def test_summary_selects_by_validation_within_each_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = plan_jobs(["base", "proximity_50"], ADJACENCIES, ["gcn_lstm"])
            datasets = {}
            for i, job in enumerate(jobs):
                folder = root / job["id"]
                folder.mkdir()
                score = {"mse": [0.3, 0.2, 0.04, 0.05][i], "mae": .1, "rmse": .2, "count": 10}
                baseline = {"mse": .1, "mae": .1, "rmse": .3, "count": 10}
                metrics = {"best_epoch": 1, "constant_prediction_node_indices": [],
                           "validation": {"model": score, "persistence": baseline}, "test": {"model": score, "persistence": baseline}}
                atomic_json(folder / "metrics.json", metrics)
                job.update(status="completed", attempts=[{"result_dir": str(folder), "gpu": i}])
                datasets[job["dataset"]] = {"n_nodes": 2}
            manifest = {"jobs": jobs, "datasets": datasets, "clusterings": ["base", "proximity_50"], "adjacencies": list(ADJACENCIES), "models": ["gcn_lstm"]}
            summarize(root, manifest)
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual([row["adjacency"] for row in summary["best_by_validation_within_each_clustering"]], ["correlation", "closeness"])
            jobs[0]["status"] = "pending"
            summarize(root, manifest)
            summary = json.loads((root / "summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(len(summary["best_by_validation_within_each_clustering"]), 1)

    def test_partial_outputs_are_not_accepted_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "output/exp_test"
            folder.mkdir(parents=True)
            job = plan_jobs(["base"], ["closeness"], ["gcn_lstm"])[0]
            dataset = {"n_nodes": 1, "horizon": 1, "signature": "example"}
            atomic_json(folder / "config.json", dict(DEFAULTS, spatial="gcn", temporal="lstm", dataset_signature="example"))
            score = {"mse": .1, "mae": .1, "rmse": .3, "count": 10}
            metrics = {"history": [None]*25, "validation": {"model": score, "persistence": score}, "test": {"model": score, "persistence": score}}
            atomic_json(folder / "metrics.json", metrics)
            for name in ("model.npz", "predictions.npz", "comparison.csv", "selected_predictions.csv", "node_0_h1.png"):
                (folder / name).write_bytes(b"fixture")
            log = root / "train.log"
            log.write_text("Epoch 25\n")
            attempt = {"folder": str(root), "log": str(log)}
            with self.assertRaisesRegex(ValueError, "completion message"):
                validate_result(attempt, job, dataset, DEFAULTS)
            log.write_text(f"Modelo, configuracion, metricas, predicciones y graficas: {folder}\n")
            result, _ = validate_result(attempt, job, dataset, DEFAULTS)
            self.assertEqual(result, str(folder))

    def test_staged_search_dimensions_and_metric_selection(self):
        initial = first_stage()
        self.assertEqual(len(initial), 80)
        self.assertEqual(len({job["id"] for job in initial}), 80)
        self.assertEqual((ALPHAS, LAYERS, WIDTHS), ((0, .2, .4, .6, .8, 1.), (1, 2, 3), (32, 64, 128)))
        self.assertEqual(set(job["architecture"] for job in initial), set(ARCHITECTURES))
        for index, job in enumerate(initial):
            job.update(status="completed", validation={"mse": (index % 4) / 10, "mae": (3 - index % 4) / 10})
        chosen = list(select(initial, ("cluster", "adjacency", "architecture")))
        self.assertEqual(len(chosen), 40)
        self.assertEqual({job["model"] for job in chosen}, {"gcn_lstm", "gat_gru"})
        manifest = {"stage": 0, "state": "pending", "jobs": initial}
        next_stage(manifest)
        self.assertEqual(len(manifest["jobs"]), 280)
        self.assertEqual(len([job for job in manifest["jobs"] if job["stage"] == 1]), 200)
        for job in manifest["jobs"]:
            if job["stage"] == 1:
                job.update(status="completed", validation={"mse": 1., "mae": 1.})
        next_stage(manifest)
        self.assertEqual(len([job for job in manifest["jobs"] if job["stage"] == 2]), 320)
        self.assertEqual(len({job["id"] for job in manifest["jobs"]}), 600)
        for job in manifest["jobs"]:
            if job["stage"] == 2:
                job.update(status="completed", validation={"mse": 1., "mae": 1.})
        next_stage(manifest)
        self.assertEqual(len(manifest["leaders"]), 50)
        self.assertEqual(manifest["stage"], 3)

    def test_exhaustive_search_covers_every_crossed_configuration(self):
        jobs = exhaustive_stage()
        self.assertEqual(len(jobs), 4320)
        self.assertEqual(len({job["id"] for job in jobs}), 4320)
        self.assertEqual({job["stage"] for job in jobs}, {2})
        self.assertEqual({job["time_alpha"] for job in jobs}, set(ALPHAS))
        self.assertEqual({job["layers"] for job in jobs}, set(LAYERS))
        self.assertEqual({job["hidden_feats"] for job in jobs}, set(WIDTHS))
        self.assertEqual(len([job for job in jobs if job["model"] == "gat_gru" and job["architecture"] == "pfm"]), 540)

    def test_upstream_training_defers_test_and_restores_finalist_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset"
            source.mkdir()
            times = np.arange("2025-01-01T00:00", "2025-01-01T08:00", dtype="datetime64[15m]").astype("datetime64[ns]")
            values = np.tile(np.arange(len(times), dtype=np.float32)[:, None]/len(times), (1, 2))
            np.savez(source / "series.npz", values=values, observed=np.ones_like(values, bool), times=times,
                     node_labels=np.arange(2), window=np.array(3), horizon=np.array(4),
                     train_starts=np.arange(3, 13, dtype=np.int32), valid_starts=np.arange(16, 20, dtype=np.int32),
                     test_starts=np.arange(23, 28, dtype=np.int32))
            np.savez(source / "graph.npz", nodes_idx=np.arange(2), node_labels=np.arange(2),
                     senders=np.array([0, 1]), receivers=np.array([1, 0]), edges=np.ones((2, 1), np.float32))
            atomic_json(source / "metadata.json", {"format_version": 1, "index": "pfm", "frequency": "15min"})
            (source / "nodes.csv").write_text("node_index,name\n0,a\n1,b\n")
            program = Path(__file__).resolve().parents[1] / "scripts/train.py"
            command = [sys.executable, "-B", str(program), "--source", str(source), "--target", str(root / "training"),
                       "--device", "cpu", "--epochs", "1", "--batch-size", "2", "--micro-batch-size", "1",
                       "--hidden-feats", "4", "--temporal-hidden", "4", "--layers", "1", "--architecture",
                       "upstream_bidirectional", "--time-alpha", ".2", "--defer-test", "--plot-points", "10"]
            env = dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg")
            subprocess.run(command, cwd=program.parents[1], env=env, check=True, capture_output=True, text=True)
            experiment, = (root / "training").glob("exp_*")
            self.assertFalse((experiment / "predictions.npz").exists())
            self.assertIsNone(json.loads((experiment / "metrics.json").read_text())["test"])
            subprocess.run([sys.executable, "-B", str(program), "--evaluate-only", str(experiment),
                            "--target", str(root / "evaluation"), "--device", "cpu", "--plot-points", "10"],
                           cwd=program.parents[1], env=env, check=True, capture_output=True, text=True)
            evaluated, = (root / "evaluation").glob("exp_*")
            metrics = json.loads((evaluated / "metrics.json").read_text())
            self.assertEqual(len(metrics["test"]["model"]["per_horizon"]), 4)
            self.assertTrue((evaluated / "predictions.npz").is_file())

    def test_exhaustive_stage_selects_test_finalists_only_after_validation(self):
        jobs = exhaustive_stage()
        for index, job in enumerate(jobs):
            job.update(status="completed", validation={"mse": (index % 13)/100, "mae": (index % 17)/100})
        manifest = {"stage": 2, "state": "pending", "jobs": jobs}
        next_stage(manifest)
        self.assertEqual(manifest["stage"], 3)
        self.assertEqual(len(manifest["leaders"]), 50)
        finalists = [job for job in manifest["jobs"] if job["kind"] == "final"]
        self.assertGreater(len(finalists), 0)
        self.assertLessEqual(len(finalists), 50)
        self.assertEqual({job["stage"] for job in finalists}, {3})
        self.assertTrue(all(job["source_id"] in {item["id"] for item in jobs} for job in finalists))

    def test_launcher_detaches_and_persists_pid_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "results"
            with patch.object(sys, "argv", ["launch_search.py", "--detach", "--target", str(target)]), \
                 patch("launch_search.subprocess.Popen") as spawn:
                spawn.return_value.pid = 54321
                launch_main()
            launch, = target.glob("launch_*")
            details = json.loads((launch / "pid.json").read_text())
            self.assertEqual(details["pid"], 54321)
            self.assertTrue((launch / "launcher.log").is_file())
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertEqual(spawn.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_launcher_orders_data_generation_before_exhaustive_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "new_dataset"
            target = Path(directory) / "results"
            dataset_info = {f"{group}/{graph}": {"horizon": 4} for group in CLUSTERINGS for graph in ADJACENCIES}
            with patch.object(sys, "argv", ["launch_search.py", "--run", "--data", str(source), "--target", str(target), "--gpus", "0", "1"]), \
                 patch("launch_search.subprocess.run") as execute, \
                 patch("launch_search.inspect_datasets", return_value=dataset_info):
                launch_main()
            self.assertEqual(execute.call_count, 2)
            generate, search = (call.args[0] for call in execute.call_args_list)
            self.assertIn("generate_datasets.py", generate[2])
            self.assertEqual(generate[generate.index("--horizon")+1], "4")
            self.assertIn("search_hyperparams.py", search[2])
            self.assertIn("--exhaustive", search)
            self.assertEqual(search[search.index("--gpus")+1:search.index("--exhaustive")], ["0", "1"])
            self.assertFalse(source.exists())
            self.assertFalse(target.exists())

    def test_search_rejects_incomplete_stage(self):
        manifest = {"stage": 0, "jobs": [candidate("base", "closeness", "pfm", "gcn_lstm")]}
        with self.assertRaisesRegex(ValueError, "missing or failed"):
            next_stage(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
