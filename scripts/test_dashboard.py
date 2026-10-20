import csv
import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from dashboard import MODELS, ResultsCatalog, SearchCatalog, handler_for


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        groups = ("base", "proximity_50")
        graphs = ("closeness", "correlation")
        results, datasets = [], {}
        times = np.array(["2025-01-27T08:15:00", "2025-01-30T08:15:00", "2025-01-31T23:45:00"], dtype="datetime64[ns]")
        for group in groups:
            n = 2 if group == "base" else 1
            target = np.array([[.2, .3], [.6, .7], [.8, .9]], np.float32) if n == 2 else np.array([[.25], [.6], [.9]], np.float32)
            observed = np.array([[True, True], [True, False], [False, True]]) if n == 2 else np.ones((3, 1), bool)
            for graph in graphs:
                source = self.root / "datasets" / group / graph
                source.mkdir(parents=True)
                datasets[f"{group}/{graph}"] = {"path": str(source), "n_nodes": n, "n_sensors": 2,
                                                 "signal_signature": f"equal-{group}", "horizon": 1}
                with (source / "node_map.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["sensor_id", "node_label", "name", "node_index"])
                    writer.writerow([3937, 0, "Serrano <b>este</b>", 0])
                    writer.writerow([5947, 1 if n == 2 else 0, "Otra calle", 1 if n == 2 else 0])
                for index, model in enumerate(MODELS):
                    folder = self.root / "jobs" / group / graph / model
                    folder.mkdir(parents=True)
                    offset = .01 * (index + (graph == "correlation"))
                    array = lambda value: value[:, None, :, None]
                    np.savez_compressed(folder / "predictions.npz", model=array(target + offset), target=array(target),
                                        observed=array(observed), persistence=array(target - .1), window_mean=array(target - .05),
                                        target_times=times[:, None], origin_times=times - np.timedelta64(15, "m"),
                                        node_labels=np.arange(n, dtype=np.int64))
                    results.append({"cluster": group, "adjacency": graph, "model": model, "status": "completed",
                                    "result_dir": str(folder), "valid_mse": .01 * (index+1) + (.004 if graph == "correlation" else 0)})
        (self.root / "summary.json").write_text(json.dumps({"complete": True, "results": results}), encoding="utf-8")
        (self.root / "manifest.json").write_text(json.dumps({"state": "completed", "clusterings": groups,
                                                              "adjacencies": graphs, "models": list(MODELS), "datasets": datasets}), encoding="utf-8")
        self.catalog = ResultsCatalog(self.root)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.catalog))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def get(self, path, method="GET"):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            content = response.read()
            return response.status, content, dict(response.getheaders())
        finally:
            connection.close()

    def json(self, path):
        status, content, _ = self.get(path)
        return status, json.loads(content)

    def test_metadata_and_sensor_lookup(self):
        status, info = self.json("/api/meta")
        self.assertEqual(status, 200)
        self.assertEqual(len(info["groups"]), 2)
        self.assertEqual(len(info["models"]), 4)
        self.assertEqual(len(info["graphs"]), 3)
        status, sensors = self.json("/api/sensors?group=base")
        self.assertEqual(status, 200)
        self.assertEqual(sensors["sensors"][0]["name"], "Serrano <b>este</b>")
        self.assertEqual(sensors["sensors"][0]["members"], 1)

    def test_four_models_have_same_target_and_masked_sensor_scores(self):
        status, data = self.json("/api/series?group=base&graph=closeness&sensor=3937")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["scores"]), 4)
        self.assertEqual(data["target"], [data["target"][0], data["target"][1], None])
        self.assertEqual(data["observed_count"], 2)
        self.assertEqual(data["points"], 3)
        self.assertEqual(data["origin_time"][0], "2025-01-27T08:00:00")
        self.assertAlmostEqual(data["scores"][0]["mse"], 0)
        self.assertAlmostEqual(data["reference_scores"]["persistence"]["mse"], .01, places=5)
        self.assertEqual(data["validation_leader"]["model"], "gcn_lstm")

    def test_both_graphs_use_identical_measurements(self):
        status, both = self.json("/api/series?group=base&graph=both&sensor=3937")
        self.assertEqual(status, 200)
        self.assertEqual(len(both["scores"]), 8)
        self.assertEqual(both["prediction"]["closeness_gcn_lstm"],
                         self.catalog.series("base", "closeness", 3937)["prediction"]["gcn_lstm"])

    def test_aggregated_sensor_is_clearly_grouped(self):
        status, data = self.json("/api/series?group=proximity_50&graph=correlation&sensor=5947")
        self.assertEqual(status, 200)
        self.assertTrue(data["aggregated"])
        self.assertEqual(data["sensor"]["members"], 2)
        self.assertEqual(data["sensor"]["node"], 0)
        self.assertEqual(data["observed_count"], 3)
        self.assertAlmostEqual(data["target"][0], .25, places=6)

    def test_ranges_recalculate_scores_and_handle_no_observations(self):
        status, short = self.json("/api/series?group=base&graph=closeness&sensor=3937&range=24h")
        self.assertEqual(status, 200)
        self.assertEqual(short["points"], 1)
        self.assertEqual(short["observed_count"], 0)
        self.assertEqual(short["target"], [None])
        self.assertIsNone(short["scores"][0]["mse"])
        self.assertIsNone(short["test_leader"])
        status, wider = self.json("/api/series?group=base&graph=closeness&sensor=3937&range=48h")
        self.assertEqual(status, 200)
        self.assertEqual(wider["points"], 2)
        self.assertEqual(wider["observed_count"], 1)
        self.assertIsNotNone(wider["scores"][0]["mse"])

    def test_invalid_parameters_and_missing_sensor_are_rejected(self):
        for path, expected in (("/api/series?group=base&graph=closeness&sensor=123456", 404),
                               ("/api/series?group=../../etc&graph=closeness&sensor=3937", 400),
                               ("/api/series?group=base&graph=closeness&sensor=../3937", 400),
                               ("/api/series?group=base&graph=closeness&sensor=3937&range=bad", 400)):
            with self.subTest(path=path):
                status, body = self.json(path)
                self.assertEqual(status, expected)
                self.assertIn("error", body)

    def test_unmatched_model_files_are_rejected(self):
        path = self.root / "jobs/base/closeness/gat_gru/predictions.npz"
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        data["observed"][0, 0, 0, 0] = False
        np.savez_compressed(path, **data)
        status, body = self.json("/api/series?group=base&graph=closeness&sensor=3937")
        self.assertEqual(status, 409)
        self.assertIn("mismos objetivos", body["error"])

    def test_static_page_and_head_are_available_without_external_scripts(self):
        status, page, headers = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"/assets/dashboard.js", page)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        status, _, _ = self.get("/", method="HEAD")
        self.assertEqual(status, 200)
        status, script, _ = self.get("/assets/dashboard.js")
        self.assertEqual(status, 200)
        self.assertIn(b"/api/meta", script)

    def test_search_finalists_keep_four_horizons_separate(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["format_version"] = 2
        manifest["state"] = "completed"
        summary = json.loads((self.root / "summary.json").read_text())
        for group in ("base", "proximity_50"):
            for graph in ("closeness", "correlation"):
                source = Path(manifest["datasets"][f"{group}/{graph}"]["path"])
                manifest["datasets"][f"{group}/{graph}"]["horizon"] = 4
                manifest["datasets"][f"{group}/{graph}"]["signature"] = f"dataset-{group}-{graph}"
                np.savez(source / "series.npz", times=np.array(["2025-01-31T23:00:00", "2025-01-31T23:15:00",
                                                                   "2025-01-31T23:30:00", "2025-01-31T23:45:00"], dtype="datetime64[ns]"),
                         test_starts=np.array([0]))
        for row in summary["results"]:
            row.update(id=f"final__{row['cluster']}__{row['adjacency']}__{row['model']}", kind="final",
                       architecture="pfm" if row["model"].startswith("gcn") else "upstream_bidirectional",
                       time_alpha=1., layers=2, hidden_feats=4,
                       validation={"mse": row["valid_mse"], "mae": row["valid_mse"]})
            folder = Path(row["result_dir"])
            with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                values = {key: saved[key] for key in saved.files}
            for key in ("model", "target", "observed", "persistence", "window_mean"):
                values[key] = np.repeat(values[key], 4, axis=1)
            values["target_times"] = values["origin_times"][:, None] + np.arange(1, 5) * np.timedelta64(15, "m")
            np.savez_compressed(folder / "predictions.npz", **values)
            scores = {"mse": row["valid_mse"], "mae": row["valid_mse"], "count": 10,
                      "per_horizon": [{"mse": row["valid_mse"], "mae": row["valid_mse"], "count": 2} for _ in range(4)]}
            (folder / "metrics.json").write_text(json.dumps({"test": {"model": scores, "persistence": scores}}))
        candidates = []
        for final in summary["results"]:
            job_id = final["id"].removeprefix("final__")
            final["source_id"] = job_id
            candidates.append(dict(final, id=job_id, kind="candidate", result_dir=None,
                                   validation=dict(final["validation"], per_horizon=[
                                       {"mse": final["valid_mse"], "mae": final["valid_mse"]} for _ in range(4)])))
        summary["results"].extend(candidates)
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "summary.json").write_text(json.dumps(summary))
        catalog = SearchCatalog(self.root)
        self.assertEqual(catalog.metadata()["horizons"], 4)
        early = catalog.series("base", "both", 3937, horizon=1)
        late = catalog.series("base", "both", 3937, horizon=4)
        self.assertEqual(len(late["scores"]), 4)
        self.assertEqual(late["horizon"], 4)
        self.assertEqual(late["ahead_minutes"], 60.)
        self.assertEqual(early["ahead_minutes"], 15.)
        self.assertEqual(late["observed_count"], 2)
        self.assertNotEqual(late["time"][0], early["time"][0])
        with self.assertRaisesRegex(ValueError, "horizonte"):
            catalog.series("base", "both", 3937, horizon=5)
        ranking = catalog.rankings("base", "both")
        self.assertEqual(ranking["total"], 8)
        self.assertEqual(ranking["evaluated"], 8)
        self.assertEqual(ranking["mse"][0]["id"], "base__closeness__gcn_lstm")
        comparison = catalog.comparison("base", "both", "pfm", "4")
        self.assertEqual(len(comparison["rows"]), 8)
        selected = catalog.series("base", "both", 3937, horizon=4,
                                  configurations=["base__closeness__gcn_lstm", "base__correlation__gat_gru"])
        self.assertEqual(len(selected["scores"]), 2)
        with self.assertRaisesRegex(ValueError, "cuatro"):
            catalog.series("base", "both", 3937, configurations=["bad"] * 5)
        extra = dict(candidates[0], id="base__closeness__extra", validation={
            "mse": .001, "mae": .001, "per_horizon": [{"mse": .001, "mae": .001} for _ in range(4)]})
        catalog.candidates[extra["id"]] = extra
        self.assertEqual(catalog.rankings("base", "closeness")["evaluated"], 4)
        curves = self.root / "curves_precomputed_v1"
        (curves / "metrics").mkdir(parents=True)
        (curves / "predictions").mkdir()
        final_folder = Path(summary["results"][0]["result_dir"])
        with np.load(final_folder / "predictions.npz", allow_pickle=False) as saved:
            np.save(curves / "predictions" / f"{extra['id']}.npy", saved["model"][..., 0])
        test_score = {"mse": .004, "mae": .045, "count": 10,
                      "per_horizon": [{"mse": .004, "mae": .045, "count": 2} for _ in range(4)]}
        (curves / "metrics" / f"{extra['id']}.json").write_text(json.dumps({
            "job_id": extra["id"], "dataset_signature": manifest["datasets"]["base/closeness"]["signature"],
            "test": {"model": test_score, "persistence": test_score}}))
        (curves / "manifest.json").write_text(json.dumps({"state": "running", "completed": [extra["id"]], "failed": {}}))
        updated = catalog.rankings("base", "closeness")
        self.assertEqual(updated["evaluated"], 5)
        self.assertEqual(updated["mse"][0]["id"], extra["id"])
        self.assertEqual(catalog.series("base", "closeness", 3937, horizon=2,
                                        configurations=[extra["id"]])["scores"][0]["id"], extra["id"])
        with self.assertRaisesRegex(ValueError, "incompatible"):
            catalog.series("base", "correlation", 3937, configurations=[extra["id"]])
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(catalog))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path, expected in (
                ("/api/rankings?group=base&graph=closeness&phase=test&horizon=2", 200),
                ("/api/comparison?group=base&graph=both&architecture=pfm&horizon=4", 200),
                ("/api/series?group=base&graph=closeness&sensor=3937&configs=base__closeness__extra", 200),
                ("/api/series?group=base&graph=correlation&sensor=3937&configs=base__closeness__extra", 400),
                ("/api/rankings?group=../../etc&graph=both", 400),
            ):
                with self.subTest(path=path):
                    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                    try:
                        connection.request("GET", path)
                        response = connection.getresponse()
                        self.assertEqual(response.status, expected)
                        response.read()
                    finally:
                        connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
