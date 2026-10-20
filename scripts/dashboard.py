import argparse
import csv
import json
import math
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

MODELS = {
    "gcn_lstm": "GCN-LSTM",
    "gcn_gru": "GCN-GRU",
    "gat_lstm": "GAT-LSTM",
    "gat_gru": "GAT-GRU",
}
GROUPS = {
    "base": "Sensores individuales",
    "proximity_50": "Proximidad 50",
    "proximity_500": "Proximidad 500",
    "proximity_behaviour_50": "Proximidad + comportamiento 50",
    "proximity_behaviour_500": "Proximidad + comportamiento 500",
}
GRAPHS = {"closeness": "Cercanía", "correlation": "Correlación", "both": "Ambos grafos"}
ASSETS = {"/": ("dashboard.html", "text/html; charset=utf-8"),
          "/assets/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
          "/assets/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8")}


class SensorNotFound(LookupError):
    pass


class DataMismatch(ValueError):
    pass


class ResultsCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if not summary.get("complete") or manifest.get("state") != "completed":
            raise ValueError("La batería todavía no está completa; esta web requiere todas las condiciones terminadas.")
        self.groups = tuple(manifest["clusterings"])
        self.adjacencies = tuple(manifest["adjacencies"])
        self.models = tuple(manifest["models"])
        if set(self.models) != set(MODELS) or not set(self.groups).issubset(GROUPS) or set(self.adjacencies) != {"closeness", "correlation"}:
            raise ValueError("La batería no contiene las cuatro arquitecturas y los dos grafos esperados.")
        if len(summary["results"]) != len(self.groups)*len(self.adjacencies)*len(self.models):
            raise ValueError("Faltan resultados para alguna combinación.")
        self.datasets = manifest["datasets"]
        self.results = {}
        for row in summary["results"]:
            key = (row["cluster"], row["adjacency"], row["model"])
            if row["status"] != "completed" or key in self.results or key[0] not in self.groups or key[1] not in self.adjacencies or key[2] not in self.models:
                raise ValueError("Hay resultados duplicados o incompletos.")
            folder = Path(row["result_dir"]).resolve(strict=True)
            if not folder.is_relative_to(self.root) or not (folder / "predictions.npz").is_file():
                raise ValueError("Falta un fichero de predicciones dentro de la batería elegida.")
            self.results[key] = dict(row, result_dir=folder)
        for cluster in self.groups:
            sources = [self.datasets[f"{cluster}/{adjacency}"] for adjacency in self.adjacencies]
            if sources[0]["signal_signature"] != sources[1]["signal_signature"]:
                raise DataMismatch("Los dos grafos no comparten exactamente los mismos datos.")
            if any((cluster, adjacency, model) not in self.results for adjacency in self.adjacencies for model in self.models):
                raise ValueError(f"Faltan modelos para {cluster}.")

    def metadata(self):
        first = self._signals(self.groups[0], self.adjacencies[0])
        return {
            "groups": [{"key": group, "label": GROUPS[group], "nodes": self.datasets[f"{group}/closeness"]["n_nodes"],
                        "sensors": self.datasets[f"{group}/closeness"]["n_sensors"]} for group in self.groups],
            "graphs": [{"key": name, "label": GRAPHS[name]} for name in (*self.adjacencies, "both")],
            "models": [{"key": name, "label": MODELS[name]} for name in self.models],
            "period": {"start": np.datetime_as_string(first["times"][0], unit="s"),
                       "end": np.datetime_as_string(first["times"][-1], unit="s")},
            "note": "Serie histórica de prueba, no tráfico en directo. El índice C del PFM no es una probabilidad de atasco.",
        }

    @lru_cache(maxsize=5)
    def sensors(self, cluster):
        self._check_group(cluster)
        source = Path(self.datasets[f"{cluster}/closeness"]["path"])
        rows = []
        with (source / "node_map.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append({"id": int(row["sensor_id"]), "name": row["name"].strip(), "node": int(row["node_index"]),
                             "node_label": int(row["node_label"])})
        if len(rows) != len({row["id"] for row in rows}):
            raise DataMismatch("Hay sensores duplicados en la correspondencia de nodos.")
        counts = Counter(row["node"] for row in rows)
        for row in rows:
            row["members"] = counts[row["node"]]
        return tuple(rows)

    def _check_group(self, group):
        if group not in self.groups:
            raise ValueError("Agrupación desconocida.")

    @lru_cache(maxsize=2)
    def _signals(self, group, adjacency):
        self._check_group(group)
        if adjacency not in self.adjacencies:
            raise ValueError("Grafo desconocido.")
        dataset = self.datasets[f"{group}/{adjacency}"]
        predictions = {}
        reference = None
        for model in self.models:
            folder = self.results[group, adjacency, model]["result_dir"]
            with np.load(folder / "predictions.npz", allow_pickle=False) as data:
                forecast = data["model"]
                if forecast.ndim != 4 or forecast.shape[1] != 1 or forecast.shape[2] != dataset["n_nodes"] or forecast.shape[3] != 1:
                    raise DataMismatch("El fichero no corresponde a predicciones de un paso para estos nodos.")
                predictions[model] = forecast[:, 0, :, 0]
                current = {"target": data["target"][:, 0, :, 0], "observed": data["observed"][:, 0, :, 0],
                           "times": data["target_times"][:, 0], "origin_times": data["origin_times"],
                           "node_labels": data["node_labels"]}
                if reference is None:
                    reference = dict(current, persistence=data["persistence"][:, 0, :, 0],
                                     window_mean=data["window_mean"][:, 0, :, 0])
                elif any(not np.array_equal(current[key], reference[key]) for key in current):
                    raise DataMismatch("Los cuatro modelos no tienen los mismos objetivos y fechas.")
        return dict(reference, predictions=predictions)

    def _score(self, forecast, target, observed):
        if not observed.any():
            return {"mse": None, "mae": None, "rmse": None}
        error = forecast[observed].astype(np.float64)-target[observed]
        mse = float(np.mean(error**2))
        return {"mse": mse, "mae": float(np.mean(np.abs(error))), "rmse": math.sqrt(mse)}

    def series(self, group, adjacency, sensor_id, time_range="all", horizon=1):
        if horizon != 1:
            raise ValueError("Esta batería solo predice un horizonte.")
        self._check_group(group)
        if adjacency != "both" and adjacency not in self.adjacencies:
            raise ValueError("Grafo desconocido.")
        if time_range not in ("all", "48h", "24h"):
            raise ValueError("Periodo de prueba desconocido.")
        members = {row["id"]: row for row in self.sensors(group)}
        if sensor_id not in members:
            raise SensorNotFound("Este sensor no forma parte de la agrupación seleccionada.")
        selected = members[sensor_id]
        graph_names = self.adjacencies if adjacency == "both" else (adjacency,)
        cases = {name: self._signals(group, name) for name in graph_names}
        reference = cases[graph_names[0]]
        index = selected["node"]
        if any(index < 0 or index >= case["target"].shape[1] or int(case["node_labels"][index]) != selected["node_label"] for case in cases.values()):
            raise DataMismatch("El sensor y las columnas de las predicciones no coinciden.")
        for case in cases.values():
            if any(not np.array_equal(case[key], reference[key]) for key in ("target", "observed", "times", "origin_times", "node_labels")):
                raise DataMismatch("Los grafos no comparten exactamente los mismos objetivos y fechas.")
        hours = {"all": None, "48h": 48, "24h": 24}[time_range]
        start = int(np.searchsorted(reference["times"], reference["times"][-1]-np.timedelta64(hours, "h"))) if hours else 0
        target, observed = reference["target"][start:, index], reference["observed"][start:, index]
        scores = []
        curves = {}
        for name, case in cases.items():
            for model in self.models:
                key = model if adjacency != "both" else f"{name}_{model}"
                prediction = case["predictions"][model][start:, index]
                score = self._score(prediction, target, observed)
                scores.append(dict(id=key, model=model, graph=name, label=f"{MODELS[model]} · {GRAPHS[name]}", **score))
                curves[key] = prediction.astype(float).tolist()
        persistence = reference["persistence"][start:, index]
        window_mean = reference["window_mean"][start:, index]
        valid_scores = [row for row in scores if row["mse"] is not None]
        validation = min((self.results[group, name, model] for name in graph_names for model in self.models),
                         key=lambda row: row["valid_mse"])
        period = np.datetime_as_string(reference["times"][start:], unit="s").tolist()
        ahead = float((reference["times"][start]-reference["origin_times"][start])/np.timedelta64(1, "m"))
        return {
            "group": group, "adjacency": adjacency, "sensor": selected, "range": time_range,
            "aggregated": group != "base", "points": len(period), "total_points": len(reference["times"]),
            "observed_count": int(observed.sum()), "ahead_minutes": ahead, "time": period,
            "origin_time": np.datetime_as_string(reference["origin_times"][start:], unit="s").tolist(),
            "target": [float(value) if available else None for value, available in zip(target, observed)],
            "prediction": curves,
            "persistence": persistence.astype(float).tolist(), "window_mean": window_mean.astype(float).tolist(),
            "scores": scores, "reference_scores": {"persistence": self._score(persistence, target, observed),
                                                    "window_mean": self._score(window_mean, target, observed)},
            "validation_leader": {"model": validation["model"], "graph": validation["adjacency"],
                                  "label": f"{MODELS[validation['model']]} · {GRAPHS[validation['adjacency']]}",
                                  "mse": validation["valid_mse"]},
            "test_leader": min(valid_scores, key=lambda row: row["mse"])["id"] if valid_scores else None,
            "warning": "Las métricas por sensor se calculan solo donde hay medida real. El mejor en prueba es descriptivo: no se usa para elegir el modelo durante el entrenamiento.",
        }


class SearchCatalog(ResultsCatalog):
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 2:
            raise ValueError("Formato de búsqueda desconocido.")
        self.ready = summary.get("complete", False) and manifest.get("state") == "completed"
        self.mode = manifest.get("mode", "staged")
        self.groups = tuple(group for group in GROUPS if f"{group}/closeness" in manifest["datasets"])
        self.adjacencies = ("closeness", "correlation")
        self.models = tuple(MODELS)
        self.datasets = manifest["datasets"]
        self.curves_root = self.root / "curves_precomputed_v1"
        self.results = {}
        self.candidates = {}
        self.final_candidates = {}
        for row in summary["results"]:
            if row["status"] != "completed":
                continue
            if row["kind"] == "candidate":
                if row["id"] in self.candidates:
                    raise ValueError("Hay configuraciones duplicadas en la búsqueda.")
                self.candidates[row["id"]] = row
                continue
            if row["kind"] != "final":
                continue
            folder = Path(row["result_dir"]).resolve(strict=True)
            if not folder.is_relative_to(self.root) or not (folder / "predictions.npz").is_file():
                raise ValueError("Falta una predicción final dentro de la búsqueda seleccionada.")
            self.results.setdefault((row["cluster"], row["adjacency"]), []).append(dict(row, result_dir=folder))
            if row.get("source_id"):
                self.final_candidates[row["source_id"]] = folder
        for group in self.groups:
            a, b = (self.datasets[f"{group}/{graph}"] for graph in self.adjacencies)
            if a["horizon"] != 4 or b["horizon"] != 4 or a["signal_signature"] != b["signal_signature"]:
                raise DataMismatch("Los grafos no comparten datos de cuatro horizontes.")

    def metadata(self):
        source = Path(self.datasets[f"{self.groups[0]}/closeness"]["path"])
        with np.load(source / "series.npz", allow_pickle=False) as data:
            times = data["times"]
            starts = data["test_starts"]
            period = {"start": np.datetime_as_string(times[starts[0]], unit="s"),
                      "end": np.datetime_as_string(times[starts[-1]+3], unit="s")}
        return {"groups": [{"key": group, "label": GROUPS[group], "nodes": self.datasets[f"{group}/closeness"]["n_nodes"],
                            "sensors": self.datasets[f"{group}/closeness"]["n_sensors"]} for group in self.groups],
                "graphs": [{"key": name, "label": GRAPHS[name]} for name in (*self.adjacencies, "both")],
                "models": [{"key": name, "label": MODELS[name]} for name in self.models],
                "horizons": 4, "ready": self.ready, "period": period,
                "configurations": len(self.candidates), "precompute": self.precompute_status(),
                "note": f"Búsqueda {'exhaustiva' if self.mode == 'exhaustive' else 'por etapas'} de cuatro horizontes; ranking por validación y curvas de los modelos disponibles."}

    def precompute_status(self):
        path = self.curves_root / "manifest.json"
        if not path.is_file():
            return {"completed": len(self.final_candidates), "total": len(self.candidates), "state": "not_started"}
        status = json.loads(path.read_text(encoding="utf-8"))
        return {"completed": len(set(status["completed"]) | self.final_candidates.keys()),
                "total": len(self.candidates), "state": status["state"], "failed": len(status.get("failed", {}))}

    def _computed_metrics(self, job_id):
        path = self.curves_root / "metrics" / f"{job_id}.json"
        if not path.is_file() or not (self.curves_root / "predictions" / f"{job_id}.npy").is_file():
            return None
        return self._load_computed_metrics(job_id)

    @lru_cache(maxsize=4320)
    def _load_computed_metrics(self, job_id):
        path = self.curves_root / "metrics" / f"{job_id}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        row = self.candidates[job_id]
        dataset = self.datasets[f"{row['cluster']}/{row['adjacency']}"]
        if result["job_id"] != job_id or result["dataset_signature"] != dataset["signature"]:
            raise DataMismatch("Las métricas precalculadas no corresponden al dataset.")
        return result["test"]

    def _test_metrics(self, row):
        result = self._computed_metrics(row["id"])
        if result is not None:
            return result
        final = self.final_candidates.get(row["id"])
        if final is None:
            return None
        return json.loads((final / "metrics.json").read_text(encoding="utf-8"))["test"]

    def rankings(self, group, graph, filters=None, offset=0, limit=25, phase="test", horizon="all"):
        self._check_group(group)
        if graph != "both" and graph not in self.adjacencies:
            raise ValueError("Grafo desconocido.")
        if offset < 0 or limit < 1 or limit > 100 or phase not in ("test", "validation") or horizon not in ("all", "1", "2", "3", "4"):
            raise ValueError("Paginación desconocida.")
        filters = filters or {}
        permitted = {"architecture": ("pfm", "upstream_bidirectional"), "model": tuple(MODELS),
                     "time_alpha": ("0", "0.2", "0.4", "0.6", "0.8", "1"),
                     "layers": ("1", "2", "3"), "hidden_feats": ("32", "64", "128")}
        if any(key not in permitted or value not in permitted[key] for key, value in filters.items()):
            raise ValueError("Filtro de configuraciones desconocido.")
        jobs = [row for row in self.candidates.values() if row["cluster"] == group and
                (graph == "both" or row["adjacency"] == graph) and
                all((float(row[key]) == float(value)) if key == "time_alpha" else str(row[key]) == value
                    for key, value in filters.items())]

        def value(row, metric, split):
            if split == "validation":
                result = row["validation"]
                return result[metric] if horizon == "all" else result["per_horizon"][int(horizon) - 1][metric]
            result = self._test_metrics(row)
            if result is None:
                return None
            score = result["model"] if horizon == "all" else result["model"]["per_horizon"][int(horizon) - 1]
            return score[metric]

        def entries(metric):
            available = [row for row in jobs if value(row, metric, phase) is not None]
            ordered = sorted(available, key=lambda row: (value(row, metric, phase), row["layers"], row["hidden_feats"], row["id"]))
            return [{"id": row["id"], "graph": row["adjacency"], "architecture": row["architecture"],
                     "model": row["model"], "time_alpha": row["time_alpha"], "layers": row["layers"],
                     "hidden_feats": row["hidden_feats"], "mse": value(row, "mse", phase),
                     "mae": value(row, "mae", phase),
                     "curves_ready": row["id"] in self.final_candidates or
                     (self.curves_root / "predictions" / f"{row['id']}.npy").is_file()}
                    for row in ordered[offset:offset+limit]]
        evaluated = sum(self._test_metrics(row) is not None for row in jobs) if phase == "test" else len(jobs)
        return {"group": group, "graph": graph, "total": len(jobs), "evaluated": evaluated, "phase": phase,
                "horizon": horizon, "offset": offset, "limit": limit,
                "mse": entries("mse"), "mae": entries("mae"), "precompute": self.precompute_status()}

    def comparison(self, group, graph="both", architecture="pfm", horizon="all", criterion="mse"):
        self._check_group(group)
        if graph != "both" and graph not in self.adjacencies:
            raise ValueError("Grafo desconocido.")
        if architecture not in ("pfm", "upstream_bidirectional") or horizon not in ("all", "1", "2", "3", "4") or criterion not in ("mse", "mae"):
            raise ValueError("Comparación desconocida.")
        rows = []
        for adjacency in self.adjacencies if graph == "both" else (graph,):
            for model in MODELS:
                options = [row for row in self.candidates.values() if row["cluster"] == group and
                           row["adjacency"] == adjacency and row["architecture"] == architecture and row["model"] == model]
                if not options:
                    rows.append({"id": None, "graph": adjacency, "model": model,
                                 "time_alpha": None, "layers": None, "hidden_feats": None, "ratio": None})
                    continue
                chosen = min(options, key=lambda row: (row["validation"][criterion], row["layers"], row["hidden_feats"], row["id"]))
                test = self._test_metrics(chosen)
                value = None
                if test:
                    model_score = test["model"] if horizon == "all" else test["model"]["per_horizon"][int(horizon) - 1]
                    baseline = test["persistence"] if horizon == "all" else test["persistence"]["per_horizon"][int(horizon) - 1]
                    if model_score["mse"] is not None and baseline["mse"]:
                        value = model_score["mse"] / baseline["mse"]
                rows.append({"id": chosen["id"], "graph": adjacency, "model": model,
                             "time_alpha": chosen["time_alpha"], "layers": chosen["layers"],
                             "hidden_feats": chosen["hidden_feats"], "ratio": value})
        return {"group": group, "graph": graph, "architecture": architecture, "criterion": criterion,
                "horizon": horizon, "rows": rows, "description": "MSE en prueba / MSE de persistencia; menor que 1 = mejora. Configuraciones elegidas por validación."}

    @lru_cache(maxsize=8)
    def _signals(self, group, adjacency, horizon):
        self._check_group(group)
        if adjacency not in self.adjacencies or horizon not in (1, 2, 3, 4):
            raise ValueError("Grafo u horizonte desconocido.")
        rows = self.results.get((group, adjacency), ())
        if not rows:
            raise ValueError("Los entrenamientos finales de este grafo aún no han terminado.")
        dataset = self.datasets[f"{group}/{adjacency}"]
        predictions = {}
        reference = None
        for row in rows:
            with np.load(row["result_dir"] / "predictions.npz", allow_pickle=False) as data:
                forecast = data["model"]
                if forecast.ndim != 4 or forecast.shape[1:] != (4, dataset["n_nodes"], 1):
                    raise DataMismatch("La predicción no corresponde a los cuatro horizontes y nodos esperados.")
                predictions[row["id"]] = forecast[:, horizon-1, :, 0]
                current = {"target": data["target"][:, horizon-1, :, 0], "observed": data["observed"][:, horizon-1, :, 0],
                           "times": data["target_times"][:, horizon-1], "origin_times": data["origin_times"],
                           "node_labels": data["node_labels"], "persistence": data["persistence"][:, horizon-1, :, 0],
                           "window_mean": data["window_mean"][:, horizon-1, :, 0]}
                if reference is None:
                    reference = current
                elif any(not np.array_equal(current[key], reference[key]) for key in current):
                    raise DataMismatch("Los finalistas no comparten objetivos, máscaras, fechas y referencias.")
        return dict(reference, predictions=predictions, rows=rows)

    def series(self, group, adjacency, sensor_id, time_range="all", horizon=1, configurations=None):
        self._check_group(group)
        if adjacency != "both" and adjacency not in self.adjacencies:
            raise ValueError("Grafo desconocido.")
        if time_range not in ("all", "48h", "24h") or horizon not in (1, 2, 3, 4):
            raise ValueError("Periodo u horizonte desconocido.")
        if configurations is not None:
            if not 1 <= len(configurations) <= 4 or len(set(configurations)) != len(configurations):
                raise ValueError("Selecciona entre una y cuatro configuraciones distintas.")
            if any(key not in self.candidates or self.candidates[key]["cluster"] != group or
                   (adjacency != "both" and self.candidates[key]["adjacency"] != adjacency) for key in configurations):
                raise ValueError("Configuración desconocida o incompatible con la agrupación y el grafo.")
        members = {row["id"]: row for row in self.sensors(group)}
        if sensor_id not in members:
            raise SensorNotFound("Este sensor no forma parte de la agrupación seleccionada.")
        selected = members[sensor_id]
        graph_names = self.adjacencies if adjacency == "both" else (adjacency,)
        cases = [self._signals(group, graph, horizon) for graph in graph_names]
        reference = cases[0]
        index = selected["node"]
        for case in cases:
            if index < 0 or index >= case["target"].shape[1] or int(case["node_labels"][index]) != selected["node_label"]:
                raise DataMismatch("El sensor y las columnas no coinciden.")
            if any(not np.array_equal(case[key], reference[key]) for key in ("target", "observed", "times", "origin_times", "node_labels")):
                raise DataMismatch("Los grafos no comparten objetivos y fechas.")
        hours = {"all": None, "48h": 48, "24h": 24}[time_range]
        start = int(np.searchsorted(reference["times"], reference["times"][-1]-np.timedelta64(hours, "h"))) if hours else 0
        target, observed = reference["target"][start:, index], reference["observed"][start:, index]
        scores, curves = [], {}
        default_ids = {row["id"] for row in sorted((row for case in cases for row in case["rows"]),
                       key=lambda row: (row["validation"]["mse"], row["id"]))[:4]} if configurations is None else set()
        for graph, case in zip(graph_names, cases):
            rows = [row for row in case["rows"] if row["id"] in default_ids] if configurations is None else [self.candidates[key] for key in configurations
                                                                 if self.candidates[key]["adjacency"] == graph]
            for row in rows:
                if configurations is None:
                    prediction = case["predictions"][row["id"]][start:, index]
                else:
                    filename = self.curves_root / "predictions" / f"{row['id']}.npy"
                    if filename.is_file():
                        saved = np.load(filename, mmap_mode="r", allow_pickle=False)
                        if saved.shape != (len(reference["times"]), 4, reference["target"].shape[1]) or saved.dtype != np.float32:
                            raise DataMismatch("La curva precalculada no coincide con el dataset.")
                        prediction = saved[start:, horizon-1, index]
                    elif row["id"] in self.final_candidates:
                        prediction = case["predictions"][f"final__{row['id']}"][start:, index]
                    else:
                        raise DataMismatch("Esta configuración aún no tiene curva calculada. Consulta el progreso y vuelve a intentarlo.")
                if not np.isfinite(prediction).all():
                    raise DataMismatch("La curva contiene valores no finitos.")
                architecture = "PFM" if row["architecture"] == "pfm" else "GitHub bidireccional"
                detail = "" if configurations is None else f" · α={row['time_alpha']:g} · {row['layers']} capas · {row['hidden_feats']} dim"
                scores.append(dict(id=row["id"], model=row["model"], graph=graph,
                                   architecture=row["architecture"], label=f"{MODELS[row['model']]} · {architecture} · {GRAPHS[graph]}{detail}",
                                   validation_mse=row["validation"]["mse"], validation_mae=row["validation"]["mae"],
                                   **self._score(prediction, target, observed)))
                curves[row["id"]] = prediction.astype(float).tolist()
        persistence = reference["persistence"][start:, index]
        window_mean = reference["window_mean"][start:, index]
        validation = min(scores, key=lambda row: (row["validation_mse"], row["id"]))
        valid_scores = [row for row in scores if row["mse"] is not None]
        return {"group": group, "adjacency": adjacency, "sensor": selected, "range": time_range, "horizon": horizon,
                "aggregated": group != "base", "points": len(target), "total_points": len(reference["times"]),
                "observed_count": int(observed.sum()),
                "ahead_minutes": float((reference["times"][start]-reference["origin_times"][start])/np.timedelta64(1, "m")),
                "time": np.datetime_as_string(reference["times"][start:], unit="s").tolist(),
                "origin_time": np.datetime_as_string(reference["origin_times"][start:], unit="s").tolist(),
                "target": [float(value) if available else None for value, available in zip(target, observed)],
                "prediction": curves, "persistence": persistence.astype(float).tolist(), "window_mean": window_mean.astype(float).tolist(),
                "scores": scores, "reference_scores": {"persistence": self._score(persistence, target, observed),
                                                        "window_mean": self._score(window_mean, target, observed)},
                "validation_leader": {"model": validation["model"], "graph": validation["graph"],
                                      "label": validation["label"], "mse": validation["validation_mse"]},
                "test_leader": min(valid_scores, key=lambda row: row["mse"])["id"] if valid_scores else None,
                "warning": "El orden de prueba por sensor es descriptivo: no sustituye la selección hecha por validación."}


def handler_for(catalog):
    class DashboardHandler(BaseHTTPRequestHandler):
        def respond(self, status, payload, content_type="application/json; charset=utf-8", *, body=True):
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") if not isinstance(payload, bytes) else payload
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'")
            self.end_headers()
            if body:
                self.wfile.write(raw)

        def dispatch(self, body=True):
            request = urlsplit(self.path)
            if request.path in ASSETS:
                filename, kind = ASSETS[request.path]
                self.respond(200, (Path(__file__).parent / filename).read_bytes(), kind, body=body)
                return
            params = parse_qs(request.query)

            def required(key):
                values = params.get(key, [])
                if len(values) != 1:
                    raise ValueError(f"Falta el parámetro {key} o está repetido.")
                return values[0]

            try:
                if request.path == "/api/meta":
                    result = catalog.metadata()
                elif request.path == "/api/sensors":
                    group = required("group")
                    result = {"group": group, "sensors": catalog.sensors(group)}
                elif request.path == "/api/progress" and isinstance(catalog, SearchCatalog):
                    result = catalog.precompute_status()
                elif request.path == "/api/rankings" and isinstance(catalog, SearchCatalog):
                    group, graph = required("group"), required("graph")
                    optional = ("architecture", "model", "time_alpha", "layers", "hidden_feats")
                    if any(len(params.get(key, [])) > 1 for key in (*optional, "offset", "limit", "phase", "horizon")):
                        raise ValueError("Filtro repetido.")
                    filters = {key: params[key][0] for key in optional if key in params}
                    result = catalog.rankings(group, graph, filters, int(params.get("offset", ["0"])[0]),
                                              int(params.get("limit", ["25"])[0]), params.get("phase", ["test"])[0],
                                              params.get("horizon", ["all"])[0])
                elif request.path == "/api/comparison" and isinstance(catalog, SearchCatalog):
                    group = required("group")
                    optional = ("graph", "architecture", "horizon", "criterion")
                    if any(len(params.get(key, [])) > 1 for key in optional):
                        raise ValueError("Filtro repetido.")
                    result = catalog.comparison(group, *(params.get(key, [default])[0] for key, default in
                                                        (("graph", "both"), ("architecture", "pfm"),
                                                         ("horizon", "all"), ("criterion", "mse"))))
                elif request.path == "/api/series":
                    group, graph, sensor = required("group"), required("graph"), required("sensor")
                    if not sensor.isascii() or not sensor.isdecimal() or len(sensor) > 20:
                        raise ValueError("El identificador del sensor debe ser numérico.")
                    ranges = params.get("range", ["all"])
                    horizons = params.get("horizon", ["1"])
                    if len(ranges) != 1 or len(horizons) != 1 or horizons[0] not in ("1", "2", "3", "4"):
                        raise ValueError("Periodo u horizonte desconocido o repetido.")
                    choices = params.get("configs", [])
                    if len(choices) > 1:
                        raise ValueError("Selección de modelos repetida.")
                    if choices and not isinstance(catalog, SearchCatalog):
                        raise ValueError("Esta batería no admite elegir configuraciones individuales.")
                    selected = choices[0].split(",") if choices else None
                    result = catalog.series(group, graph, int(sensor), ranges[0], int(horizons[0]), selected) if selected else catalog.series(group, graph, int(sensor), ranges[0], int(horizons[0]))
                else:
                    self.respond(404, {"error": "Ruta no encontrada."}, body=body)
                    return
                self.respond(200, result, body=body)
            except SensorNotFound as error:
                self.respond(404, {"error": str(error)}, body=body)
            except DataMismatch as error:
                self.respond(409, {"error": str(error)}, body=body)
            except ValueError as error:
                self.respond(400, {"error": str(error)}, body=body)
            except Exception as error:
                print(f"Error interno del panel: {type(error).__name__}: {error}", flush=True)
                self.respond(500, {"error": "No se pudieron leer los resultados guardados."}, body=body)

        def do_GET(self):
            self.dispatch()

        def do_HEAD(self):
            self.dispatch(body=False)

    return DashboardHandler


def main():
    parser = argparse.ArgumentParser(description="Comparador local de modelos del PFM, sin entrenar ni modificar resultados.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 0 y 65535.")
    manifest = json.loads((args.results_dir / "manifest.json").read_text(encoding="utf-8"))
    catalog = SearchCatalog(args.results_dir) if manifest.get("format_version") == 2 else ResultsCatalog(args.results_dir)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(catalog)) as server:
        server.daemon_threads = True
        print(f"Comparador local: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Resultados: {catalog.root} | grupos: {len(catalog.groups)} | modelos: {len(catalog.models)}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
