import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from run_experiments import atomic_json, busy_gpus, dataset_signature, environment_versions, gpu_inventory, pid_exists

PROJECT = Path(__file__).resolve().parents[1]


def training_signature():
    digest = hashlib.sha256()
    paths = [PROJECT / "scripts" / name for name in ("evaluate_all.py", "train.py")]
    paths += [PROJECT / "src/traffic_flow_prediction" / f"{name}.py" for name in ("models", "engine", "datasets")]
    for path in paths:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def candidates(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "completed" or not summary.get("complete"):
        raise ValueError("La búsqueda original debe estar completa.")
    jobs = {row["id"]: row for row in summary["results"] if row["kind"] == "candidate"}
    if len(jobs) != 4320 or any(row["status"] != "completed" for row in jobs.values()):
        raise ValueError("Faltan checkpoints completados de la búsqueda exhaustiva.")
    return manifest["datasets"], jobs


def score(square, absolute, count):
    n = int(np.sum(count))
    mse = float(np.sum(square) / n) if n else None
    return {"mse": mse, "rmse": math.sqrt(mse) if mse is not None else None,
            "mae": float(np.sum(absolute) / n) if n else None, "count": n,
            "per_horizon": [{"step": step + 1, "mse": float(np.sum(square[step]) / np.sum(count[step])) if np.sum(count[step]) else None,
                             "mae": float(np.sum(absolute[step]) / np.sum(count[step])) if np.sum(count[step]) else None,
                             "count": int(np.sum(count[step]))} for step in range(len(count))]}


def output_paths(folder, job_id):
    return folder / "predictions" / f"{job_id}.npy", folder / "metrics" / f"{job_id}.json", folder / "per_node" / f"{job_id}.npz"


def valid_output(folder, job, dataset):
    prediction, metrics_path, nodes_path = output_paths(folder, job["id"])
    if not all(path.is_file() and path.stat().st_size for path in (prediction, metrics_path, nodes_path)):
        return False
    try:
        with np.load(dataset["path"] + "/series.npz", allow_pickle=False) as source:
            expected = (len(source["test_starts"]), 4, dataset["n_nodes"])
        stored = np.load(prediction, mmap_mode="r", allow_pickle=False)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(nodes_path, allow_pickle=False) as nodes:
            return (stored.shape == expected and stored.dtype == np.float32 and
                    nodes["count"].shape == (4, dataset["n_nodes"]) and
                    metrics["test"]["model"]["count"] > 0 and metrics["dataset_signature"] == dataset["signature"])
    except (OSError, ValueError, KeyError):
        return False


def evaluate_one(root, folder, job, dataset, batch_size, device_kind="gpu"):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from traffic_flow_prediction.datasets import WindowDataLoader, load_checkpoint
    from traffic_flow_prediction.engine import evaluate
    from traffic_flow_prediction.models import GNN, SpatialType, TemporalType
    from train import load_data, prepare_data

    jax.config.update("jax_default_matmul_precision", "highest")
    device = jax.devices(device_kind)[0]
    source = Path(dataset["path"])
    experiment = Path(job["result_dir"]).resolve(strict=True)
    if not experiment.is_relative_to(root):
        raise ValueError("El checkpoint está fuera de la búsqueda.")
    config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    if config["dataset_signature"] != dataset["signature"] or Path(config["source"]) != source or any(
        config[key] != job[key] for key in ("architecture", "layers", "hidden_feats", "time_alpha")
    ):
        raise ValueError("El checkpoint no corresponde a esta configuración y dataset.")
    graph_data, data, _ = load_data(source)
    n, horizon, starts = dataset["n_nodes"], int(data["horizon"]), data["test_starts"]
    if horizon != 4 or data["values"].shape[1] != n:
        raise ValueError("Se esperan cuatro horizontes y el mismo orden de nodos.")
    path, metrics_path, nodes_path = output_paths(folder, job["id"])
    if valid_output(folder, job, dataset):
        return
    temporary = path.with_name(f".{job['id']}.{uuid4().hex}.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(starts), horizon, n))
    totals = {name: {"square": np.zeros((horizon, n), np.float64), "absolute": np.zeros((horizon, n), np.float64),
                     "count": np.zeros((horizon, n), np.int64)} for name in ("model", "persistence", "window_mean")}
    with jax.default_device(device):
        model = GNN(1, config["hidden_feats"], 1, int(data["window"]), horizon, config["layers"],
                    spatial=SpatialType[config["spatial"]], temporal=TemporalType[config["temporal"]],
                    temporal_hidden=config["temporal_hidden"], architecture=config["architecture"],
                    rngs=nnx.Rngs(config["seed"], params=config["seed"] + 1))
        load_checkpoint(experiment / "model.npz", model)
        for start in range(0, len(starts), batch_size):
            current = starts[start:start + batch_size]
            past = current[:, None] + np.arange(-int(data["window"]), 0)
            future = current[:, None] + np.arange(horizon)
            x = data["values"][past, :, None]
            target = data["values"][future, :, None]
            observed = data["observed"][future, :, None]
            graph, _ = prepare_data(x, target, graph_data)
            predicted = np.asarray(evaluate(model, graph)).reshape(len(current), n, horizon, 1).transpose(0, 2, 1, 3)
            if not np.isfinite(predicted).all():
                raise ValueError("Predicciones no finitas.")
            output[start:start + len(current)] = predicted[..., 0]
            alternatives = {"model": predicted, "persistence": np.repeat(x[:, -1:], horizon, axis=1),
                            "window_mean": np.repeat(x.mean(axis=1, keepdims=True), horizon, axis=1)}
            for name, values in alternatives.items():
                error = np.where(observed, values.astype(np.float64) - target, 0.)[..., 0]
                totals[name]["square"] += np.sum(error**2, axis=0)
                totals[name]["absolute"] += np.sum(np.abs(error), axis=0)
                totals[name]["count"] += np.sum(observed[..., 0], axis=0)
    output.flush()
    del output
    metrics = {"job_id": job["id"], "dataset_signature": dataset["signature"], "n_nodes": n,
               "test": {name: score(**value) for name, value in totals.items()}}
    if any(metrics["test"]["model"][name] is None for name in ("mse", "mae")):
        raise ValueError("No hay objetivos de prueba observados.")
    with nodes_path.open("wb") as file:
        np.savez_compressed(file, **{f"{name}_{key}": value for name, values in totals.items()
                                    for key, value in values.items()}, count=totals["model"]["count"])
    atomic_json(metrics_path, metrics)
    temporary.replace(path)
    print(f"Evaluado {job['id']} en {device.device_kind}: {len(starts)} ventanas, {n} nodos.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evalúa todos los checkpoints en prueba sin reentrenarlos ni generar imágenes.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.gpus or min(args.gpus) < 0:
        parser.error("Selecciona GPUs y un tamaño de lote positivos.")
    root = args.results_dir.resolve(strict=True)
    datasets, jobs = candidates(root)
    folder = root / "curves_precomputed_v1"
    if args.dry_run:
        print(f"Evaluaciones: {len(jobs)} | GPUs: {args.gpus} | salida nueva: {folder} | sin reentrenar ni generar imágenes.")
        return
    if args.worker:
        job = jobs.get(args.worker)
        if job is None:
            raise ValueError("Identificador de checkpoint desconocido.")
        evaluate_one(root, folder, job, datasets[f"{job['cluster']}/{job['adjacency']}"], args.batch_size)
        return
    if args.detach:
        folder.mkdir(exist_ok=True)
        log = folder / "coordinator.log"
        if log.exists():
            raise ValueError(f"Ya existe {log}; usa el proceso anterior o reanuda sin --detach.")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--results-dir", str(root),
                   "--batch-size", str(args.batch_size), "--gpus", *(str(g) for g in args.gpus)]
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Evaluación en segundo plano: PID {process.pid} | log: {log}")
        return
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    (folder / "metrics").mkdir(exist_ok=True)
    (folder / "per_node").mkdir(exist_ok=True)
    (folder / "logs").mkdir(exist_ok=True)
    lock_path = folder / "runner.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = folder / "manifest.json"
        signature = training_signature()
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["code_signature"] != signature or manifest["environment"] != environment_versions():
                raise ValueError("El código o las dependencias han cambiado: no mezclar evaluaciones.")
            if manifest["failed"] and not args.retry_failed:
                raise ValueError("Hay fallos registrados: revisa sus logs y reanuda con --retry-failed.")
            if args.retry_failed:
                manifest["failed"] = {}
        else:
            manifest = {"format_version": 1, "source": str(root), "code_signature": signature,
                        "environment": environment_versions(), "datasets": {key: value["signature"] for key, value in datasets.items()},
                        "completed": [], "failed": {}, "state": "running"}
        if any(dataset_signature(Path(value["path"])) != manifest["datasets"][key] for key, value in datasets.items()):
            raise ValueError("Un dataset ha cambiado desde el entrenamiento.")
        inventory = gpu_inventory()
        if not set(args.gpus).issubset(inventory):
            raise ValueError("GPU solicitada no disponible.")
        completed = set(manifest["completed"])
        if any(not valid_output(folder, jobs[key], datasets[f"{jobs[key]['cluster']}/{jobs[key]['adjacency']}"]) for key in completed):
            raise ValueError("Falta un resultado marcado como completado.")
        pending = [job for job in jobs.values() if job["id"] not in completed]
        active = {}
        manifest["state"] = "running"
        atomic_json(manifest_path, manifest)
        try:
            while pending or active:
                for gpu, (process, job, stream) in list(active.items()):
                    result = process.poll()
                    if result is None:
                        continue
                    stream.close()
                    del active[gpu]
                    if result or not valid_output(folder, job, datasets[f"{job['cluster']}/{job['adjacency']}"]):
                        manifest["failed"][job["id"]] = f"Salida {result}; ver logs/{job['id']}.log"
                        print(f"FALLO {job['id']} GPU {gpu} | ver log individual.", flush=True)
                    else:
                        completed.add(job["id"])
                        manifest["completed"].append(job["id"])
                        manifest["failed"].pop(job["id"], None)
                        print(f"COMPLETADO {len(completed)}/{len(jobs)} {job['id']} GPU {gpu}", flush=True)
                    atomic_json(manifest_path, manifest)
                if manifest["failed"]:
                    pending.clear()
                occupied = busy_gpus(inventory) if pending else set()
                for gpu in args.gpus:
                    if gpu in active or gpu in occupied or not pending:
                        continue
                    job = pending.pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda",
                               XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    stream = (folder / "logs" / f"{job['id']}.log").open("a", encoding="utf-8")
                    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--results-dir", str(root),
                               "--worker", job["id"], "--batch-size", str(args.batch_size)]
                    process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    active[gpu] = (process, job, stream)
                    print(f"INICIADO {job['id']} GPU {gpu}", flush=True)
                if pending or active:
                    time.sleep(5)
        finally:
            for process, _, stream in active.values():
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                stream.close()
            manifest["state"] = "completed" if len(completed) == len(jobs) else "incomplete"
            atomic_json(manifest_path, manifest)
        print(f"Estado {manifest['state']} | {len(completed)}/{len(jobs)} evaluados.", flush=True)
        if manifest["state"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT / "src"))
    main()
