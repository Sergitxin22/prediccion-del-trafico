import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CLUSTERINGS = ("base", "proximity_50", "proximity_500", "proximity_behaviour_50", "proximity_behaviour_500")
ADJACENCIES = ("closeness", "correlation")
MODELS = ("gcn_lstm", "gcn_gru", "gat_lstm", "gat_gru")
DEFAULTS = {"epochs": 25, "batch_size": 32, "micro_batch_size": 4, "hidden_feats": 64,
            "temporal_hidden": 64, "layers": 2, "learning_rate": 1e-4, "time_decay": .05, "seed": 42}


def now():
    return datetime.now(timezone.utc).isoformat()


def interrupt_handler(signum, frame):
    raise KeyboardInterrupt


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def fingerprint_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024*1024), b""):
                digest.update(block)
    return digest.hexdigest()


def environment_versions():
    return {name: version(name) for name in ("jax", "jaxlib", "flax", "jraph", "optax", "numpy", "pandas", "scikit-learn", "matplotlib")}


def training_code_signature():
    paths = [PROJECT / "scripts/train.py", PROJECT / "requirements.txt"]
    paths += [PROJECT / f"src/traffic_flow_prediction/{name}.py" for name in ("models", "engine", "datasets")]
    return fingerprint_files(paths)


def dataset_signature(path):
    return fingerprint_files([path / name for name in ("series.npz", "graph.npz", "metadata.json")])


def signal_signature(path):
    digest = hashlib.sha256()
    with np.load(path / "series.npz", allow_pickle=False) as data:
        for key in sorted(data.files):
            value = data[key]
            digest.update(str((key, value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def inspect_datasets(source, clusterings, adjacencies):
    datasets = {}
    for cluster in clusterings:
        signatures = []
        for adjacency in adjacencies:
            path = (source / cluster / adjacency).resolve()
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("format_version") != 1 or metadata.get("index") != "pfm":
                raise ValueError(f"Incompatible dataset: {path}")
            if metadata["cluster"] != cluster or metadata["adjacency"] != adjacency:
                raise ValueError(f"Dataset labels do not match its path: {path}")
            signals = signal_signature(path)
            signatures.append(signals)
            datasets[f"{cluster}/{adjacency}"] = {"path": str(path), "signature": dataset_signature(path),
                                                   "signal_signature": signals, "n_nodes": metadata["n_nodes"],
                                                   "n_sensors": metadata.get("n_sensors"), "horizon": metadata["horizon"]}
        if len(set(signatures)) != 1:
            raise ValueError(f"Unmatched signals, masks or splits for {cluster}. Generate both adjacencies together with the current generator.")
    return datasets


def plan_jobs(clusterings, adjacencies, models):
    return [{"id": f"{cluster}__{adjacency}__{model}", "cluster": cluster, "adjacency": adjacency,
             "model": model, "dataset": f"{cluster}/{adjacency}", "status": "pending", "attempts": []}
            for cluster, adjacency, model in itertools.product(dict.fromkeys(clusterings), dict.fromkeys(adjacencies), dict.fromkeys(models))]


def training_command(job, dataset, settings, target):
    spatial, temporal = job["model"].split("_")
    command = [sys.executable, "-B", str(PROJECT / "scripts/train.py"), "--source", dataset["path"],
               "--target", str(target), "--spatial", spatial, "--temporal", temporal, "--device", "gpu"]
    for key, value in settings.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def tail(path, size=32768):
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size-size))
        return file.read().decode("utf-8", errors="replace")


def validate_result(attempt, job, dataset, settings):
    outputs = list(Path(attempt["folder"]).glob("output/exp_*/metrics.json"))
    if len(outputs) != 1:
        raise ValueError("Expected exactly one completed experiment.")
    folder = outputs[0].parent
    required = ["config.json", "model.npz", "predictions.npz", "comparison.csv", "selected_predictions.csv"]
    required += [f"node_{node}_h{step}.png" for node in range(min(3, dataset["n_nodes"])) for step in range(1, dataset["horizon"]+1)]
    if any(not (folder / name).is_file() or (folder / name).stat().st_size == 0 for name in required):
        raise ValueError("Missing or empty experiment artifacts.")
    if f"Modelo, configuracion, metricas, predicciones y graficas: {folder}" not in tail(Path(attempt["log"])):
        raise ValueError("The final completion message is missing from the log.")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads(outputs[0].read_text(encoding="utf-8"))
    spatial, temporal = job["model"].split("_")
    if config["spatial"] != spatial or config["temporal"] != temporal or config["dataset_signature"] != dataset["signature"]:
        raise ValueError("Experiment configuration differs from the planned task.")
    if any(config[key] != value for key, value in settings.items()) or len(metrics["history"]) != settings["epochs"]:
        raise ValueError("Training settings or epoch count differ from the plan.")
    for split in ("validation", "test"):
        scores = metrics[split]
        if scores["model"]["count"] <= 0 or scores["model"]["count"] != scores["persistence"]["count"]:
            raise ValueError("Model and baseline observation counts differ.")
        for score in scores.values():
            if any(not math.isfinite(score[key]) for key in ("mse", "rmse", "mae")):
                raise ValueError("Nonfinite evaluation metric.")
    return str(folder), metrics


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    return {int(row[0].strip()): {"uuid": row[1].strip(), "name": row[2].strip()} for row in csv.reader(result.stdout.splitlines())}


def busy_gpus(inventory):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    uuids = {row[0].strip() for row in csv.reader(result.stdout.splitlines()) if row}
    return {gpu for gpu, info in inventory.items() if info["uuid"] in uuids}


def pid_exists(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def result_row(job, dataset, metrics):
    valid, test = metrics["validation"], metrics["test"]
    baseline = test["persistence"]["mse"]
    return {"id": job["id"], "cluster": job["cluster"], "adjacency": job["adjacency"], "model": job["model"],
            "status": job["status"], "nodes": dataset["n_nodes"], "sensors": dataset.get("n_sensors"), "best_epoch": metrics["best_epoch"],
            "valid_mse": valid["model"]["mse"], "valid_mae": valid["model"]["mae"],
            "valid_persistence_mse": valid["persistence"]["mse"], "test_mse": test["model"]["mse"],
            "test_rmse": test["model"]["rmse"], "test_mae": test["model"]["mae"],
            "test_persistence_mse": baseline, "test_persistence_mae": test["persistence"]["mae"],
            "test_mse_skill": 1-test["model"]["mse"]/baseline if baseline else None,
            "observations": test["model"]["count"], "constant_nodes": len(metrics["constant_prediction_node_indices"]),
            "result_dir": job["attempts"][-1]["result_dir"], "gpu": job["attempts"][-1]["gpu"]}


def summarize(root, manifest):
    rows = []
    for job in manifest["jobs"]:
        if job["status"] == "completed":
            folder = Path(job["attempts"][-1]["result_dir"])
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            rows.append(result_row(job, manifest["datasets"][job["dataset"]], metrics))
        else:
            rows.append({key: job[key] for key in ("id", "cluster", "adjacency", "model", "status")})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = root / f".results.{uuid4().hex}.csv"
    with temporary.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "results.csv")
    expected = len(manifest["adjacencies"])*len(manifest["models"])
    best = []
    for cluster in manifest["clusterings"]:
        completed = [row for row in rows if row["cluster"] == cluster and row["status"] == "completed"]
        if len(completed) == expected:
            best.append(min(completed, key=lambda row: row["valid_mse"]))
    atomic_json(root / "summary.json", {"results": rows, "best_by_validation_within_each_clustering": best,
                                        "complete": all(job["status"] == "completed" for job in manifest["jobs"]),
                                        "warning": "Do not rank absolute errors across different clusterings: the targets have different aggregation levels. GAT-GRU is an additional model beyond the three in the PFM."})
    return rows


def plot_summary(root, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))["results"]
    for cluster in manifest["clusterings"]:
        selected = [row for row in rows if row["cluster"] == cluster and row["status"] == "completed"]
        if not selected:
            continue
        for split in ("valid", "test"):
            ratios = [row[f"{split}_mse"]/row[f"{split}_persistence_mse"] if row[f"{split}_persistence_mse"] else np.nan for row in selected]
            labels = [f"{row['adjacency']}\n{row['model'].upper().replace('_', '-')}" for row in selected]
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.bar(np.arange(len(selected)), ratios, color=["#4477aa" if row["adjacency"] == "closeness" else "#ee7733" for row in selected])
            ax.axhline(1, color="black", linestyle="--")
            ax.set_xticks(np.arange(len(selected)), labels, rotation=30, ha="right")
            ax.set_ylabel("MSE del modelo / MSE de persistencia")
            phase = "validacion" if split == "valid" else "prueba"
            ax.set_title(f"{cluster} | {phase} | menor que 1 = mejor que persistencia")
            ax.legend(handles=[Patch(color="#4477aa", label="Cercania"), Patch(color="#ee7733", label="Correlacion")])
            fig.tight_layout()
            fig.savefig(root / f"comparison_{cluster}_{split}.png", dpi=130)
            plt.close(fig)


def recover_jobs(manifest, retry_failed):
    for job in manifest["jobs"]:
        if job["status"] in ("running", "launching"):
            attempt = job["attempts"][-1]
            try:
                folder, _ = validate_result(attempt, job, manifest["datasets"][job["dataset"]], manifest["training"])
                attempt.update(status="completed", result_dir=folder, recovered=True)
                job["status"] = "completed"
            except (ValueError, OSError, KeyError):
                if pid_exists(attempt.get("pid")) or not attempt.get("pid"):
                    raise RuntimeError(f"Cannot resume {job['id']} while its previous launch may still be active. Check its log and process first.")
                attempt.update(status="interrupted", ended=now())
                job["status"] = "pending"
        elif job["status"] == "failed" and retry_failed:
            job["status"] = "pending"


def run_queue(root, manifest, gpus):
    inventory = gpu_inventory()
    if not set(gpus).issubset(inventory):
        raise ValueError("A requested GPU does not exist.")
    manifest["gpu_inventory"] = inventory
    manifest["state"] = "running"
    running = {}
    last_report = 0.
    atomic_json(root / "manifest.json", manifest)
    try:
        while any(job["status"] == "pending" for job in manifest["jobs"]) or running:
            for gpu, (process, job, attempt, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                attempt.update(exit_code=code, ended=now())
                try:
                    if code != 0:
                        raise ValueError(f"Training exited with code {code}.")
                    folder, _ = validate_result(attempt, job, manifest["datasets"][job["dataset"]], manifest["training"])
                    attempt.update(status="completed", result_dir=folder)
                    job["status"] = "completed"
                    print(f"COMPLETED {job['id']} | GPU {gpu}", flush=True)
                except (ValueError, OSError, KeyError) as error:
                    attempt.update(status="failed", error=str(error))
                    job["status"] = "failed"
                    print(f"FAILED {job['id']}: {error} | log={attempt['log']}", flush=True)
                del running[gpu]
                atomic_json(root / "manifest.json", manifest)
                summarize(root, manifest)
            pending = [job for job in manifest["jobs"] if job["status"] == "pending"]
            occupied = busy_gpus(inventory) if pending else set()
            for gpu in gpus:
                if gpu in running or gpu in occupied or not pending:
                    continue
                if training_code_signature() != manifest["code_signature"] or environment_versions() != manifest["environment"]:
                    raise RuntimeError("Training code or dependencies changed during the experiment matrix.")
                job = pending.pop(0)
                dataset = manifest["datasets"][job["dataset"]]
                if dataset_signature(Path(dataset["path"])) != dataset["signature"]:
                    raise RuntimeError(f"Dataset changed: {dataset['path']}")
                folder = root / "jobs" / job["id"] / f"attempt_{len(job['attempts'])+1:02d}_{uuid4().hex[:8]}"
                folder.mkdir(parents=True, exist_ok=False)
                command = training_command(job, dataset, manifest["training"], folder / "output")
                attempt = {"folder": str(folder), "log": str(folder / "train.log"), "gpu": gpu, "command": command,
                           "started": now(), "status": "launching"}
                job["attempts"].append(attempt)
                job["status"] = "launching"
                atomic_json(root / "manifest.json", manifest)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                           MPLBACKEND="Agg", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                log = Path(attempt["log"]).open("x", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError:
                    log.close()
                    attempt.update(status="failed", ended=now(), error="Could not start training process.")
                    job["status"] = "failed"
                    atomic_json(root / "manifest.json", manifest)
                    raise
                attempt.update(pid=process.pid, status="running")
                job["status"] = "running"
                running[gpu] = (process, job, attempt, log)
                atomic_json(root / "manifest.json", manifest)
                print(f"STARTED {job['id']} | GPU {gpu} | log={attempt['log']}", flush=True)
            if time.monotonic()-last_report >= 60:
                counts = {state: sum(job["status"] == state for job in manifest["jobs"]) for state in ("completed", "running", "pending", "failed")}
                epochs = {job["id"]: max([int(n) for n in re.findall(r"Epoch\s+(\d+)", tail(Path(attempt["log"])))], default=0)
                          for _, job, attempt, _ in running.values()}
                manifest["running_epochs"] = epochs
                manifest["heartbeat"] = now()
                atomic_json(root / "manifest.json", manifest)
                print(f"PROGRESS {counts} | epochs={epochs}", flush=True)
                last_report = time.monotonic()
            if running or any(job["status"] == "pending" for job in manifest["jobs"]):
                time.sleep(5)
    finally:
        for process, job, attempt, log in running.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
            attempt.update(status="interrupted", ended=now())
            job["status"] = "pending"
        manifest["state"] = "completed" if all(job["status"] == "completed" for job in manifest["jobs"]) else "incomplete"
        manifest["updated"] = now()
        atomic_json(root / "manifest.json", manifest)
        summarize(root, manifest)


def main():
    parser = argparse.ArgumentParser(description="Ejecuta una matriz de experimentos independientes, uno por GPU.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path, default=Path("target/pfm_matrix"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--clusterings", nargs="+", choices=CLUSTERINGS)
    parser.add_argument("--adjacencies", nargs="+", choices=ADJACENCIES)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(default))
    args = parser.parse_args()
    environment = environment_versions()
    gpus = list(dict.fromkeys(args.gpus))
    if min(gpus) < 0:
        parser.error("GPU indices must be nonnegative.")
    if args.resume:
        root = args.resume.resolve()
        if not root.is_dir():
            parser.error("The resume directory does not exist.")
    else:
        if args.source is None:
            parser.error("--source is required for a new matrix.")
        root = (args.target / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}").resolve()
        if not args.dry_run:
            root.mkdir(parents=True, exist_ok=False)
    lock = None
    try:
        if not args.dry_run:
            lock = (root / "runner.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("format_version") != 1 or manifest["code_signature"] != training_code_signature() or manifest["environment"] != environment:
                raise ValueError("Code or dependency versions changed; do not mix different protocols in a resumed matrix.")
            for key in DEFAULTS:
                if getattr(args, key) is not None and getattr(args, key) != manifest["training"][key]:
                    raise ValueError(f"Cannot change {key} when resuming.")
            for key in ("clusterings", "adjacencies", "models"):
                if getattr(args, key) is not None and getattr(args, key) != manifest[key]:
                    raise ValueError(f"Cannot change {key} when resuming.")
            if args.source is not None and args.source.resolve() != Path(manifest["source"]):
                raise ValueError("Cannot change the dataset root when resuming.")
            for dataset in manifest["datasets"].values():
                if dataset_signature(Path(dataset["path"])) != dataset["signature"]:
                    raise ValueError("A dataset changed since the original matrix.")
            if not args.dry_run:
                recover_jobs(manifest, args.retry_failed)
        else:
            settings = {key: getattr(args, key) if getattr(args, key) is not None else default for key, default in DEFAULTS.items()}
            if any(not math.isfinite(value) or value <= 0 for key, value in settings.items() if key not in ("seed", "time_decay")) or not math.isfinite(settings["time_decay"]) or settings["time_decay"] < 0:
                parser.error("Invalid training parameters.")
            clusterings = list(dict.fromkeys(args.clusterings or CLUSTERINGS))
            adjacencies = list(dict.fromkeys(args.adjacencies or ADJACENCIES))
            models = list(dict.fromkeys(args.models or MODELS))
            datasets = inspect_datasets(args.source, clusterings, adjacencies)
            manifest = {"format_version": 1, "created": now(), "source": str(args.source.resolve()), "training": settings,
                        "clusterings": clusterings, "adjacencies": adjacencies, "models": models, "datasets": datasets,
                        "code_signature": training_code_signature(), "environment": environment,
                        "jobs": plan_jobs(clusterings, adjacencies, models)}
        print(f"Matriz: {root} | experimentos={len(manifest['jobs'])} | GPUs={gpus}", flush=True)
        if args.dry_run:
            for job in manifest["jobs"]:
                print(job["id"], job["status"])
            return
        summarize(root, manifest)
        signal.signal(signal.SIGTERM, interrupt_handler)
        run_queue(root, manifest, gpus)
        plot_summary(root, manifest)
        print(f"Estado final: {manifest['state']} | resumen={root / 'results.csv'}", flush=True)
        if manifest["state"] != "completed":
            raise SystemExit(1)
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
