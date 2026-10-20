import argparse
import csv
import fcntl
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
from pathlib import Path
from uuid import uuid4

from run_experiments import (ADJACENCIES, CLUSTERINGS, MODELS, atomic_json, busy_gpus,
                             dataset_signature, environment_versions, fingerprint_files,
                             gpu_inventory, inspect_datasets, pid_exists, signal_signature)

PROJECT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("pfm", "upstream_bidirectional")
ALPHAS = (0, .2, .4, .6, .8, 1.)
LAYERS = (1, 2, 3)
WIDTHS = (32, 64, 128)
CRITERIA = ("mse", "mae")


def signature():
    paths = [PROJECT / "scripts" / name for name in ("search_hyperparams.py", "launch_search.py", "train.py")]
    paths += [PROJECT / "src/traffic_flow_prediction" / f"{name}.py" for name in ("models", "engine", "datasets")]
    paths.append(PROJECT / "requirements.txt")
    return fingerprint_files(paths)


def key(group, graph, architecture, model, alpha, layers, width):
    return f"{group}__{graph}__{architecture}__{model}__a{round(alpha*10):02d}__l{layers}__h{width}"


def candidate(group, graph, architecture, model, alpha=1., layers=2, width=64, stage=0):
    return {"id": key(group, graph, architecture, model, alpha, layers, width), "cluster": group,
            "adjacency": graph, "architecture": architecture, "model": model, "time_alpha": alpha,
            "layers": layers, "hidden_feats": width, "stage": stage, "kind": "candidate",
            "status": "pending", "attempts": []}


def first_stage():
    return [candidate(*values) for values in itertools.product(CLUSTERINGS, ADJACENCIES, ARCHITECTURES, MODELS)]


def exhaustive_stage():
    return [candidate(group, graph, architecture, model, alpha, layers, width, stage=2)
            for group, graph, architecture, model, alpha, layers, width in
            itertools.product(CLUSTERINGS, ADJACENCIES, ARCHITECTURES, MODELS, ALPHAS, LAYERS, WIDTHS)]


def rank(jobs, criterion):
    return min(jobs, key=lambda job: (job["validation"][criterion], job["layers"], job["hidden_feats"], job["id"]))


def select(jobs, fields):
    groups = {}
    for job in jobs:
        groups.setdefault(tuple(job[field] for field in fields), []).append(job)
    chosen = {}
    for entries in groups.values():
        for criterion in CRITERIA:
            job = rank(entries, criterion)
            chosen[job["id"]] = job
    return chosen.values()


def next_stage(manifest):
    stage = manifest["stage"]
    jobs = [job for job in manifest["jobs"] if job["kind"] == "candidate" and job["stage"] == stage]
    if any(job["status"] != "completed" for job in jobs):
        raise ValueError("A stage cannot advance with missing or failed jobs.")
    chosen = select(jobs if stage == 0 else [job for job in manifest["jobs"] if job["kind"] == "candidate" and job["stage"] <= stage],
                    ("cluster", "adjacency", "architecture"))
    if stage == 0:
        proposals = [candidate(job["cluster"], job["adjacency"], job["architecture"], job["model"], alpha, stage=1)
                     for job in chosen for alpha in ALPHAS if alpha != 1.]
    elif stage == 1:
        proposals = [candidate(job["cluster"], job["adjacency"], job["architecture"], job["model"], job["time_alpha"], layers, width, 2)
                     for job in chosen for layers, width in itertools.product(LAYERS, WIDTHS) if (layers, width) != (2, 64)]
    elif stage == 2:
        leaders = {}
        completed = [job for job in manifest["jobs"] if job["kind"] == "candidate" and job["status"] == "completed"]
        for group in CLUSTERINGS:
            for criterion in CRITERIA:
                source = rank([job for job in completed if job["cluster"] == group], criterion)
                leaders[f"{group}/{criterion}"] = source["id"]
                for graph, architecture in itertools.product(ADJACENCIES, ARCHITECTURES):
                    local = rank([job for job in completed if job["cluster"] == group and job["adjacency"] == graph
                                  and job["architecture"] == architecture], criterion)
                    leaders[f"{group}/{graph}/{architecture}/{criterion}"] = local["id"]
        manifest["leaders"] = leaders
        proposals = []
        for source_id in dict.fromkeys(leaders.values()):
            source = next(job for job in completed if job["id"] == source_id)
            proposals.append({**{field: source[field] for field in ("cluster", "adjacency", "architecture", "model", "time_alpha", "layers", "hidden_feats")},
                              "id": f"final__{source_id}", "source_id": source_id, "stage": 3, "kind": "final",
                              "status": "pending", "attempts": []})
    else:
        manifest["stage"] = 4
        manifest["state"] = "completed"
        return
    existing = {job["id"] for job in manifest["jobs"]}
    for job in proposals:
        if job["id"] not in existing:
            manifest["jobs"].append(job)
            existing.add(job["id"])
    manifest["stage"] += 1


def command(job, manifest, output):
    if job["kind"] == "final":
        source = next(item for item in manifest["jobs"] if item["id"] == job["source_id"])
        return [sys.executable, "-B", str(PROJECT / "scripts/train.py"), "--evaluate-only", source["result_dir"],
                "--target", str(output), "--device", "gpu"]
    spatial, temporal = job["model"].split("_")
    dataset = manifest["datasets"][f"{job['cluster']}/{job['adjacency']}"]["path"]
    result = [sys.executable, "-B", str(PROJECT / "scripts/train.py"), "--source", dataset, "--target", str(output),
              "--device", "gpu", "--defer-test", "--architecture", job["architecture"], "--spatial", spatial,
              "--temporal", temporal, "--time-alpha", str(job["time_alpha"]), "--layers", str(job["layers"]),
              "--hidden-feats", str(job["hidden_feats"]), "--temporal-hidden", str(job["hidden_feats"])]
    for name, value in manifest["training"].items():
        result.extend([f"--{name.replace('_', '-')}", str(value)])
    return result


def validate(job, attempt, manifest):
    outputs = list(Path(attempt["folder"]).glob("output/exp_*/metrics.json"))
    if len(outputs) != 1:
        raise ValueError("Expected exactly one completed experiment.")
    folder = outputs[0].parent
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads(outputs[0].read_text(encoding="utf-8"))
    required = ("model.npz", "predictions.npz", "comparison.csv") if job["kind"] == "final" else ("model.npz",)
    if any(not (folder / name).is_file() or not (folder / name).stat().st_size for name in required):
        raise ValueError("Missing or empty experiment artifact.")
    if config["architecture"] != job["architecture"] or config["time_alpha"] != job["time_alpha"]:
        raise ValueError("Architecture or alpha differs from the planned condition.")
    if job["kind"] == "candidate":
        dataset = manifest["datasets"][f"{job['cluster']}/{job['adjacency']}"]
        if any(config[name] != value for name, value in (("spatial", job["model"].split("_")[0]),
                   ("temporal", job["model"].split("_")[1]), ("hidden_feats", job["hidden_feats"]),
                   ("temporal_hidden", job["hidden_feats"]), ("layers", job["layers"]),
                   ("dataset_signature", dataset["signature"]))):
            raise ValueError("Experiment does not match its planned condition or dataset.")
        if metrics["test"] is not None or len(metrics["history"]) != manifest["training"]["epochs"]:
            raise ValueError("Search candidates must not access test data.")
        scores = metrics["validation"]["model"]
    else:
        if metrics["test"] is None or not (folder / "predictions.npz").is_file():
            raise ValueError("Finalist lacks test evaluation.")
        scores = metrics["test"]["model"]
    if scores["count"] <= 0 or any(not math.isfinite(scores[name]) for name in CRITERIA):
        raise ValueError("Missing or nonfinite model metrics.")
    if job["kind"] == "final" and metrics["test"]["model"]["count"] != metrics["test"]["persistence"]["count"]:
        raise ValueError("Model and baseline masks differ.")
    return str(folder), metrics


def save(root, manifest):
    atomic_json(root / "manifest.json", manifest)
    rows = []
    for job in manifest["jobs"]:
        row = {field: job.get(field) for field in ("id", "cluster", "adjacency", "architecture", "model", "time_alpha", "layers", "hidden_feats", "kind", "stage", "status", "source_id", "result_dir", "validation", "test")}
        rows.append(row)
    atomic_json(root / "summary.json", {"complete": manifest["state"] == "completed", "stage": manifest["stage"],
                                        "leaders": manifest.get("leaders", {}), "results": rows,
                                        "warning": "Validation selects finalists; test is only evaluated for selected finalists. Do not compare absolute errors across different groupings."})


def write_rankings(root, manifest):
    destination = root / "rankings.csv"
    if destination.exists():
        return
    jobs = {job["id"]: job for job in manifest["jobs"]}
    rows = []
    for selection, source_id in manifest["leaders"].items():
        source = jobs[source_id]
        finalist = jobs[f"final__{source_id}"]
        if finalist["status"] != "completed":
            raise ValueError("Cannot rank an unevaluated finalist.")
        for horizon, model, baseline in [("all", finalist["test"]["model"], finalist["test"]["persistence"])]:
            rows.append({"selection": selection, "horizon": horizon, "cluster": source["cluster"],
                         "adjacency": source["adjacency"], "architecture": source["architecture"], "model": source["model"],
                         "time_alpha": source["time_alpha"], "layers": source["layers"], "hidden_feats": source["hidden_feats"],
                         "valid_mse": source["validation"]["mse"], "valid_mae": source["validation"]["mae"],
                         "test_mse": model["mse"], "test_mae": model["mae"], "persistence_mse": baseline["mse"],
                         "persistence_mae": baseline["mae"], "mse_skill": 1-model["mse"]/baseline["mse"] if model["mse"] is not None and baseline["mse"] else None,
                         "count": model["count"], "result_dir": finalist["result_dir"]})
        for model, baseline in zip(finalist["test"]["model"]["per_horizon"], finalist["test"]["persistence"]["per_horizon"]):
            row = dict(rows[-1], horizon=model["step"], test_mse=model["mse"], test_mae=model["mae"],
                       persistence_mse=baseline["mse"], persistence_mae=baseline["mae"], count=model["count"],
                       mse_skill=1-model["mse"]/baseline["mse"] if model["mse"] is not None and baseline["mse"] else None)
            rows.append(row)
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def recover(manifest, retry_failed):
    for job in manifest["jobs"]:
        if job["status"] == "running":
            attempt = job["attempts"][-1]
            if pid_exists(attempt.get("pid")):
                raise RuntimeError(f"Previous training may still be running: {job['id']}")
            try:
                folder, metrics = validate(job, attempt, manifest)
                job.update(status="completed", result_dir=folder)
                job["validation"] = metrics["validation"]["model"]
                if job["kind"] == "final":
                    job["test"] = metrics["test"]
                attempt["status"] = "completed"
            except (OSError, ValueError, KeyError):
                attempt["status"] = "interrupted"
                job["status"] = "pending"
        elif job["status"] == "failed" and retry_failed:
            job["status"] = "pending"


def run(root, manifest, gpus):
    inventory = gpu_inventory()
    if not set(gpus).issubset(inventory):
        raise ValueError("Unknown GPU index.")
    active = {}
    last_report = 0.
    try:
        while manifest["stage"] < 4:
            pending = [job for job in manifest["jobs"] if job["stage"] == manifest["stage"] and job["status"] == "pending"]
            for gpu, (process, job, attempt, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                attempt.update(status="failed" if code else "completed", exit_code=code)
                try:
                    if code:
                        raise ValueError(f"Training returned {code}")
                    folder, metrics = validate(job, attempt, manifest)
                    job.update(status="completed", result_dir=folder, validation=metrics["validation"]["model"])
                    if job["kind"] == "final":
                        job["test"] = metrics["test"]
                    print(f"COMPLETED {job['id']} GPU {gpu}", flush=True)
                except (OSError, ValueError, KeyError) as error:
                    job["status"] = "failed"
                    attempt.update(status="failed", error=str(error))
                    print(f"FAILED {job['id']}: {error} log={attempt['log']}", flush=True)
                del active[gpu]
                save(root, manifest)
            pending = [job for job in manifest["jobs"] if job["stage"] == manifest["stage"] and job["status"] == "pending"]
            if not pending and not active:
                if any(job["stage"] == manifest["stage"] and job["status"] == "failed" for job in manifest["jobs"]):
                    manifest["state"] = "incomplete"
                    save(root, manifest)
                    return False
                next_stage(manifest)
                save(root, manifest)
                continue
            occupied = busy_gpus(inventory) if pending else set()
            for gpu in gpus:
                if gpu in occupied or gpu in active or not pending:
                    continue
                if signature() != manifest["code_signature"] or environment_versions() != manifest["environment"]:
                    raise RuntimeError("Training code or dependencies changed during the search.")
                job = pending.pop(0)
                dataset = manifest["datasets"][f"{job['cluster']}/{job['adjacency']}"]
                if dataset_signature(Path(dataset["path"])) != dataset["signature"]:
                    raise RuntimeError("A dataset changed during the search.")
                folder = root / "jobs" / job["id"] / f"attempt_{len(job['attempts'])+1:02d}_{uuid4().hex[:8]}"
                folder.mkdir(parents=True, exist_ok=False)
                cmd = command(job, manifest, folder / "output")
                attempt = {"folder": str(folder), "log": str(folder / "train.log"), "gpu": gpu,
                           "command": cmd, "status": "running"}
                job["attempts"].append(attempt)
                job["status"] = "running"
                save(root, manifest)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda",
                           XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                stream = Path(attempt["log"]).open("x", encoding="utf-8")
                try:
                    process = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError:
                    stream.close()
                    job["status"] = attempt["status"] = "failed"
                    save(root, manifest)
                    raise
                attempt["pid"] = process.pid
                active[gpu] = (process, job, attempt, stream)
                save(root, manifest)
                print(f"STARTED {job['id']} GPU {gpu} log={attempt['log']}", flush=True)
            if time.monotonic()-last_report >= 60:
                counts = {status: sum(job["status"] == status for job in manifest["jobs"])
                          for status in ("completed", "running", "pending", "failed")}
                epochs = {}
                for _, job, attempt, _ in active.values():
                    with Path(attempt["log"]).open("rb") as stream:
                        stream.seek(max(0, Path(attempt["log"]).stat().st_size-8192))
                        recent = stream.read().decode("utf-8", errors="replace")
                    epochs[job["id"]] = max((int(value) for value in re.findall(r"Epoch\s+(\d+)", recent)), default=0)
                print(f"PROGRESO etapa={manifest['stage']} {counts} épocas_activas={epochs}", flush=True)
                last_report = time.monotonic()
            if active or pending:
                time.sleep(5)
    finally:
        for process, job, attempt, stream in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stream.close()
            attempt["status"] = "interrupted"
            job["status"] = "pending"
        save(root, manifest)
    return True


def main():
    parser = argparse.ArgumentParser(description="Búsqueda por etapas sobre cuatro horizontes; solo se entrena sin --dry-run.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path, default=Path("target/pfm_search_h4"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--exhaustive", action="store_true", help="Ejecuta las 4320 combinaciones sin poda.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if not args.gpus or min(args.gpus) < 0:
        parser.error("Select nonnegative GPU indices.")
    if args.resume:
        root = args.resume.resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if args.exhaustive and manifest.get("mode") != "exhaustive":
            raise ValueError("Cannot change the search mode when resuming.")
        if manifest["code_signature"] != signature() or manifest["environment"] != environment_versions():
            raise ValueError("Code or dependencies changed since the original search.")
        if args.source and args.source.resolve() != Path(manifest["source"]):
            raise ValueError("Cannot change dataset when resuming.")
        if any(dataset_signature(Path(entry["path"])) != entry["signature"] for entry in manifest["datasets"].values()):
            raise ValueError("A dataset changed since the original search.")
        if not args.dry_run:
            recover(manifest, args.retry_failed)
    else:
        if args.source is None:
            parser.error("--source is required for a new search.")
        datasets = inspect_datasets(args.source, CLUSTERINGS, ADJACENCIES)
        if any(entry["horizon"] != 4 for entry in datasets.values()):
            raise ValueError("Search requires four-step datasets.")
        root = (args.target / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}").resolve()
        manifest = {"format_version": 2, "source": str(args.source.resolve()), "datasets": datasets,
                    "code_signature": signature(), "environment": environment_versions(),
                    "mode": "exhaustive" if args.exhaustive else "staged", "stage": 2 if args.exhaustive else 0,
                    "state": "pending", "training": {"epochs": 25, "batch_size": 32, "micro_batch_size": 4,
                    "learning_rate": 1e-4, "time_decay": .05, "seed": 42},
                    "jobs": exhaustive_stage() if args.exhaustive else first_stage()}
    print(f"Búsqueda: {root} | etapa {manifest['stage']} | trabajos {len(manifest['jobs'])} | GPUs {args.gpus}", flush=True)
    if args.dry_run:
        print("Modo exhaustivo: 4320 entrenamientos + finalistas." if manifest["mode"] == "exhaustive" else
              "Etapas: 80 iniciales + hasta 200 alpha + hasta 320 profundidad/dimensión; prueba solo para finalistas.", flush=True)
        return
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    with (root / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(root, manifest)
        if not run(root, manifest, list(dict.fromkeys(args.gpus))):
            raise SystemExit("Incomplete search. Inspect failed job logs and resume with --retry-failed.")
        write_rankings(root, manifest)
    print(f"Resultados: {root / 'summary.json'} | ranking={root / 'rankings.csv'}", flush=True)


if __name__ == "__main__":
    main()
