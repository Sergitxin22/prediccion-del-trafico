import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from run_experiments import ADJACENCIES, CLUSTERINGS, inspect_datasets

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Prepara datos reales de cuatro pasos y ejecuta la búsqueda exhaustiva reanudable.")
    parser.add_argument("--data", type=Path, default=Path("data/processed_pfm_search_h4_20261008"))
    parser.add_argument("--target", type=Path, default=Path("target/pfm_search_h4"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.gpus or min(args.gpus) < 0:
        parser.error("Las GPUs deben ser índices no negativos.")
    data, target = args.data.resolve(), args.target.resolve()
    if args.dry_run:
        if data.exists():
            datasets = inspect_datasets(data, CLUSTERINGS, ADJACENCIES)
            if any(item["horizon"] != 4 for item in datasets.values()):
                raise ValueError("El dataset existente no tiene horizonte 4.")
            command = [sys.executable, "-B", str(PROJECT / "scripts/search_hyperparams.py"),
                       "--source", str(data), "--target", str(target), "--exhaustive", "--dry-run",
                       "--gpus", *(str(gpu) for gpu in args.gpus)]
            subprocess.run(command, cwd=PROJECT, check=True)
        else:
            print(f"El dataset aún no existe: {data}. El lanzamiento lo generará a partir de los CSV reales.")
            print("Después verificará diez datasets y planificará 4320 entrenamientos en cuatro GPUs.")
        return
    if args.detach and not args.run:
        target.mkdir(parents=True, exist_ok=True)
        folder = target / f"launch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
        folder.mkdir(exist_ok=False)
        log = folder / "launcher.log"
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--run", "--data", str(data),
                   "--target", str(target), "--gpus", *(str(gpu) for gpu in args.gpus)]
        if args.resume:
            command.extend(["--resume", str(args.resume.resolve(strict=True))])
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        with (folder / "pid.json").open("x", encoding="utf-8") as file:
            json.dump({"pid": process.pid, "log": str(log), "data": str(data), "resume": str(args.resume) if args.resume else None}, file)
        print(f"Lanzador en segundo plano. PID: {process.pid}\nLog principal: {log}")
        print(f"Consultar: tail -f '{log}'")
        return
    if args.resume and not args.resume.resolve(strict=True).is_dir():
        raise ValueError("La ruta de reanudación debe ser un directorio de búsqueda.")
    if not data.exists():
        if args.resume:
            raise ValueError("No se puede reanudar sin los datasets originales.")
        command = [sys.executable, "-B", str(PROJECT / "scripts/generate_datasets.py"),
                   "--target", str(data), "--window", "12", "--horizon", "4", "--clusterings", *CLUSTERINGS,
                   "--adjacencies", *ADJACENCIES, "--negative-correlation-policy", "zero"]
        print(f"GENERANDO diez datasets nuevos desde data/raw/traffic y data/raw/sensors: {data}", flush=True)
        subprocess.run(command, cwd=PROJECT, check=True, env=dict(os.environ, JAX_PLATFORMS="cpu"))
    datasets = inspect_datasets(data, CLUSTERINGS, ADJACENCIES)
    if any(item["horizon"] != 4 for item in datasets.values()):
        raise ValueError("Los datasets existentes deben tener horizonte cuatro; no se sobrescriben.")
    command = [sys.executable, "-B", str(PROJECT / "scripts/search_hyperparams.py"),
               "--source", str(data), "--target", str(target), "--gpus", *(str(gpu) for gpu in args.gpus)]
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve(strict=True)), "--retry-failed"])
    else:
        command.append("--exhaustive")
    print("INICIANDO búsqueda exhaustiva de 4320 entrenamientos; los logs individuales se guardan en jobs/*/attempt_*/train.log.", flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)
    print("BÚSQUEDA TERMINADA. Ver el resumen del run indicado arriba y abrirlo con scripts/dashboard.py.", flush=True)


if __name__ == "__main__":
    main()
