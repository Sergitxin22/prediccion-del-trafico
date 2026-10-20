"""Épocas por modelo a partir de un presupuesto de operaciones.

n_ops = k * (x * n * λ * T + y * n * T)
n_epocas = MAX_OPS // n_ops

n, T y k salen de cada dataset; λ es el --layers del entrenamiento (2 en el protocolo).
x es el coste de una aplicación GCN o GAT (un nodo, una capa, un instante) e y es
el coste de un paso LSTM o GRU. El bloque espacial se repite en cada instante de
la ventana, igual que el recurrente. Cada modelo aporta x, y y MAX_OPS.
"""

import argparse
import csv
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from run_experiments import DEFAULTS, MODELS

FIELDS = ("dataset", "cluster", "adjacency", "model", "n", "layers", "T", "k",
          "x", "y", "spatial_ops", "temporal_ops", "n_ops", "max_ops", "n_epochs")


def parse_count(value, name):
    try:
        number = Decimal(str(value).strip().replace("_", ""))
    except InvalidOperation as error:
        raise ValueError(f"{name} debe ser un entero positivo.") from error
    if not number.is_finite() or number != number.to_integral_value() or number < 1:
        raise ValueError(f"{name} debe ser un entero positivo.")
    return int(number)


def operation_count(nodes, layers, time_steps, graphs, x, y):
    if min(nodes, layers, time_steps, graphs, x, y) < 1:
        raise ValueError("Nodos, capas, iteraciones, grafos, x e y tienen que ser positivos.")
    spatial = graphs * x * nodes * layers * time_steps
    temporal = graphs * y * nodes * time_steps
    return spatial, temporal, spatial + temporal


def epoch_count(max_ops, n_ops):
    if max_ops < 1 or n_ops < 1:
        raise ValueError("MAX_OPS y n_ops tienen que ser positivos.")
    return max_ops // n_ops


def discover_datasets(source):
    source = Path(source)
    if (source / "metadata.json").is_file():
        return [source]
    found = sorted(path.parent for path in source.glob("*/*/metadata.json"))
    if not found:
        raise ValueError(f"No hay datasets con metadata.json en {source}.")
    return found


def training_factors(path):
    path = Path(path)
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Falta metadata.json en {path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(path / "series.npz", mmap_mode="r") as series:
        nodes = int(series["values"].shape[1])
        window = int(series["window"])
        graphs = int(series["train_starts"].shape[0])
    with np.load(path / "graph.npz", mmap_mode="r") as graph:
        indexed = int(graph["nodes_idx"].shape[0])
    if int(metadata["n_nodes"]) != nodes or indexed != nodes:
        raise ValueError(f"El número de nodos no coincide entre metadata, series y grafo: {path}.")
    if int(metadata["window"]) != window:
        raise ValueError(f"La ventana de metadata.json no coincide con series.npz: {path}.")
    declared = metadata.get("split_samples", {}).get("train")
    if declared is not None and int(declared) != graphs:
        raise ValueError(f"split_samples.train no coincide con train_starts: {path}.")
    return {"dataset": str(path), "cluster": metadata.get("cluster", path.parent.name),
            "adjacency": metadata.get("adjacency", path.name), "n": nodes, "T": window, "k": graphs}


def parse_budget(model, x, y, max_ops):
    if model not in MODELS:
        raise ValueError(f"Modelo desconocido: {model}. Esperados: {', '.join(MODELS)}.")
    return {"model": model, "x": parse_count(x, "x"), "y": parse_count(y, "y"),
            "max_ops": parse_count(max_ops, "MAX_OPS")}


def load_budgets(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("El JSON de presupuestos debe ser un objeto con un modelo por clave.")
    budgets = [parse_budget(model, item["x"], item["y"], item.get("max_ops", item.get("max-ops")))
               for model, item in payload.items()]
    if len({item["model"] for item in budgets}) != len(budgets):
        raise ValueError("Cada modelo solo puede aparecer una vez.")
    return budgets


def build_rows(datasets, budgets, layers):
    rows = []
    for dataset in datasets:
        factors = training_factors(dataset)
        for layer_count in layers:
            for budget in budgets:
                spatial, temporal, n_ops = operation_count(
                    factors["n"], layer_count, factors["T"], factors["k"], budget["x"], budget["y"])
                rows.append({**factors, "model": budget["model"], "layers": layer_count,
                             "x": budget["x"], "y": budget["y"], "spatial_ops": spatial,
                             "temporal_ops": temporal, "n_ops": n_ops, "max_ops": budget["max_ops"],
                             "n_epochs": epoch_count(budget["max_ops"], n_ops)})
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows):
    lines = ["dataset modelo capas n T k n_ops max_ops n_epocas"]
    for row in rows:
        label = f"{row['cluster']}/{row['adjacency']}"
        lines.append(f"{label} {row['model']} {row['layers']} {row['n']} {row['T']} {row['k']} "
                     f"{row['n_ops']} {row['max_ops']} {row['n_epochs']}")
    return "\n".join(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Calcula n_epocas = MAX_OPS // (k * (x * n * capas * T + y * n * T)) para cada modelo y dataset.")
    parser.add_argument("--source", type=Path, required=True,
                        help="Carpeta de un dataset o raíz con agrupacion/grafo/metadata.json.")
    parser.add_argument("--layers", type=int, nargs="+", default=[DEFAULTS["layers"]],
                        help="Capas espaciales λ. Por defecto, el valor del protocolo de entrenamiento.")
    parser.add_argument("--budget", action="append", nargs=4, metavar=("MODELO", "X", "Y", "MAX_OPS"),
                        help="Presupuesto de un modelo. Repetible. X: op/nodo/capa. Y: op/iteración temporal.")
    parser.add_argument("--budgets", type=Path, help="JSON {modelo: {x, y, max_ops}} en lugar de --budget.")
    parser.add_argument("--output", type=Path, help="CSV con una fila por dataset, modelo y número de capas.")
    args = parser.parse_args(argv)
    if min(args.layers) < 1:
        parser.error("El número de capas tiene que ser positivo.")
    if bool(args.budget) == bool(args.budgets):
        parser.error("Indica los presupuestos con --budget repetido o con --budgets, no con ambos ni con ninguno.")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    budgets = load_budgets(args.budgets) if args.budgets else [parse_budget(*item) for item in args.budget]
    if len({item["model"] for item in budgets}) != len(budgets):
        raise ValueError("Cada modelo solo puede aparecer una vez.")
    rows = build_rows(discover_datasets(args.source), budgets, args.layers)
    print(format_table(rows), flush=True)
    if args.output:
        write_csv(args.output, rows)
        print(f"CSV: {args.output}", flush=True)
    return rows


if __name__ == "__main__":
    main()
