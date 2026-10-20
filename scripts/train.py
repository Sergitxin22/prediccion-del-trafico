import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.append(Path("src").absolute().as_posix())

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import jraph
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import nnx
from jax.typing import ArrayLike

from traffic_flow_prediction.datasets import WindowDataLoader, load_checkpoint, save_checkpoint
from traffic_flow_prediction.engine import ErrorTotals, baselines, evaluate, train_step
from traffic_flow_prediction.models import GNN, SpatialType, TemporalType

fprint = partial(print, flush=True)


def fprint_tree(tree: Any, name: str = "tree", shape: bool = True) -> None:
    if shape:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}.shape = {v.shape}"), tree)
    else:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}\n{v}"), tree)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelos espacio-temporales del PFM con evaluacion frente a referencias.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path, default=Path("target/pfm"))
    parser.add_argument("--nodes", type=int, nargs="+", help="Indices internos que se dibujan; no limita los nodos de entrenamiento.")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32, help="Muestras por actualizacion del optimizador.")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Muestras simultaneas en memoria; se acumulan gradientes.")
    parser.add_argument("--shuffle", action="store_true", help="Desordena solo las ventanas de entrenamiento.")
    parser.add_argument("--hidden-feats", type=int, default=64)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--spatial", choices=[x.name for x in SpatialType], default="gcn")
    parser.add_argument("--temporal", choices=[x.name for x in TemporalType], default="lstm")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--time-decay", type=float, default=.05)
    parser.add_argument("--time-alpha", "--time_alpha", type=float, default=1.)
    parser.add_argument("--architecture", choices=("pfm", "upstream_bidirectional"), default="pfm")
    parser.add_argument("--selection-metric", choices=("mse", "mae"), default="mse")
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--plot-points", type=int, default=192)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--evaluate-only", type=Path, help="Carpeta de un experimento guardado; no reentrena.")
    args = parser.parse_args()
    if min(args.batch_size, args.micro_batch_size, args.hidden_feats, args.temporal_hidden, args.layers, args.epochs, args.plot_points) < 1:
        parser.error("Batch sizes, model dimensions, epochs and plot points must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.time_decay) or args.time_decay < 0:
        parser.error("Invalid learning rate or time decay.")
    if not np.isfinite(args.time_alpha) or not 0 <= args.time_alpha <= 1:
        parser.error("--time-alpha must be between 0 and 1.")
    if args.defer_test and args.evaluate_only:
        parser.error("--defer-test cannot be used with --evaluate-only.")
    return args


def load_data(path_to_source: Path):
    path = path_to_source / "metadata.json"
    if not path.is_file():
        raise ValueError("Legacy dataset without PFM metadata. Generate a new dataset with scripts/generate_datasets.py --target <new-folder>.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("index") != "pfm":
        raise ValueError("Incompatible dataset: regenerate it with the current scripts/generate_datasets.py.")
    with np.load(path_to_source / "series.npz", allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    with np.load(path_to_source / "graph.npz", allow_pickle=False) as file:
        graph = {key: file[key] for key in file.files}
    n = data["values"].shape[1]
    if data["values"].shape != data["observed"].shape or not np.isfinite(data["values"]).all():
        raise ValueError("Invalid values or observation mask.")
    if not np.array_equal(graph["nodes_idx"], np.arange(n)) or not np.array_equal(data["node_labels"], graph["node_labels"]):
        raise ValueError("Series and graph must use the same compact node order.")
    if not np.isfinite(graph["edges"]).all() or np.any(graph["edges"] < 0):
        raise ValueError("Expected finite nonnegative graph weights.")
    for split in ("train", "valid", "test"):
        starts = data[f"{split}_starts"]
        if not len(starts) or np.any(starts < int(data["window"])) or np.any(starts+int(data["horizon"]) > len(data["times"])):
            raise ValueError(f"Invalid forecast starts in {split}.")
    return graph, data, metadata


def prepare_data(x: ArrayLike, y: ArrayLike, graph_data: Mapping[str, ArrayLike]) -> tuple[jraph.GraphsTuple, jax.Array]:
    x, y = np.asarray(x), np.asarray(y)
    batch, time, n, features = x.shape
    if not np.array_equal(graph_data["nodes_idx"], np.arange(n)) or y.shape[0] != batch or y.shape[2] != n:
        raise ValueError("Batch and graph must use identical compact node indices.")
    senders = np.asarray(graph_data["senders"], np.int32)
    receivers = np.asarray(graph_data["receivers"], np.int32)
    edges = np.asarray(graph_data["edges"], np.float32)
    if senders.shape != receivers.shape or edges.shape != (len(senders), 1):
        raise ValueError("Inconsistent graph edge arrays.")
    if len(senders) and (min(senders.min(), receivers.min()) < 0 or max(senders.max(), receivers.max()) >= n):
        raise ValueError("Graph edges are outside the compact node range.")
    offsets = np.arange(batch, dtype=np.int32)[:, None]*n
    graph = jraph.GraphsTuple(
        nodes=jnp.asarray(x.transpose(0, 2, 1, 3).reshape(batch*n, time, features)),
        edges=jnp.asarray(np.tile(edges, (batch, 1))),
        senders=jnp.asarray((senders[None, :]+offsets).reshape(-1)),
        receivers=jnp.asarray((receivers[None, :]+offsets).reshape(-1)),
        n_node=jnp.full((batch,), n, dtype=jnp.int32), n_edge=jnp.full((batch,), len(senders), dtype=jnp.int32), globals=None,
    )
    return graph, jnp.asarray(y.transpose(0, 2, 1, 3).reshape(batch*n, y.shape[1], y.shape[3]))


@jax.jit
def get_signal(x: jraph.GraphsTuple, y: ArrayLike) -> jax.Array:
    n_graph = x.n_node.shape[0]
    ys = jnp.array_split(y[:, 0], n_graph, axis=0)
    return jnp.stack(ys, axis=1)
    # n_batch = x.n_node.shape[0]
    # n_sensor = y.shape[0] // n_batch
    # n_time = y.shape[1]
    # ys = jnp.array_split(y, n_batch, axis=0)
    # ys = jtree.tree_map(
    #     lambda yi, i: jnp.zeros((n_sensor, n_batch + n_time - 1)).at[:, i:(i+n_time)].set(yi),
    #     ys,
    #     list(range(n_batch)),
    # )
    # z = jnp.stack(ys, axis=0)
    
    # r = jnp.arange(n_time, dtype=jnp.float32)
    # n = jnp.zeros(n_batch + n_time - 1).at[-n_time:].add(r).at[:n_time].add(jnp.flip(r))
    # n = jnp.reshape(n, (1, -1))
    # return jnp.sum(z, axis=0) / (n_time - n)


def micro_batches(x, y, mask, graph_data, size):
    for offset in range(0, len(x), size):
        graph, target = prepare_data(x[offset:offset+size], y[offset:offset+size], graph_data)
        valid = mask[offset:offset+size].transpose(0, 2, 1, 3).reshape(target.shape)
        yield graph, target, jnp.asarray(valid)


def evaluate_split(model, data, graph_data, split, batch_size, *, collect=False):
    horizon = int(data["horizon"])
    totals = {name: ErrorTotals(horizon) for name in ("model", "persistence", "window_mean", "zero")}
    records = {name: [] for name in ("model", "target", "observed", "persistence", "window_mean", "starts")}
    for x, target, mask, starts in WindowDataLoader(data, split, batch_size):
        graph, _ = prepare_data(x, target, graph_data)
        prediction = np.asarray(evaluate(model, graph)).reshape(len(x), x.shape[2], horizon, 1).transpose(0, 2, 1, 3)
        predictions = {"model": prediction, **baselines(x, horizon)}
        for name, output in predictions.items():
            totals[name].update(output, target, mask)
        if collect:
            values = {"model": prediction, "target": target, "observed": mask, "persistence": predictions["persistence"],
                      "window_mean": predictions["window_mean"], "starts": starts}
            for name, value in values.items():
                records[name].append(value)
    return {name: total.compute() for name, total in totals.items()}, {name: np.concatenate(values) for name, values in records.items()} if collect else None


def dataset_signature(source):
    digest = hashlib.sha256()
    for name in ("series.npz", "graph.npz", "metadata.json"):
        with (source / name).open("rb") as file:
            for block in iter(lambda: file.read(1024*1024), b""):
                digest.update(block)
    return digest.hexdigest()


def plot_signals(folder, signals, data, metadata, nodes, labels, limit):
    times = data["times"][signals["starts"][:, None]+np.arange(int(data["horizon"]))]
    origins = data["times"][signals["starts"]-1]
    np.savez_compressed(folder / "predictions.npz", **signals, target_times=times, origin_times=origins, node_labels=data["node_labels"])
    rows = []
    for node in nodes:
        for step in range(int(data["horizon"])):
            valid = signals["observed"][:, step, node, 0]
            frame = pd.DataFrame({"node_index": node, "node_label": data["node_labels"][node], "horizon_step": step+1,
                                  "origin_time": origins, "target_time": times[:, step], "observed": valid,
                                  "target": np.where(valid, signals["target"][:, step, node, 0], np.nan),
                                  "prediction": signals["model"][:, step, node, 0],
                                  "persistence": signals["persistence"][:, step, node, 0],
                                  "window_mean": signals["window_mean"][:, step, node, 0]})
            rows.append(frame)
            grid = pd.date_range(times[0, step], times[-1, step], freq=pd.Timedelta(metadata["frequency"]))
            plotted = frame.set_index("target_time").reindex(grid).iloc[:limit]
            fig, ax = plt.subplots(figsize=(12, 4))
            for column, name, style in (("target", "Indice PFM observado", "-"), ("prediction", "Modelo", "-"),
                                       ("persistence", "Ultimo valor disponible", "--"), ("window_mean", "Media reciente", ":")):
                ax.plot(plotted.index, plotted[column], style, label=name, alpha=.9)
            minutes = (step+1)*pd.Timedelta(metadata["frequency"])/pd.Timedelta(minutes=1)
            ax.set_title(f"{labels[node]} | prediccion a +{minutes:g} minutos")
            ax.set_ylabel("Indice C del PFM (no es una probabilidad)")
            ax.set_xlabel("Fecha objetivo")
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / f"node_{node}_h{step+1}.png", dpi=130)
            plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(folder / "selected_predictions.csv", index=False)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)


def main() -> None:
    args = get_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices(args.device)
    if not devices:
        raise RuntimeError(f"No {args.device} device available; choose --device explicitly, no silent fallback.")
    device = devices[0]
    previous = None
    if args.evaluate_only:
        previous = json.loads((args.evaluate_only / "config.json").read_text(encoding="utf-8"))
        if previous.get("architecture_version") not in ("pfm-jax-1", "pfm-jax-upstream-1"):
            raise ValueError("Incompatible checkpoint architecture.")
        for key in ("spatial", "temporal", "hidden_feats", "temporal_hidden", "layers", "seed", "architecture", "time_alpha"):
            setattr(args, key, previous.get(key, getattr(args, key)))
    source = args.source or (Path(previous["source"]) if previous else Path("data/processed_pfm/base/closeness"))
    graph_data, data, metadata = load_data(source)
    if args.time_alpha < 1 and int(data["horizon"]) < 2 and not previous:
        raise ValueError("--time-alpha below 1 requires a dataset with horizon >= 2.")
    signature = dataset_signature(source)
    if previous and signature != previous["dataset_signature"]:
        raise ValueError("Checkpoint and dataset differ.")
    n = data["values"].shape[1]
    nodes = args.nodes or list(range(min(3, n)))
    if min(nodes) < 0 or max(nodes) >= n:
        raise ValueError(f"--nodes selects plot indices from 0 to {n-1}; see nodes.csv.")
    labels = pd.read_csv(source / "nodes.csv").set_index("node_index")["name"].to_dict()
    folder = args.target / f"exp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"source": str(source.resolve()), "nodes": nodes, "architecture_version": "pfm-jax-1" if args.architecture == "pfm" else "pfm-jax-upstream-1", "dataset_signature": signature,
                   "device_info": {"platform": device.platform, "name": device.device_kind, "local_id": device.id},
                   "jax_version": jax.__version__, "matmul_precision": "highest",
                   "selection": f"lowest validation {args.selection_metric.upper()} over observed targets and all horizons",
                   "effective_batch_size": args.batch_size, "external_edge_weights_used": args.spatial == "gcn",
                   "model_flow": "spatial layers with ReLU -> recurrent last state -> linear output" if args.architecture == "pfm" else "spatial layers with ReLU -> bidirectional recurrent encoder -> temporal projection -> linear output",
                   "upstream_commit": "d5498c1d97dfd47f1dd106d8691a10c44cd81b50" if args.architecture == "upstream_bidirectional" else None,
                   "implementation_choices": "hidden sizes, initialization, learning rate, shuffle and missing-data policy are explicit choices where PFM lacks exact settings",
                   "scope": "PFM-style architecture in JAX, not bitwise parity with unavailable original PyTorch implementation"})
    with jax.default_device(device):
        model = GNN(1, args.hidden_feats, 1, int(data["window"]), int(data["horizon"]), args.layers,
                    spatial=SpatialType[args.spatial], temporal=TemporalType[args.temporal], temporal_hidden=args.temporal_hidden,
                    architecture=args.architecture, rngs=nnx.Rngs(args.seed, params=args.seed+1))
        parameters = jtree.tree_leaves(nnx.state(model, nnx.Param))
        if any(d.platform != args.device for value in parameters for d in value.devices()):
            raise RuntimeError("Model parameters are on an unexpected device.")
        config["parameter_count"] = sum(value.size for value in parameters)
        write_json(folder / "config.json", config)
        fprint(f"Experimento nuevo: {folder}")
        fprint(f"{args.spatial.upper()} + {args.temporal.upper()} | nodos={n} | dispositivo={device.device_kind}")
        fprint(f"Lote efectivo={args.batch_size}, micro-lote={args.micro_batch_size} | Ambas ReLU conservadas | Graficas: {nodes}")
        writer = None
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(folder / "runs")
        history, best_state, best_epoch, best_mse = [], None, None, float("inf")
        if previous:
            load_checkpoint(args.evaluate_only / "model.npz", model)
        else:
            optimiser = nnx.Optimizer(model, optax.adam(args.learning_rate))
            loader = WindowDataLoader(data, "train", args.batch_size, shuffle=args.shuffle, seed=args.seed)
            for epoch in range(args.epochs):
                loss_sum, count = 0., 0.
                for x, y, mask, _ in loader:
                    loss, weight = train_step(model, optimiser, micro_batches(x, y, mask, graph_data, args.micro_batch_size), decay=args.time_decay, alpha=args.time_alpha)
                    loss_sum += loss*weight
                    count += weight
                if not count:
                    raise ValueError("No valid training targets.")
                scores, _ = evaluate_split(model, data, graph_data, "valid", args.micro_batch_size)
                mse = scores["model"]["mse"]
                selected = scores["model"][args.selection_metric]
                history.append({"epoch": epoch+1, "train_loss": loss_sum/count, "validation": scores})
                fprint(f"Epoch {epoch+1:3d} | Train loss {loss_sum/count:.8f} | Valid MSE {mse:.8f} | Valid MAE {scores['model']['mae']:.8f} | Persistence MSE {scores['persistence']['mse']:.8f}")
                if writer:
                    writer.add_scalar("train/loss", loss_sum/count, epoch+1)
                    writer.add_scalar("valid/mse", mse, epoch+1)
                if selected is not None and selected < best_mse:
                    best_mse, best_epoch = selected, epoch+1
                    best_state = jtree.tree_map(lambda value: jnp.array(value), nnx.state(model, nnx.Param))
            if best_state is None:
                raise ValueError("No valid checkpoint was produced.")
            nnx.update(model, best_state)
        save_checkpoint(folder / "model.npz", model)
        validation, _ = evaluate_split(model, data, graph_data, "valid", args.micro_batch_size)
        if args.defer_test:
            write_json(folder / "metrics.json", {"best_epoch": best_epoch, "history": history, "validation": validation,
                                                 "test": None, "selection_metric": args.selection_metric})
            if writer:
                writer.flush()
                writer.close()
            fprint(f"Validación completada sin consultar prueba: {folder}")
            return
        test, signals = evaluate_split(model, data, graph_data, "test", args.micro_batch_size, collect=True)
        if writer:
            writer.flush()
            writer.close()
    constant = np.all(np.ptp(signals["model"], axis=0) <= 1e-7, axis=(0, 2))
    valid_predictions = signals["model"][signals["observed"]]
    reference = test["persistence"]["mse"]
    result = {"best_epoch": best_epoch, "loaded_checkpoint": str(args.evaluate_only) if previous else None,
              "history": history, "validation": validation, "test": test, "dataset": metadata,
              "constant_prediction_node_indices": np.flatnonzero(constant).tolist(),
              "observed_prediction_range": [float(valid_predictions.min()), float(valid_predictions.max())],
              "observed_predictions_outside_index_range": int(np.sum((valid_predictions < 0) | (valid_predictions > 1))),
              "mse_skill_vs_persistence": 1-test["model"]["mse"]/reference if reference else None,
              "warning": "Low average error does not prove congestion-event anticipation; evaluate against baselines and inspect temporal changes."}
    write_json(folder / "metrics.json", result)
    pd.DataFrame([dict(method=name, **{key: score[key] for key in ("mse", "rmse", "mae", "count")}) for name, score in test.items()]).to_csv(folder / "comparison.csv", index=False)
    fprint(f"Mejor epoca: {best_epoch} | Nodos con prediccion constante: {int(constant.sum())}/{n}")
    for name, score in test.items():
        fprint(f"{name:12s} | MSE {score['mse']:.8f} | RMSE {score['rmse']:.8f} | MAE {score['mae']:.8f} | n={score['count']}")
    plot_signals(folder, signals, data, metadata, nodes, labels, args.plot_points)
    fprint(f"Modelo, configuracion, metricas, predicciones y graficas: {folder}")


if __name__ == "__main__":
    main()
