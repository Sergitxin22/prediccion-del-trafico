import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import generate_datasets as original


def training_cutoff(dates, window=12, horizon=1, train_ratio=.7):
    dates = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
    if min(window, horizon) < 1 or not 0 < train_ratio < 1:
        raise ValueError("Invalid window, horizon or training ratio.")
    samples = len(dates)-window-horizon+1
    n_train = int(samples*train_ratio)
    if n_train < 1 or (samples-n_train)//2 < 1:
        raise ValueError("Not enough timestamps for the original three-way split.")
    return dates[window+n_train+horizon-2]


def fit_saturation(traffic, cutoff):
    valid = (traffic["date"] <= cutoff) & np.isfinite(traffic["intensity"]) & (traffic["intensity"] >= 0)
    return traffic.loc[valid].groupby("id", observed=True)["intensity"].quantile(.98)


def pfm_congestion(traffic, saturation):
    intensity, occupation = traffic["intensity"], traffic["occupation"]
    capacity = traffic["id"].map(saturation)
    valid = np.isfinite(intensity) & np.isfinite(occupation) & np.isfinite(capacity)
    valid &= (intensity >= 0) & occupation.between(0, 100) & (capacity > 0)
    y = (intensity.where(valid)/capacity.where(valid)).clip(0, 1)
    return (y + (1-y)*occupation/100).where(valid).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Cambia solo el indice de congestion, reutilizando la preparacion original de base/closeness.")
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--target", type=Path, default=Path("data/processed_original_pfm_index"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=.7)
    args = parser.parse_args()
    if args.target.exists():
        parser.error("El destino ya existe. Elige una carpeta nueva; no se sobrescriben datasets.")
    reference = args.reference or args.source / "processed/base/closeness"
    if not (reference / "graph.npz").is_file():
        parser.error("Se requiere el dataset original de referencia para verificar la comparacion.")
    traffic, sensors = original.load_data(args.source)
    cluster = original.generate_base(sensors)
    dates = pd.DatetimeIndex(traffic.loc[traffic["id"].isin(cluster["id"]), "date"].unique()).sort_values()
    cutoff = training_cutoff(dates, args.window, args.horizon, args.train_ratio)
    saturation = fit_saturation(traffic, cutoff)
    index = pfm_congestion(traffic, saturation)
    invalid = int(index.isna().sum())
    traffic["congestion"] = index.fillna(0.)
    zero_intensity_valid = int(((traffic["intensity"] == 0) & index.notna()).sum())
    print(f"Indice PFM calculado. Fin de calibracion P98: {cutoff}", flush=True)
    matrix = original.get_cluster_time_matrix(cluster, original.CLType.base, traffic)
    if not matrix.index.equals(dates):
        raise ValueError("The calibration timestamps do not match the generated matrix.")
    centroids = original.compute_centroids(cluster, original.CLType.base)
    matrix, df_edges, nodes_idx = original.generate_closeness(matrix, 5, centroids)
    edges, receivers, senders = original.to_edges(df_edges)
    graph = {"edges": edges, "receivers": receivers, "senders": senders,
             "nodes_idx": original.jnp.asarray(nodes_idx, dtype=original.jnp.int32)}
    with np.load(reference / "graph.npz", allow_pickle=False) as previous:
        for key, value in graph.items():
            if not np.array_equal(np.asarray(value), previous[key]):
                raise ValueError(f"The graph changed at {key}; this would not be an index-only comparison.")
    previous_dates = pd.read_csv(reference / "matrix_gnn.csv", usecols=["date"], parse_dates=["date"])["date"]
    if not np.array_equal(matrix.index.to_numpy(), previous_dates.to_numpy()):
        raise ValueError("Timestamps differ from the original dataset.")
    values = original.jnp.asarray(matrix.values, dtype=original.jnp.float32)
    if not np.isfinite(matrix.to_numpy()).all() or np.any((matrix.to_numpy() < 0) | (matrix.to_numpy() > 1)):
        raise ValueError("The PFM index must be finite and in [0,1].")
    x, y = original.create_windows(values, args.window, args.horizon)
    partitions = original.split_time(x, y, args.train_ratio)
    for name, target in zip(("train", "valid", "test"), partitions[1::2]):
        with np.load(reference / f"{name}.npz", allow_pickle=False) as previous:
            if target.shape != previous["y"].shape:
                raise ValueError(f"The {name} target shape differs from the original dataset.")
    folder = args.target / "base/closeness"
    args.target.mkdir(parents=True, exist_ok=False)
    folder.mkdir(parents=True, exist_ok=False)
    for offset, name in enumerate(("train", "valid", "test")):
        original.jnp.savez(folder / f"{name}.npz", x=partitions[2*offset], y=partitions[2*offset+1])
    original.jnp.savez(folder / "graph.npz", **graph)
    matrix.to_csv(folder / "matrix_gnn.csv")
    df_edges.to_csv(folder / "df_edges.csv", index=False)
    saturation.rename("saturation_p98").to_csv(folder / "saturation_p98.csv", index_label="sensor_id")
    metadata = {
        "reference": str(reference.resolve()), "source": str(args.source.resolve()),
        "index_formula": "Is_j=P98(I_j); Y=clip(I/Is_j,0,1); C=Y+(1-Y)*(occupation/100)",
        "document_reference": "PFM.pdf section 4.1, printed pages 15-16",
        "saturation_fit": "per sensor, valid nonnegative training intensities only; linear percentile interpolation",
        "saturation_training_cutoff_inclusive": str(cutoff),
        "window": args.window, "horizon": args.horizon, "train_ratio": args.train_ratio,
        "node_case": "base", "adjacency": "closeness", "n_nodes": len(nodes_idx), "n_timestamps": len(matrix),
        "split_shapes": {name: {"x": list(partitions[2*i].shape), "y": list(partitions[2*i+1].shape)} for i, name in enumerate(("train", "valid", "test"))},
        "graph_identical_to_reference": True, "timestamps_identical_to_reference": True,
        "target_shapes_identical_to_reference": True,
        "invalid_index_rows_filled_with_zero": invalid, "zero_intensity_rows_with_valid_index": zero_intensity_valid,
        "invalid_index_policy": "negative/nonfinite intensity, occupancy outside [0,100], or nonpositive/missing saturation produce NaN; fill with zero as in the original pipeline",
        "unchanged": "original sensor grouping, closeness graph, bidirectional interpolation, windows, splits, graph indexing/batching, architecture and training/evaluation code",
        "scope": "index-only experiment, not a full reproduction or correction of the PFM architecture",
        "warning": "Original preprocessing and implementation limitations remain. Errors are not directly comparable to occupancy/intensity errors.",
    }
    with (folder / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Dataset nuevo: {folder}", flush=True)
    print(f"Nodos: {len(nodes_idx)} | Instantes: {len(matrix)} | Grafo, fechas y tamanos verificados contra el original.", flush=True)
    print(f"Indice en [0,1] | Registros invalidos rellenados con cero: {invalid} | Intensidad cero con indice valido: {zero_intensity_valid}", flush=True)


if __name__ == "__main__":
    main()
