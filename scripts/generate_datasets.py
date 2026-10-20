import argparse
import json
import re
import sys
from collections.abc import Collection
from functools import partial
from pathlib import Path
from typing import Any

sys.path.append(Path("src").absolute().as_posix())

import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import pandas as pd
from tqdm import tqdm

from traffic_flow_prediction.adjacency import ADJType, generate_closeness, generate_correlation
from traffic_flow_prediction.clustering import CLType, compute_centroids, generate_base, generate_behaviour, generate_proximity, intersect
from traffic_flow_prediction.datasets import create_windows, prepare_series, split_time, to_edges

WEEKDAY = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday", 5: "saturday", 6: "sunday"}
TFS = np.asarray([f"{i:02d}-{i+2:02d}" for i in range(0, 24, 2)])
fprint = partial(print, flush=True)


def fprint_tree(tree: Any, name: str = "tree", shape: bool = True) -> None:
    if shape:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}.shape = {v.shape}"), tree)
    else:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}\n{v}"), tree)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara el indice PFM y grafos ajustados solo con entrenamiento.")
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--target", type=Path, default=Path("data/processed_pfm"))
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=.70)
    parser.add_argument("--frequency", default="15min")
    parser.add_argument("--max-fill-steps", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--negative-correlation-policy", choices=("error", "zero"), default="error",
                        help="zero aplica max(correlacion,0), conserva las conexiones y registra los pesos originales.")
    parser.add_argument("--clusterings", nargs="+", default=["base"],
                        choices=["base", "proximity_50", "proximity_500", "proximity_behaviour_50", "proximity_behaviour_500"])
    parser.add_argument("--adjacencies", nargs="+", default=["closeness"], choices=[x.name for x in ADJType])
    args = parser.parse_args()
    if args.target.exists():
        parser.error("El destino existe. Elige una carpeta nueva con --target; no se sobrescriben datasets.")
    if min(args.window, args.horizon, args.neighbors) < 1 or args.max_fill_steps < 0 or not 0 < args.train_ratio < 1:
        parser.error("Invalid window, horizon, neighbors, fill limit or split ratio.")
    return args


def read_traffic_csv(path_to_file: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_file, sep=";", usecols=["id", "fecha", "tipo_elem", "intensidad", "ocupacion"],
                       dtype={"id": np.int64, "tipo_elem": "category", "intensidad": np.float32, "ocupacion": np.float32},
                       parse_dates=["fecha"]).rename(columns={"fecha": "date", "tipo_elem": "type", "intensidad": "intensity", "ocupacion": "occupation"})


def read_sensors_csv(path_to_file: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_file, sep=";", encoding="latin-1", usecols=["id", "distrito", "nombre", "utm_x", "utm_y", "longitud", "latitud"],
                       dtype={"id": np.int64, "distrito": "str", "nombre": "str", "utm_x": np.float32, "utm_y": np.float32,
                              "longitud": np.float32, "latitud": np.float32}).rename(
                                  columns={"distrito": "district", "nombre": "name", "longitud": "longitude", "latitud": "latitude"})


def fit_saturation(traffic: pd.DataFrame, train_until: pd.Timestamp) -> pd.Series:
    valid = (traffic["date"] <= train_until) & np.isfinite(traffic["intensity"]) & (traffic["intensity"] >= 0)
    return traffic.loc[valid].groupby("id", observed=True)["intensity"].quantile(.98)


def prepare_traffic(traffic: pd.DataFrame, saturation: pd.Series) -> pd.DataFrame:
    df = traffic.dropna(subset=["id", "date"]).copy()
    intensity, occupation = df["intensity"], df["occupation"]
    capacity = df["id"].map(saturation)
    valid = np.isfinite(intensity) & np.isfinite(occupation) & np.isfinite(capacity)
    valid &= (intensity >= 0) & occupation.between(0, 100) & (capacity > 0)
    y = (intensity.where(valid)/capacity.where(valid)).clip(0, 1)
    df["congestion"] = (y + (1-y)*occupation/100).where(valid).astype(np.float32)
    dt = df["date"].dt
    for name in ("year", "month", "day", "hour", "minute"):
        df[name] = getattr(dt, name)
    df["weekday"] = dt.dayofweek.map(WEEKDAY)
    df["daytype"] = np.where(dt.dayofweek < 5, "WD", "WE")
    df["TF"] = TFS[df["hour"]//2]
    df["daytype_TF"] = df["daytype"] + " " + df["TF"]
    for name in ("weekday", "daytype", "TF", "daytype_TF"):
        df[name] = df[name].astype("category")
    return df


def prepare_sensors(sensors: pd.DataFrame) -> pd.DataFrame:
    return sensors.replace([np.inf, -np.inf], np.nan).dropna(subset=["id", "utm_x", "utm_y"])


def load_data(path_to_source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    traffic_files = sorted((path_to_source / "raw/traffic").rglob("*.csv"))
    sensor_files = sorted((path_to_source / "raw/sensors").rglob("*.csv"))
    if not traffic_files or not sensor_files:
        raise ValueError("CSV files are required under raw/traffic and raw/sensors.")
    traffic = pd.concat([read_traffic_csv(path) for path in tqdm(traffic_files, desc="Loading traffic data")], ignore_index=True)
    sensors = pd.concat([prepare_sensors(read_sensors_csv(path)) for path in tqdm(sensor_files, desc="Loading sensors data")], ignore_index=True)
    sensors = sensors.drop_duplicates("id", keep="last")
    traffic = traffic.dropna(subset=["id", "date"])
    traffic = traffic.loc[traffic["id"].isin(sensors["id"])]
    if traffic.empty:
        raise ValueError("No traffic observations have valid sensor coordinates.")
    return traffic, sensors


def build_clusters(traffic: pd.DataFrame, sensors: pd.DataFrame, cluster_targets: Collection[int] = (50, 500), requested=None) -> dict[str, pd.DataFrame]:
    names = set(requested) if requested is not None else {"base", "proximity_50", "proximity_500", "proximity_behaviour_50", "proximity_behaviour_500"}
    clusters = {}
    if "base" in names:
        clusters["base"] = generate_base(sensors)
    behaviour = generate_behaviour(traffic, 2) if any(name.startswith("proximity_behaviour") for name in names) else None
    for target in cluster_targets:
        proximity_name, behaviour_name = f"proximity_{target}", f"proximity_behaviour_{target}"
        if proximity_name in names or behaviour_name in names:
            proximity = generate_proximity(sensors, target)
            if proximity_name in names:
                clusters[proximity_name] = proximity
            if behaviour_name in names:
                clusters[behaviour_name] = intersect(proximity, behaviour)
    return clusters


def get_cluster_time_matrix(cluster: pd.DataFrame, cl_type: CLType, traffic: pd.DataFrame, *, fill_missing=True) -> pd.DataFrame:
    merged = traffic.merge(cluster[["id", cl_type.value, "utm_x", "utm_y"]], on="id", how="inner")
    matrix = (merged.groupby([cl_type.value, "date"], observed=True)["congestion"].mean().unstack(cl_type.value).sort_index())
    matrix.index.name = "date"
    return matrix.interpolate(limit_direction="both") if fill_missing else matrix


def build_and_save_dataset(path_to_target, cluster, cl_name, traffic, *, times, train_end, valid_end,
                           window, horizon, train_ratio, max_fill_steps, adjacencies, neighbors, source_info, negative_correlation_policy="error"):
    cl_type = CLType[re.sub(r"_\d+$", "", cl_name)]
    raw = get_cluster_time_matrix(cluster, cl_type, traffic, fill_missing=False).reindex(times)
    raw.index.name = "date"
    raw = raw.loc[:, raw.iloc[:train_end].notna().any()]
    if raw.empty:
        raise ValueError(f"No observed training nodes for {cl_name}.")
    centroids = compute_centroids(cluster, cl_type)
    candidate_labels = raw.columns.tolist()
    adjacencies = list(dict.fromkeys(adjacencies))
    correlation_edges, correlation_labels = None, None
    if "correlation" in adjacencies:
        fprint(f"Fitting training correlations for {cl_name}...", flush=True)
        correlation_edges, correlation_labels = generate_correlation(raw.iloc[:train_end], neighbors, negative_policy=negative_correlation_policy)[1:]
        if "closeness" in adjacencies:
            raw = raw.loc[:, correlation_labels]
    for adjacency in adjacencies:
        fprint(f"Building {cl_name}/{adjacency}...", end=" ")
        if adjacency == "correlation":
            df_edges, labels = correlation_edges, correlation_labels
            matrix = raw.loc[:, labels].copy()
            matrix.columns = np.arange(len(labels))
        else:
            matrix, df_edges, labels = generate_closeness(raw, neighbors, centroids)
        data = prepare_series(matrix, window=window, horizon=horizon, train_end=train_end, valid_end=valid_end, max_fill_steps=max_fill_steps)
        data["node_labels"] = np.asarray(labels, dtype=np.int64)
        edges, receivers, senders = to_edges(df_edges)
        folder = path_to_target / cl_name / adjacency
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / "series.npz", **data)
        np.savez(folder / "graph.npz", edges=np.asarray(edges), receivers=np.asarray(receivers), senders=np.asarray(senders),
                 nodes_idx=np.arange(len(labels), dtype=np.int32), node_labels=np.asarray(labels, dtype=np.int64))
        mapping = {label: index for index, label in enumerate(labels)}
        members = cluster.loc[cluster[cl_type.value].isin(labels), ["id", cl_type.value, "name", "utm_x", "utm_y"]].drop_duplicates("id")
        members = members.rename(columns={"id": "sensor_id", cl_type.value: "node_label"})
        members["node_index"] = members["node_label"].map(mapping)
        members.sort_values(["node_index", "sensor_id"]).to_csv(folder / "node_map.csv", index=False)
        names = []
        for label in labels:
            ids = members.loc[members["node_label"] == label, "sensor_id"].tolist()
            names.append(f"sensor {ids[0]}" if cl_name == "base" else f"{cl_name} {label} ({len(ids)} sensores)")
        pd.DataFrame({"node_index": np.arange(len(labels)), "node_label": labels, "name": names}).to_csv(folder / "nodes.csv", index=False)
        df_edges.to_csv(folder / "df_edges.csv", index=False)
        metadata = dict(source_info, format_version=1, index="pfm", cluster=cl_name, adjacency=adjacency,
                        window=window, horizon=horizon, train_ratio=train_ratio, train_end=train_end, valid_end=valid_end,
                        max_fill_steps=max_fill_steps, neighbors=neighbors, n_nodes=len(labels),
                        n_sensors=int(members["sensor_id"].nunique()), sensors_before_node_filter=int(cluster["id"].nunique()),
                        n_timestamps=len(times), observed_fraction=float(data["observed"].mean()),
                        train_last_timestamp=str(times[train_end-1]), valid_last_timestamp=str(times[valid_end-1]),
                        split_samples={name: len(data[f"{name}_starts"]) for name in ("train", "valid", "test")},
                        removed_constant_node_labels=sorted(set(candidate_labels)-set(labels)),
                        paired_adjacencies=len(adjacencies) > 1,
                        comparison_node_policy="both graphs use the same training-variable nodes when generated together",
                        graph_direction="node i sends to its top-k neighbors j",
                        missing_inputs="forward fill up to max_fill_steps, then zero; no backward fill",
                        missing_targets="excluded by observation mask, including added timestamps",
                        correlation_fit="pairwise observed training values only",
                        negative_correlation_policy=negative_correlation_policy,
                        negative_weights_clipped=int((df_edges["raw_correlation"] < 0).sum()) if adjacency == "correlation" else 0,
                        scope="PFM index and implemented graph cases; not an exact reproduction of unspecified author settings")
        with (folder / "metadata.json").open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False, allow_nan=False)
        fprint(f"OK | nodos={len(labels)} | observados={metadata['observed_fraction']:.1%} | pesos negativos recortados={metadata['negative_weights_clipped']}")


def main() -> None:
    args = get_args()
    traffic, sensors = load_data(args.source)
    step = pd.Timedelta(args.frequency)
    if step <= pd.Timedelta(0):
        raise ValueError("Frequency must be positive.")
    times = pd.date_range(traffic["date"].min(), traffic["date"].max(), freq=step)
    if not traffic["date"].isin(times).all():
        raise ValueError("Some timestamps are not on the requested regular grid.")
    train_end = int(len(times)*args.train_ratio)
    valid_end = train_end+(len(times)-train_end)//2
    if not args.window+args.horizon <= train_end < valid_end < len(times):
        raise ValueError("Not enough data for the requested temporal splits.")
    saturation = fit_saturation(traffic, times[train_end-1])
    traffic = prepare_traffic(traffic, saturation)
    raw_ids = set(traffic["id"].unique().tolist())
    training = traffic.loc[traffic["date"] < times[train_end]]
    eligible = training.loc[training["congestion"].notna(), "id"].unique()
    sensors = sensors.loc[sensors["id"].isin(eligible)]
    traffic = traffic.loc[traffic["id"].isin(eligible)]
    training = training.loc[training["id"].isin(eligible)]
    if not len(eligible):
        raise ValueError("No sensors with a valid training congestion index.")
    source_info = {
        "source": str(args.source.resolve()), "frequency": str(step),
        "index_formula": "TO=occupation/100; Is_j=P98(intensity_j); Y=clip(intensity/Is_j,0,1); C=Y+(1-Y)*TO",
        "saturation_fit": "valid nonnegative training intensities only, separately for each sensor",
        "saturation_training_cutoff_inclusive": str(times[train_end-1]),
        "input_traffic_sensors": len(raw_ids), "eligible_training_sensors": len(eligible),
        "excluded_no_valid_training_index": sorted(raw_ids-set(eligible.tolist())),
        "inserted_timestamps": len(times)-traffic["date"].nunique(),
        "preprocessing_choices": "regular grid, causal fill, observed-target masks, 70/15/15 default split; details not fully specified in PFM",
        "invalid_measurements": "negative/nonfinite intensity, occupancy outside [0,100], missing/nonpositive saturation",
        "error_flag": "not used; validation currently checks values and ranges",
        "sensor_metadata": "last available row per sensor ID; provide a coordinate snapshot valid for the intended deployment",
    }
    clusters = build_clusters(training, sensors, requested=args.clusterings)
    args.target.mkdir(parents=True, exist_ok=False)
    saturation.rename("saturation_p98").to_csv(args.target / "saturation_p98.csv", index_label="sensor_id")
    fprint(f"Sensores con trafico: {len(raw_ids)} | Con indice de entrenamiento valido: {len(eligible)}")
    for name, cluster in clusters.items():
        build_and_save_dataset(args.target, cluster, name, traffic, times=times, train_end=train_end, valid_end=valid_end,
                               window=args.window, horizon=args.horizon, train_ratio=args.train_ratio,
                               max_fill_steps=args.max_fill_steps, adjacencies=args.adjacencies, neighbors=args.neighbors,
                               source_info=source_info, negative_correlation_policy=args.negative_correlation_policy)
    fprint(f"Datasets nuevos: {args.target}")


if __name__ == "__main__":
    main()
