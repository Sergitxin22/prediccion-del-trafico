import argparse
import csv
import json
from pathlib import Path

import numpy as np


# c4 is the 50-node proximity+behaviour grouping, as in the requested notation.
CASES = (
    ("Ba1", "base", "closeness"),
    ("Ba2", "base", "correlation"),
    ("c1a1", "proximity_50", "closeness"),
    ("c1a2", "proximity_50", "correlation"),
    ("c2a1", "proximity_500", "closeness"),
    ("c2a2", "proximity_500", "correlation"),
    ("c3a1", "proximity_behaviour_500", "closeness"),
    ("c3a2", "proximity_behaviour_500", "correlation"),
    ("c4a1", "proximity_behaviour_50", "closeness"),
    ("c4a2", "proximity_behaviour_50", "correlation"),
)
SETTINGS = ("architecture", "model", "time_alpha", "layers", "hidden_feats")
FIELDS = (*SETTINGS, *(label for label, _, _ in CASES))


def sensor_nodes(folder):
    with (folder / "node_map.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {int(row["sensor_id"]): int(row["node_index"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sensor ID in {folder}")
    return result


def grouped_stats(values, observed, starts, sensor_columns, group_columns, n_nodes):
    """Sufficient statistics for squared error against individual sensor targets."""
    count = np.zeros((len(starts), 4, n_nodes), dtype=np.int32)
    total = np.zeros_like(count, dtype=np.float64)
    squares = np.zeros_like(total)
    flat = np.arange(len(starts))[:, None] * n_nodes + group_columns[None, :]
    for horizon in range(4):
        target = values[starts + horizon][:, sensor_columns].astype(np.float64)
        mask = observed[starts + horizon][:, sensor_columns]
        bins = flat[mask]
        count[:, horizon] = np.bincount(bins, minlength=len(starts) * n_nodes).reshape(len(starts), n_nodes)
        total[:, horizon] = np.bincount(bins, weights=target[mask], minlength=len(starts) * n_nodes).reshape(len(starts), n_nodes)
        squares[:, horizon] = np.bincount(bins, weights=target[mask] ** 2, minlength=len(starts) * n_nodes).reshape(len(starts), n_nodes)
    return count, total, squares


def squared_error(prediction, stats):
    count, total, squares = stats
    if prediction.shape != count.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape or finite values do not match the dataset")
    prediction = np.asarray(prediction, dtype=np.float64)
    error = np.sum(count * prediction ** 2 - 2 * total * prediction + squares, dtype=np.float64)
    return max(0.0, float(error)) / int(count.sum())


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(run, output, horizon="all"):
    if horizon not in ("all", "1", "2", "3", "4"):
        raise ValueError("Expected all four horizons or a step from 1 to 4")
    selected = slice(None) if horizon == "all" else slice(int(horizon) - 1, int(horizon))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    evaluation = run / "curves_precomputed_v1"
    progress = json.loads((evaluation / "manifest.json").read_text(encoding="utf-8"))
    jobs = [row for row in summary["results"] if row["kind"] == "candidate"]
    if manifest["state"] != "completed" or progress["state"] != "completed" or progress.get("failed") or len(progress["completed"]) != len(jobs):
        raise ValueError("All candidate evaluations must finish successfully before comparison")
    datasets = manifest["datasets"]
    base = Path(datasets["base/closeness"]["path"])
    with np.load(base / "series.npz", allow_pickle=False) as series:
        values, observed, times, starts = (series[key].copy() for key in ("values", "observed", "times", "test_starts"))
    if len(starts) == 0 or any(datasets[f"{group}/closeness"]["signal_signature"] != datasets[f"{group}/correlation"]["signal_signature"]
                               for group in {group for _, group, _ in CASES}):
        raise ValueError("Unpaired graphs or empty test split")
    maps = {group: sensor_nodes(Path(datasets[f"{group}/closeness"]["path"]))
            for group in {group for _, group, _ in CASES}}
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    if not common:
        raise ValueError("No sensors shared by every grouping")
    base_columns = np.array([maps["base"][sensor] for sensor in common], dtype=np.int32)
    if np.any(base_columns >= values.shape[1]):
        raise ValueError("Base node map does not match the base dataset")
    stats, naive_mse = {}, {}
    for group, mapping in maps.items():
        data = datasets[f"{group}/closeness"]
        with np.load(Path(data["path"]) / "series.npz", allow_pickle=False) as series:
            if not np.array_equal(series["times"], times) or not np.array_equal(series["test_starts"], starts):
                raise ValueError(f"Test timestamps differ for {group}")
            last = series["values"][starts - 1].copy()
        columns = np.array([mapping[sensor] for sensor in common], dtype=np.int32)
        if np.any(columns >= data["n_nodes"]):
            raise ValueError(f"Node map does not match the dataset for {group}")
        stats[group] = tuple(array[:, selected, :] for array in
                             grouped_stats(values, observed, starts, base_columns, columns, data["n_nodes"]))
        if last.shape[1] != data["n_nodes"]:
            raise ValueError(f"Inputs do not match the dataset for {group}")
        naive_mse[group] = squared_error(np.broadcast_to(last[:, None, :], stats[group][0].shape), stats[group])
    reference = np.zeros(stats["base"][0].shape, dtype=np.float32)
    reference[:, :, base_columns] = values[starts - 1][:, base_columns, None].transpose(0, 2, 1)
    persistence_mse = squared_error(reference, stats["base"])
    if persistence_mse <= 0:
        raise ValueError("Persistence MSE is zero; cannot compute relative error")

    entries = {}
    for job in jobs:
        group, graph = job["cluster"], job["adjacency"]
        label = next((key for key, g, a in CASES if (g, a) == (group, graph)), None)
        if label is None or job["status"] != "completed":
            raise ValueError("Unexpected or incomplete search candidate")
        key = tuple(job[field] for field in SETTINGS)
        row = entries.setdefault(key, dict(zip(SETTINGS, key)))
        if label in row:
            raise ValueError(f"Duplicate result for {key}/{label}")
        metric_path = evaluation / "metrics" / f"{job['id']}.json"
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        if metric["job_id"] != job["id"] or metric["dataset_signature"] != datasets[f"{group}/{graph}"]["signature"]:
            raise ValueError(f"Metric belongs to another checkpoint: {job['id']}")
        prediction = np.load(evaluation / "predictions" / f"{job['id']}.npy", mmap_mode="r", allow_pickle=False)
        if prediction.ndim != 3 or prediction.shape[1] != 4:
            raise ValueError(f"Expected four saved horizons for {job['id']}")
        row[label] = squared_error(prediction[:, selected, :], stats[group])

    if any(set(row) != set(FIELDS) for row in entries.values()):
        raise ValueError("A configuration does not have all ten cases")
    raw = [entries[key] for key in sorted(entries)]
    ratio = [{field: (row[field] / persistence_mse if field not in SETTINGS else row[field])
              for field in FIELDS} for row in raw]
    contrasts = [(label, "Ba1" if graph == "closeness" else "Ba2")
                 for label, group, graph in CASES if group != "base"]
    contrasts += [(f"{prefix}a2", f"{prefix}a1") for prefix in ("B", "c1", "c2", "c3", "c4")]
    effects = []
    for label, baseline in contrasts:
        for family in (None, *(sorted({(row["architecture"], row["model"]) for row in ratio}))):
            selected = [row for row in ratio if family is None or (row["architecture"], row["model"]) == family]
            deltas = np.array([row[baseline] - row[label] for row in selected])
            group = next(group for case, group, _ in CASES if case == label)
            effects.append({"case": label, "versus": baseline, "architecture": family[0] if family else "all",
                            "model": family[1] if family else "all", "configurations": len(selected),
                            "improved": int((deltas > 0).sum()), "fraction_improved": float(np.mean(deltas > 0)),
                            "median_relative_mse_gain": float(np.median(deltas)),
                            "beats_persistence": int(sum(row[label] < 1 for row in selected)),
                            "beats_own_group_persistence": int(sum(row[label] < naive_mse[group] / persistence_mse for row in selected))})

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "paired_mse.csv", raw, FIELDS)
    write_csv(output / "paired_mse_ratio.csv", ratio, FIELDS)
    write_csv(output / "paired_effects.csv", effects, tuple(effects[0]))
    write_csv(output / "paired_baselines.csv",
              [{"case": label, "group": group, "graph": graph, "naive_mse": naive_mse[group],
                "ratio_to_individual_persistence": naive_mse[group] / persistence_mse} for label, group, graph in CASES],
              ("case", "group", "graph", "naive_mse", "ratio_to_individual_persistence"))
    (output / "metadata.json").write_text(json.dumps({
        "source": str(run.resolve()), "evaluation": str(evaluation.resolve()),
        "split": "test", "horizons": [1, 2, 3, 4] if horizon == "all" else [int(horizon)],
        "common_sensors": len(common),
        "evaluated_targets": int(stats["base"][0].sum()), "persistence_mse": persistence_mse,
        "cases": {label: {"group": group, "graph": graph, "nodes": datasets[f"{group}/{graph}"]["n_nodes"]}
                  for label, group, graph in CASES},
        "interpretation": "All forecasts are compared on identical observed individual sensor targets and timestamps. "
                          "For grouped forecasts, the group prediction is assigned to each member sensor. "
                          "paired_mse_ratio uses the same individual-sensor persistence MSE for all cases; lower is better. "
                          "paired_baselines compares the last known group value against the same sensor targets. "
                          "This evaluates transfer from an aggregate to sensors, not accuracy of the aggregate itself. "
                          "Positive median_relative_mse_gain means lower error than versus. "
                          "Counts across hyperparameters are descriptive, not independent statistical trials."
    }, indent=2), encoding="utf-8")
    return len(raw), len(common)


def main():
    parser = argparse.ArgumentParser(description="Compare saved forecasts on identical observed individual sensors; no training or GPUs.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", choices=("all", "1", "2", "3", "4"), default="all",
                        help="all pools four future steps; 1=+15, 2=+30, 3=+45, 4=+60 minutes")
    args = parser.parse_args()
    count, sensors = compare(args.results_dir, args.output, horizon=args.horizon)
    print(f"Compared {count} configurations across ten cases on {sensors} common sensors: {args.output}")


if __name__ == "__main__":
    main()
