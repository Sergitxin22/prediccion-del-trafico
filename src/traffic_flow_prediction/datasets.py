from collections.abc import Iterator
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import pandas as pd
from flax import nnx
from jax.typing import ArrayLike


class DataLoader:
    
    def __init__(self, x: ArrayLike, y: ArrayLike, batch_size: int, shuffle: bool = False, rngs: nnx.Rngs | None = None) -> None:
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples.")

        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rngs = rngs

    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        n = len(self._x)

        if self._shuffle:
            if self._rngs is None:
                raise ValueError("A PRNG key must be provided when shuffle=True.")
            indices = jrand.permutation(self._rngs(), n)
        else:
            indices = jnp.arange(n)

        for start in range(0, n, self._batch_size):
            batch_idx = indices[start : start + self._batch_size]
            yield self._x[batch_idx], self._y[batch_idx]

    def __len__(self) -> int:
        return (len(self._x) + self._batch_size - 1) // self._batch_size


def to_edges(df_edges: pd.DataFrame) -> tuple[jax.Array, jax.Array, jax.Array]:
    m = len(df_edges)
    if "weight" in df_edges.columns:
        edges = jnp.asarray(df_edges["weight"].values, dtype=jnp.float32)[:, jnp.newaxis]
    else:
        edges = jnp.ones((m, 1), dtype=jnp.float32)
    receivers = jnp.asarray(df_edges["target"].values, dtype=jnp.int32)
    senders   = jnp.asarray(df_edges["source"].values, dtype=jnp.int32)
    return edges, receivers, senders


def create_windows(data: ArrayLike, window: int = 32, horizon: int = 1) -> tuple[jax.Array, jax.Array]:
    x = []
    y = []
    n_time = data.shape[0]
    for t in range(n_time - window - horizon + 1):
        x_t = data[t : t + window]
        y_t = data[t + window : t + window + horizon]
        x.append(x_t[..., jnp.newaxis])
        y.append(y_t[..., jnp.newaxis])
    if not x:
        raise ValueError(f"There is no sufficient time iteration ({n_time}) for windows={window} and horizon={horizon}.")
    return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)


def split_time(
    x: ArrayLike,
    y: ArrayLike,
    train_ratio: float = 0.70
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n = len(x)
    n_train = int(train_ratio * n)
    n_valid = (n - n_train) // 2
    return (
        x[                  : n_train          ], y[                  : n_train          ],
        x[n_train           : n_train + n_valid], y[n_train           : n_train + n_valid],
        x[n_train + n_valid :                  ], y[n_train + n_valid :                  ],
    )


def prepare_series(matrix, *, window, horizon, train_end, valid_end, max_fill_steps=4):
    if min(window, horizon) < 1 or max_fill_steps < 0:
        raise ValueError("Invalid window, horizon or fill limit.")
    if not window+horizon <= train_end < valid_end < len(matrix):
        raise ValueError("Not enough data for chronological train, validation and test splits.")
    if min(valid_end-train_end, len(matrix)-valid_end) < horizon:
        raise ValueError("The forecast horizon does not fit inside each split.")
    if not matrix.index.is_unique or not matrix.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing.")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    if not matrix.iloc[:train_end].notna().any().all():
        raise ValueError("Every node must have observed training data.")
    observed = matrix.notna().to_numpy()
    inputs = matrix.ffill(limit=max_fill_steps) if max_fill_steps else matrix
    data = {
        "values": inputs.fillna(0.).to_numpy(dtype=np.float32),
        "observed": observed,
        "times": matrix.index.to_numpy(dtype="datetime64[ns]"),
        "node_labels": matrix.columns.to_numpy(dtype=np.int64),
        "window": np.array(window), "horizon": np.array(horizon),
        "train_starts": np.arange(window, train_end-horizon+1, dtype=np.int32),
        "valid_starts": np.arange(train_end, valid_end-horizon+1, dtype=np.int32),
        "test_starts": np.arange(valid_end, len(matrix)-horizon+1, dtype=np.int32),
    }
    for split in ("train", "valid", "test"):
        starts = data[f"{split}_starts"]
        keep = observed[starts[:, None]+np.arange(horizon)].any(axis=(1, 2))
        data[f"{split}_starts"] = starts[keep]
        if not keep.any():
            raise ValueError(f"No observed targets in {split}.")
    return data


class WindowDataLoader:
    def __init__(self, data, split, batch_size, *, shuffle=False, seed=42):
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        if shuffle and split != "train":
            raise ValueError("Validation and test must remain chronological.")
        self.data = data
        self.starts = data[f"{split}_starts"]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.starts)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        starts = self.rng.permutation(self.starts) if self.shuffle else self.starts
        window, horizon = int(self.data["window"]), int(self.data["horizon"])
        for offset in range(0, len(starts), self.batch_size):
            current = starts[offset:offset+self.batch_size]
            past = current[:, None]+np.arange(-window, 0)
            future = current[:, None]+np.arange(horizon)
            yield (self.data["values"][past, :, None], self.data["values"][future, :, None],
                   self.data["observed"][future, :, None], current)


def save_checkpoint(path: Path, model):
    entries, _ = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    arrays = {f"param_{i}": np.asarray(value) for i, (_, value) in enumerate(entries)}
    arrays["parameter_paths"] = np.asarray([jax.tree_util.keystr(key) for key, _ in entries])
    with path.open("xb") as file:
        np.savez(file, **arrays)


def load_checkpoint(path: Path, model):
    entries, structure = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    with np.load(path, allow_pickle=False) as saved:
        paths = [jax.tree_util.keystr(key) for key, _ in entries]
        if saved["parameter_paths"].tolist() != paths:
            raise ValueError("Checkpoint architecture does not match this model.")
        leaves = [saved[f"param_{i}"] for i in range(len(entries))]
        if any(array.shape != value.shape for array, (_, value) in zip(leaves, entries)):
            raise ValueError("Checkpoint dimensions do not match this model.")
    nnx.update(model, jax.tree_util.tree_unflatten(structure, [jnp.asarray(x) for x in leaves]))
