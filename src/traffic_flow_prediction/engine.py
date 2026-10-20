import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax import nnx


def weight_decay(t_max: int, decay: float = 0.05) -> jax.Array:
    return jnp.exp(-decay*jnp.arange(t_max))


@nnx.jit
def loss_gradients(model, graph, target, mask, *, decay=0.05):
    weights = weight_decay(target.shape[1], decay).reshape(1, -1, 1)*mask
    count = jnp.sum(weights)

    def loss_fn(model):
        prediction = model(graph)
        error = jnp.where(mask, prediction, 0.)-jnp.where(mask, target, 0.)
        return jnp.sum(weights*error**2)/jnp.maximum(count, 1.)

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    return loss, count, gradients


@nnx.jit
def difference_gradients(model, graph, target, mask, *, decay=0.05):
    weights = weight_decay(target.shape[1], decay).reshape(1, -1, 1)
    pairs = mask[:, 1:] & mask[:, :-1]
    pair_weights = weights[:, :-1] * pairs
    count = jnp.sum(pair_weights)
    clean_target = jnp.where(mask, target, 0.)

    def loss_fn(model):
        prediction = model(graph)
        difference = jnp.diff(prediction, axis=1) - jnp.diff(clean_target, axis=1)
        return jnp.sum(pair_weights * jnp.where(pairs, difference, 0.)**2) / jnp.maximum(count, 1.)

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    return loss, count, gradients


@nnx.jit
def apply_gradients(optimiser, gradients):
    optimiser.update(gradients)


def train_step(model, optimiser, batches, *, decay=0.05, alpha=1.):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("time-alpha must be between 0 and 1.")
    accumulated = [None, None]
    totals = [0., 0.]
    counts = [0., 0.]
    for graph, target, mask in batches:
        if alpha < 1 and target.shape[1] < 2:
            raise ValueError("time-alpha below 1 requires at least two forecast steps.")
        for index, active in enumerate((alpha > 0, alpha < 1)):
            if not active:
                continue
            loss, count, gradients = (loss_gradients if index == 0 else difference_gradients)(model, graph, target, mask, decay=decay)
            value, weight = float(loss), float(count)
            if not np.isfinite(value):
                raise ValueError("Nonfinite training loss.")
            if weight == 0:
                continue
            gradients = jax.tree_util.tree_map(lambda g: g*weight, gradients)
            accumulated[index] = gradients if accumulated[index] is None else jax.tree_util.tree_map(lambda a, b: a+b, accumulated[index], gradients)
            totals[index] += value*weight
            counts[index] += weight
    combined = None
    for index, coefficient in enumerate((alpha, 1-alpha)):
        if accumulated[index] is None:
            continue
        gradients = jax.tree_util.tree_map(lambda g: g*coefficient/counts[index], accumulated[index])
        combined = gradients if combined is None else jax.tree_util.tree_map(lambda a, b: a+b, combined, gradients)
    if combined is not None:
        apply_gradients(optimiser, combined)
    loss = sum(coefficient*total/count if count else 0. for coefficient, total, count in zip((alpha, 1-alpha), totals, counts))
    return loss, counts[0] if alpha > 0 else counts[1]


@nnx.jit
def evaluate(model: nnx.Module, graph: jraph.GraphsTuple) -> jax.Array:
    return model(graph)


def baselines(x, horizon):
    return {
        "persistence": np.repeat(x[:, -1:], horizon, axis=1),
        "window_mean": np.repeat(x.mean(axis=1, keepdims=True), horizon, axis=1),
        "zero": np.zeros((len(x), horizon, x.shape[2], x.shape[3]), dtype=x.dtype),
    }


class ErrorTotals:
    def __init__(self, horizon):
        self.square = np.zeros(horizon, np.float64)
        self.absolute = np.zeros(horizon, np.float64)
        self.count = np.zeros(horizon, np.int64)

    def update(self, prediction, target, mask):
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("Prediction, target and observation mask shapes differ.")
        if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
            raise ValueError("Nonfinite prediction or observed target.")
        error = np.where(mask, prediction.astype(np.float64)-target, 0.)
        self.square += np.sum(error**2, axis=(0, 2, 3))
        self.absolute += np.sum(np.abs(error), axis=(0, 2, 3))
        self.count += np.sum(mask, axis=(0, 2, 3))

    def compute(self):
        def score(square, absolute, count):
            mse = float(square/count) if count else None
            return {"mse": mse, "rmse": float(np.sqrt(mse)) if mse is not None else None,
                    "mae": float(absolute/count) if count else None, "count": int(count)}
        result = score(self.square.sum(), self.absolute.sum(), self.count.sum())
        result["per_horizon"] = [dict(step=i+1, **score(s, a, n)) for i, (s, a, n) in enumerate(zip(self.square, self.absolute, self.count))]
        return result
