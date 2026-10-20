import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pandas as pd
import optax
from flax import nnx

import generate_datasets as generator
from train import evaluate_split, load_data, micro_batches, prepare_data
from traffic_flow_prediction.adjacency import generate_correlation, knn_from_centroids
from traffic_flow_prediction.datasets import WindowDataLoader, load_checkpoint, prepare_series, save_checkpoint
from traffic_flow_prediction.engine import ErrorTotals, baselines, train_step
from traffic_flow_prediction.models import GATLayer, GCNLayer, GNN, SpatialType, TemporalType


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.precision = jax.default_matmul_precision("highest")
        cls.precision.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.precision.__exit__(None, None, None)

    def test_batch_connections_cover_every_sample(self):
        x = np.arange(12, dtype=np.float32).reshape(2, 3, 2, 1)
        y = np.zeros((2, 1, 2, 1), np.float32)
        graph_data = {"nodes_idx": np.array([0, 1]), "edges": np.array([[.2], [.8]], np.float32),
                      "senders": np.array([0, 1], np.int32), "receivers": np.array([1, 0], np.int32)}
        graph, target = prepare_data(x, y, graph_data)
        np.testing.assert_array_equal(graph.senders, [0, 1, 2, 3])
        np.testing.assert_array_equal(graph.receivers, [1, 0, 3, 2])
        np.testing.assert_allclose(graph.edges, [[.2], [.8], [.2], [.8]])
        self.assertEqual(target.shape, (4, 1, 1))

    def test_gcn_uses_edge_weights(self):
        graph = jraph.GraphsTuple(nodes=jnp.array([[2.], [8.]]), edges=jnp.array([[1.]]),
                                 senders=jnp.array([0]), receivers=jnp.array([1]), globals=None,
                                 n_node=jnp.array([2]), n_edge=jnp.array([1]))
        layer = GCNLayer(1, 1, rngs=nnx.Rngs(42))
        layer.linear.kernel.value = jnp.ones((1, 1))
        original = np.asarray(layer(graph).nodes)
        changed = np.asarray(layer(graph._replace(edges=jnp.array([[3.]]))).nodes)
        self.assertGreater(float(np.max(np.abs(original-changed))), .1)

    def test_spatial_features_and_temporal_readout_match_pfm(self):
        model = GNN(1, 4, 1, 3, 1, 2, rngs=nnx.Rngs(42))
        self.assertEqual(model.layers[2].linear.kernel.shape, (4, 4))
        self.assertEqual(model.proj.kernel.shape, (4, 1))
        for carry in jax.tree_util.tree_leaves(model.carry):
            self.assertEqual(carry.shape[-1], 4)

    def test_correlation_remaps_and_excludes_constant_nodes(self):
        matrix = pd.DataFrame({10: [1., 2., 3.], 30: [0., 0., 0.], 99: [2., 4., 6.]})
        output, edges, labels = generate_correlation(matrix, 1)
        self.assertEqual(labels, [10, 99])
        self.assertEqual(output.shape[1], 2)
        np.testing.assert_array_equal(edges["source"], [0, 1])
        np.testing.assert_array_equal(edges["target"], [1, 0])

    def test_gat_preserves_own_signal_for_isolated_nodes(self):
        graph = jraph.GraphsTuple(nodes=jnp.array([[2.]]), edges=jnp.empty((0, 1)),
                                 senders=jnp.array([], dtype=jnp.int32), receivers=jnp.array([], dtype=jnp.int32),
                                 globals=None, n_node=jnp.array([1]), n_edge=jnp.array([0]))
        layer = GATLayer(1, 1, rngs=nnx.Rngs(42))
        layer.linear.kernel.value = jnp.ones((1, 1))
        np.testing.assert_allclose(layer(graph).nodes, [[2.]], atol=1e-6)

    def graph_data(self):
        return {"nodes_idx": np.arange(2), "edges": np.array([[.2], [.8]], np.float32),
                "senders": np.array([0, 1], np.int32), "receivers": np.array([1, 0], np.int32)}

    def test_pfm_index_zero_intensity_and_invalid_values(self):
        frame = pd.DataFrame({"id": [1]*5, "date": pd.date_range("2025-01-01", periods=5, freq="15min"),
                              "intensity": [500., 0., 2000., -1., 10.], "occupation": [20., 20., 20., 20., 101.]})
        result = generator.prepare_traffic(frame, pd.Series({1: 1000.}))
        np.testing.assert_allclose(result["congestion"][:3], [.6, .2, 1.])
        self.assertTrue(result["congestion"][3:].isna().all())

    def test_saturation_is_per_sensor_and_excludes_future(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="15min")
        frame = pd.DataFrame({"id": [10]*4+[90]*4, "date": list(dates)*2,
                              "intensity": [10., 20., 30., 10000., 100., 200., 300., 100000.]})
        capacity = generator.fit_saturation(frame, dates[2])
        np.testing.assert_allclose(capacity.loc[[10, 90]], [29.6, 296.])
        frame.loc[frame["date"] > dates[2], "intensity"] = 1e9
        pd.testing.assert_series_equal(capacity, generator.fit_saturation(frame, dates[2]))

    def test_causal_fill_masks_and_disjoint_target_periods(self):
        matrix = pd.DataFrame({10: np.arange(20)/20, 90: np.arange(20)/25}, index=pd.date_range("2025-01-01", periods=20, freq="15min"))
        matrix.iloc[0, 0] = np.nan
        matrix.iloc[5:8, 0] = np.nan
        data = prepare_series(matrix, window=3, horizon=2, train_end=12, valid_end=16, max_fill_steps=1)
        self.assertEqual(data["values"][0, 0], 0)
        self.assertEqual(data["values"][5, 0], data["values"][4, 0])
        np.testing.assert_array_equal(data["values"][6:8, 0], 0)
        self.assertFalse(data["observed"][5:8, 0].any())
        self.assertLess(data["train_starts"][-1]+1, data["valid_starts"][0])
        self.assertLess(data["valid_starts"][-1]+1, data["test_starts"][0])
        x, y, mask, starts = next(iter(WindowDataLoader(data, "valid", 2)))
        np.testing.assert_array_equal(x[0, :, :, 0], data["values"][starts[0]-3:starts[0]])
        self.assertEqual(y.shape, mask.shape)
        with self.assertRaises(ValueError):
            WindowDataLoader(data, "test", 2, shuffle=True)

    def test_future_values_cannot_change_filled_training_inputs(self):
        matrix = pd.DataFrame({10: np.arange(20)/20}, index=pd.date_range("2025-01-01", periods=20, freq="15min"))
        matrix.iloc[9:12] = np.nan
        a = prepare_series(matrix, window=3, horizon=1, train_end=12, valid_end=16)
        matrix.iloc[12:] = 1000.
        b = prepare_series(matrix, window=3, horizon=1, train_end=12, valid_end=16)
        np.testing.assert_array_equal(a["values"][:12], b["values"][:12])

    def test_closeness_excludes_actual_self_even_with_duplicate_coordinates(self):
        centroids = pd.DataFrame({"cluster": [0, 1, 2], "utm_x": [0., 0., 10.], "utm_y": [0., 0., 0.]})
        edges = knn_from_centroids(centroids, 1)
        self.assertFalse((edges["source"] == edges["target"]).any())
        np.testing.assert_array_equal(edges.groupby("source").size().to_numpy(), 1)

    def test_negative_selected_correlations_are_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError, "Negative"):
            generate_correlation(pd.DataFrame({10: [1., 2., 3.], 90: [3., 2., 1.]}), 1)

    def test_zero_policy_preserves_negative_connections_and_raw_correlations(self):
        _, edges, labels = generate_correlation(pd.DataFrame({10: [1., 2., 3.], 90: [3., 2., 1.]}), 1, negative_policy="zero")
        self.assertEqual(labels, [10, 90])
        np.testing.assert_array_equal(edges["source"], [0, 1])
        np.testing.assert_array_equal(edges["target"], [1, 0])
        np.testing.assert_array_equal(edges["weight"], 0.)
        np.testing.assert_allclose(edges["raw_correlation"], -1.)

    def test_negative_weight_policy_is_saved_in_dataset(self):
        times = pd.date_range("2025-01-01", periods=20, freq="15min")
        values = np.arange(20)/20
        traffic = pd.DataFrame({"id": [101]*20+[303]*20, "date": list(times)*2,
                                "congestion": np.concatenate([values, 1-values])})
        sensors = pd.DataFrame({"id": [101, 303], "name": ["a", "b"], "utm_x": [0., 1.], "utm_y": [0., 0.]})
        cluster = generator.generate_base(sensors)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator.build_and_save_dataset(root, cluster, "base", traffic, times=times, train_end=12, valid_end=16,
                                             window=3, horizon=1, train_ratio=.6, max_fill_steps=1, adjacencies=["correlation"],
                                             neighbors=1, source_info={"frequency": "15min"}, negative_correlation_policy="zero")
            graph, _, metadata = load_data(root / "base/correlation")
            self.assertEqual(metadata["negative_weights_clipped"], 2)
            self.assertEqual(metadata["negative_correlation_policy"], "zero")
            np.testing.assert_array_equal(graph["edges"], 0.)

    def test_gcn_matches_weighted_normalization_and_bias_after_aggregation(self):
        graph = jraph.GraphsTuple(nodes=jnp.array([[2.], [8.]]), edges=jnp.array([[3.]]),
                                 senders=jnp.array([0]), receivers=jnp.array([1]), globals=None,
                                 n_node=jnp.array([2]), n_edge=jnp.array([1]))
        layer = GCNLayer(1, 1, rngs=nnx.Rngs(42))
        layer.linear.kernel.value = jnp.ones((1, 1))
        layer.bias.value = jnp.array([.5])
        np.testing.assert_allclose(layer(graph).nodes[:, 0], [2.5, 5.5], atol=1e-6)

    def test_last_recurrent_state_not_all_states_is_projected(self):
        x = np.arange(6, dtype=np.float32).reshape(1, 3, 2, 1)/10
        graph, _ = prepare_data(x, np.zeros((1, 1, 2, 1), np.float32), self.graph_data())
        model = GNN(1, 4, 1, 3, 1, 2, temporal_hidden=6, rngs=nnx.Rngs(42))
        sequence = []
        for t in range(3):
            current = graph._replace(nodes=graph.nodes[:, t])
            for layer in model.layers:
                current = layer(current)
            sequence.append(current.nodes)
        expected = []
        for node in range(2):
            carry = model.carry
            for t in range(3):
                carry, hidden = model.lstm(carry, sequence[t][node])
            expected.append(model.proj(hidden))
        np.testing.assert_allclose(model(graph)[:, 0], jnp.stack(expected), atol=1e-6)

    def test_all_model_combinations_are_batch_independent(self):
        x = np.random.default_rng(17).uniform(size=(2, 3, 2, 1)).astype(np.float32)
        y = np.full((2, 2, 2, 1), .7, np.float32)
        graph, _ = prepare_data(x, y, self.graph_data())
        single, _ = prepare_data(x[1:], y[1:], self.graph_data())
        for spatial in SpatialType:
            for temporal in TemporalType:
                model = GNN(1, 4, 1, 3, 2, 2, spatial=spatial, temporal=temporal, rngs=nnx.Rngs(42))
                np.testing.assert_allclose(model(graph)[2:], model(single), atol=1e-6, rtol=1e-6)
                before = np.asarray(model(graph)).copy()
                optimizer = nnx.Optimizer(model, optax.adam(1e-4))
                loss, count = train_step(model, optimizer, micro_batches(x, y, np.ones_like(y, bool), self.graph_data(), 1))
                self.assertTrue(np.isfinite(loss))
                self.assertGreater(count, 0)
                self.assertGreater(float(np.max(np.abs(np.asarray(model(graph))-before))), 0.)

    def test_gat_does_not_use_external_edge_weights(self):
        graph, _ = prepare_data(np.ones((1, 1, 2, 1), np.float32), np.zeros((1, 1, 2, 1), np.float32), self.graph_data())
        graph = graph._replace(nodes=graph.nodes[:, 0])
        layer = GATLayer(1, 3, rngs=nnx.Rngs(42))
        np.testing.assert_allclose(layer(graph).nodes, layer(graph._replace(edges=graph.edges*10)).nodes, atol=1e-7)

    def test_gradient_accumulation_matches_full_batch_with_unequal_masks(self):
        x = np.random.default_rng(3).uniform(.1, .9, size=(3, 3, 2, 1)).astype(np.float32)
        y = np.array([.1, .2, .8, .7, .3, .6], np.float32).reshape(3, 1, 2, 1)
        mask = np.ones_like(y, bool)
        mask[1, 0, 1] = False
        mask[2] = False
        models = [GNN(1, 4, 1, 3, 1, 2, rngs=nnx.Rngs(42)) for _ in range(2)]
        losses = []
        for model, size in zip(models, (3, 1)):
            optimizer = nnx.Optimizer(model, optax.adam(1e-4))
            loss, count = train_step(model, optimizer, micro_batches(x, y, mask, self.graph_data(), size))
            self.assertEqual(count, 3)
            losses.append(loss)
        self.assertAlmostEqual(losses[0], losses[1], places=6)
        for a, b in zip(jax.tree_util.tree_leaves(nnx.state(models[0], nnx.Param)), jax.tree_util.tree_leaves(nnx.state(models[1], nnx.Param))):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)

    def test_all_missing_targets_skip_optimizer_update(self):
        x = np.ones((1, 3, 2, 1), np.float32)
        y = np.ones((1, 1, 2, 1), np.float32)
        model = GNN(1, 4, 1, 3, 1, 2, rngs=nnx.Rngs(42))
        optimizer = nnx.Optimizer(model, optax.adam(1e-4))
        train_step(model, optimizer, micro_batches(x, y, np.ones_like(y, bool), self.graph_data(), 1))
        before = [np.asarray(value).copy() for value in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))]
        y[:] = np.nan
        loss, count = train_step(model, optimizer, micro_batches(x, y, np.zeros_like(y, bool), self.graph_data(), 1))
        self.assertEqual((loss, count), (0., 0.))
        for a, b in zip(before, jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))):
            np.testing.assert_array_equal(a, b)

    def test_metrics_count_observations_not_batches_and_cover_all_horizons(self):
        prediction = np.array([[1., 10.], [2., 20.], [3., 30.]]).reshape(3, 2, 1, 1)
        target = np.zeros_like(prediction)
        mask = np.ones_like(prediction, bool)
        mask[2, 1] = False
        totals = ErrorTotals(2)
        totals.update(prediction[:2], target[:2], mask[:2])
        totals.update(prediction[2:], target[2:], mask[2:])
        score = totals.compute()
        self.assertAlmostEqual(score["mse"], (1+100+4+400+9)/5)
        self.assertEqual(score["count"], 5)
        self.assertEqual(score["per_horizon"][1]["count"], 2)
        self.assertAlmostEqual(score["rmse"]**2, score["mse"])

    def test_baselines_cover_each_horizon(self):
        x = np.array([1., 2., 6.]).reshape(1, 3, 1, 1)
        output = baselines(x, 2)
        np.testing.assert_array_equal(output["persistence"].ravel(), [6, 6])
        np.testing.assert_array_equal(output["window_mean"].ravel(), [3, 3])

    def test_checkpoint_round_trip_and_no_overwrite(self):
        model = GNN(1, 4, 1, 3, 1, 2, rngs=nnx.Rngs(42))
        restored = GNN(1, 4, 1, 3, 1, 2, rngs=nnx.Rngs(17))
        graph, _ = prepare_data(np.ones((1, 3, 2, 1), np.float32), np.zeros((1, 1, 2, 1), np.float32), self.graph_data())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(path, model)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model)
            load_checkpoint(path, restored)
        np.testing.assert_allclose(model(graph), restored(graph), atol=1e-7)

    def test_generator_rejects_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["generate_datasets.py", "--target", directory]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    generator.get_args()
                self.assertEqual(raised.exception.code, 2)

    def test_dataset_round_trip_and_training_only_correlation(self):
        times = pd.date_range("2025-01-01", periods=40, freq="15min")
        traffic = pd.DataFrame({"id": [1002]*40+[7000]*40+[9000]*40, "date": list(times)*3,
                                "intensity": list(np.arange(1, 41))+[100.]*40+list(np.arange(1, 41)*2), "occupation": 20.})
        traffic = generator.prepare_traffic(traffic, generator.fit_saturation(traffic, times[27]))
        sensors = pd.DataFrame({"id": [1002, 7000, 9000], "name": ["a", "b", "c"], "utm_x": [0., 10., 20.], "utm_y": [0., 0., 0.]})
        cluster = generator.generate_base(sensors)
        with tempfile.TemporaryDirectory() as directory:
            graphs = []
            for iteration in range(2):
                root = Path(directory) / str(iteration)
                generator.build_and_save_dataset(root, cluster, "base", traffic, times=times, train_end=28, valid_end=34,
                                                 window=3, horizon=2, train_ratio=.7, max_fill_steps=1,
                                                 adjacencies=["closeness", "correlation"], neighbors=1, source_info={"frequency": "15min"})
                graph, data, metadata = load_data(root / "base/correlation")
                _, close_data, close_metadata = load_data(root / "base/closeness")
                self.assertTrue(close_metadata["paired_adjacencies"])
                for key in data:
                    np.testing.assert_array_equal(data[key], close_data[key])
                self.assertEqual(metadata["n_nodes"], 2)
                self.assertEqual(metadata["removed_constant_node_labels"], [1])
                np.testing.assert_array_equal(graph["node_labels"], [0, 2])
                np.testing.assert_array_equal(graph["nodes_idx"], [0, 1])
                graphs.append(graph)
                model = GNN(1, 4, 1, 3, 2, 2, rngs=nnx.Rngs(42))
                one, _ = evaluate_split(model, data, graph, "valid", 1)
                two, _ = evaluate_split(model, data, graph, "valid", 2)
                self.assertAlmostEqual(one["model"]["mse"], two["model"]["mse"], places=6)
                self.assertEqual(one["model"]["count"], two["model"]["count"])
                traffic.loc[traffic["date"] >= times[28], "congestion"] = 0.
            for key in graphs[0]:
                np.testing.assert_array_equal(graphs[0][key], graphs[1][key])

    def test_upstream_variant_is_trainable_for_four_models_and_four_steps(self):
        x = np.random.default_rng(9).uniform(size=(2, 3, 2, 1)).astype(np.float32)
        y = np.random.default_rng(10).uniform(size=(2, 4, 2, 1)).astype(np.float32)
        graph, _ = prepare_data(x, y, self.graph_data())
        individual, _ = prepare_data(x[1:], y[1:], self.graph_data())
        for spatial in SpatialType:
            for temporal in TemporalType:
                with self.subTest(spatial=spatial, temporal=temporal):
                    model = GNN(1, 4, 1, 3, 4, 1, spatial=spatial, temporal=temporal,
                                architecture="upstream_bidirectional", rngs=nnx.Rngs(42))
                    self.assertEqual(model(graph).shape, (4, 4, 1))
                    np.testing.assert_allclose(model(graph)[2:], model(individual), atol=1e-6)
                    optimizer = nnx.Optimizer(model, optax.adam(1e-4))
                    loss, count = train_step(model, optimizer, micro_batches(x, y, np.ones_like(y, bool), self.graph_data(), 1), alpha=.4)
                    self.assertTrue(np.isfinite(loss))
                    self.assertGreater(count, 0)

    def test_time_alpha_changes_loss_and_preserves_accumulation_with_missing_pairs(self):
        x = np.random.default_rng(19).uniform(size=(3, 3, 2, 1)).astype(np.float32)
        y = np.random.default_rng(20).uniform(size=(3, 4, 2, 1)).astype(np.float32)
        mask = np.ones_like(y, bool)
        mask[0, 2:] = False
        mask[2, 1::2] = False
        outcomes = []
        for size in (3, 1):
            model = GNN(1, 4, 1, 3, 4, 2, rngs=nnx.Rngs(42))
            optimizer = nnx.Optimizer(model, optax.adam(1e-4))
            outcomes.append((model, train_step(model, optimizer, micro_batches(x, y, mask, self.graph_data(), size), alpha=.2)))
        self.assertAlmostEqual(outcomes[0][1][0], outcomes[1][1][0], places=5)
        for a, b in zip(jax.tree_util.tree_leaves(nnx.state(outcomes[0][0], nnx.Param)),
                        jax.tree_util.tree_leaves(nnx.state(outcomes[1][0], nnx.Param))):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
        losses = []
        for alpha in (0., 1.):
            model = GNN(1, 4, 1, 3, 4, 2, rngs=nnx.Rngs(42))
            optimizer = nnx.Optimizer(model, optax.adam(1e-4))
            losses.append(train_step(model, optimizer, micro_batches(x, y, mask, self.graph_data(), 3), alpha=alpha)[0])
        self.assertNotAlmostEqual(*losses, places=3)

    def test_time_alpha_requires_multiple_horizons_and_valid_range(self):
        x = np.ones((1, 3, 2, 1), np.float32)
        y = np.ones((1, 1, 2, 1), np.float32)
        model = GNN(1, 4, 1, 3, 1, 2, rngs=nnx.Rngs(42))
        optimizer = nnx.Optimizer(model, optax.adam(1e-4))
        with self.assertRaisesRegex(ValueError, "at least two"):
            train_step(model, optimizer, micro_batches(x, y, np.ones_like(y, bool), self.graph_data(), 1), alpha=.2)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            train_step(model, optimizer, [], alpha=1.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
