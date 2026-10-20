from enum import Enum

import jax
import jax.numpy as jnp
import jraph
from flax import nnx
from jax.typing import ArrayLike


class SpatialType(Enum):
    gat = "GAT"
    gcn = "GCN"


class TemporalType(Enum):
    lstm = "LSTM"
    gru  = "GRU"


class GATLayer(nnx.Module):

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        *,
        negative_slope: float = 0.2,
        rngs:           nnx.Rngs,
    ) -> None:
        self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)
        self.att_src = nnx.Linear(out_features, 1, use_bias=False, rngs=rngs)
        self.att_dst = nnx.Linear(out_features, 1, use_bias=False, rngs=rngs)
        self.bias = nnx.Param(jnp.zeros(out_features))

        self.negative_slope = negative_slope

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

        h = self.linear(graph.nodes)                  # (N,F)

        senders = jnp.concatenate([graph.senders, jnp.arange(h.shape[0])])
        receivers = jnp.concatenate([graph.receivers, jnp.arange(h.shape[0])])
        keep = jnp.concatenate([graph.senders != graph.receivers, jnp.ones(h.shape[0], dtype=bool)])

        h_src = h[senders]                            # (E,F)
        h_dst = h[receivers]                          # (E,F)

        # e_ij = a_l^T Wh_i + a_r^T Wh_j
        e = self.att_src(h_src) + self.att_dst(h_dst) # (E,1)
        e = nnx.leaky_relu(e, self.negative_slope)
        e = jnp.where(keep, e.squeeze(-1), -jnp.inf)

        alpha = jraph.segment_softmax(e, receivers, num_segments=h.shape[0])
        messages = alpha[:, None] * h_src
        out = jraph.segment_sum(messages, receivers, num_segments=h.shape[0]) + self.bias.value

        return graph._replace(nodes=out)


class GCNLayer(nnx.Module):

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        *,
        rngs:     nnx.Rngs,
        use_bias: bool = True,
    ) -> None:
        self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)
        self.bias = nnx.Param(jnp.zeros(out_features)) if use_bias else None

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        n = graph.nodes.shape[0]
        weights = jnp.ones_like(graph.senders, dtype=graph.nodes.dtype) if graph.edges is None else graph.edges.reshape(-1)
        is_loop = graph.senders == graph.receivers
        loop_count = jraph.segment_sum(is_loop.astype(jnp.int32), graph.receivers, num_segments=n)
        loop_weight = jraph.segment_sum(jnp.where(is_loop, weights, 0.), graph.receivers, num_segments=n)
        weights = jnp.concatenate([jnp.where(is_loop, 0., weights), jnp.where(loop_count > 0, loop_weight, 1.)])
        senders = jnp.concatenate([graph.senders, jnp.arange(n)])
        receivers = jnp.concatenate([graph.receivers, jnp.arange(n)])
        degree = jraph.segment_sum(weights, receivers, num_segments=n)
        inv_sqrt = jnp.where(degree > 0, jax.lax.rsqrt(jnp.maximum(degree, 1e-12)), 0.)
        coefficients = weights * inv_sqrt[senders] * inv_sqrt[receivers]
        messages = graph.nodes[senders] * coefficients[:, None]
        nodes = self.linear(jraph.segment_sum(messages, receivers, num_segments=n))
        if self.bias is not None:
            nodes = nodes + self.bias.value
        return graph._replace(nodes=nodes)


class ReLULayer(nnx.Module):
    
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return graph._replace(nodes=nnx.relu(graph.nodes))


class SpatialModule(nnx.Module):
    
    def __init__(
        self,
        in_features:     int,
        hidden_features: int,
        layers:          int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()


class TemporalModule(nnx.Module):
    
    def __init__(
        self,
        in_features:     int,
        hidden_features: int,
        out_time:        int,
        *,
        cell: type[nnx.RNNCellBase],
        rngs: nnx.Rngs,
    ) -> None:
        nnx.LSTMCell()
        self.encode = nnx.RNN(cell(in_features, hidden_features))
    
    def __call__(self, x: ArrayLike) -> jax.Array:
        # (T, Hin)
        pass


class GNN(nnx.Module):

    def __init__(
        self,
        in_feat:     int,
        hidden_feat: int,
        out_feat:    int,
        in_time:     int,
        out_time:    int,
        num_layers:  int,
        *,
        spatial:  SpatialType  = SpatialType.gcn,
        temporal: TemporalType = TemporalType.lstm,
        temporal_hidden: int | None = None,
        architecture: str = "pfm",
        rngs:     nnx.Rngs,
    ):
        temporal_hidden = hidden_feat if temporal_hidden is None else temporal_hidden
        if min(in_feat, hidden_feat, out_feat, in_time, out_time, num_layers, temporal_hidden) < 1:
            raise ValueError("Model dimensions must be positive.")
        if architecture not in ("pfm", "upstream_bidirectional"):
            raise ValueError("Unknown model architecture.")
        self.architecture = architecture
        self.out_time, self.out_feat = out_time, out_feat
        match spatial:
            case SpatialType.gat:
                layer_cls = GATLayer
            case SpatialType.gcn:
                layer_cls = GCNLayer
        match temporal:
            case TemporalType.lstm:
                cell_cls = nnx.LSTMCell
            case TemporalType.gru:
                cell_cls = nnx.GRUCell
        self.layers = []
        for layer in range(num_layers):
            in_features  = in_feat  if layer == 0              else hidden_feat
            out_features = hidden_feat
            # self.layers.append(GATLayer(in_features, out_features, rngs=rngs))
            # self.layers.append(GCNLayer(in_features, out_features, rngs=rngs))
            self.layers.append(layer_cls(in_features, out_features, rngs=rngs))
            self.layers.append(ReLULayer())
        # self.lstm = nnx.LSTMCell(out_feat, out_feat, rngs=rngs)
        if architecture == "pfm":
            self.lstm = cell_cls(hidden_feat, temporal_hidden, rngs=rngs)
            self.proj = nnx.Linear(temporal_hidden, out_time*out_feat, rngs=rngs)
            self.carry = self.lstm.initialize_carry((hidden_feat,))
        else:
            self.encoder = nnx.Bidirectional(
                nnx.RNN(cell_cls(hidden_feat, temporal_hidden, rngs=rngs)),
                nnx.RNN(cell_cls(hidden_feat, temporal_hidden, rngs=rngs)),
                merge_fn=jnp.add,
            )
            self.decoder = nnx.Linear(in_time, out_time, rngs=rngs)
            self.output = nnx.Linear(temporal_hidden, out_feat, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jax.Array:
        # (N,T,1)
        x = graph.nodes

        def spatial_step(nodes):
            g = graph._replace(nodes=nodes)
            for layer in self.layers:
                g = layer(g)
            return g.nodes
            # return self.gat(g).nodes

        # (T,N,1)
        x = jnp.swapaxes(x, 0, 1)
        # (T,N,H)
        x = nnx.vmap(spatial_step)(x)
        # (N,T,H)
        x = jnp.swapaxes(x, 0, 1)
        if self.architecture == "upstream_bidirectional":
            hidden = self.encoder(x)
            projected = self.decoder(jnp.swapaxes(hidden, 1, 2))
            return self.output(jnp.swapaxes(projected, 1, 2))

        def node_forward(sequence):
            def step(carry, x_t):
                carry, h_t = self.lstm(carry, x_t)
                return carry, h_t
            
            scan_fn = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            _, h = scan_fn(self.carry, sequence)
            
            y = self.proj(h[-1]).reshape(self.out_time, self.out_feat)
            return y

        # (N,T)
        y = nnx.vmap(node_forward)(x)
        return y