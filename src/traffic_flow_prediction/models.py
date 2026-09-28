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
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.att_src = nnx.Linear(out_features, 1, use_bias=False, rngs=rngs)
        self.att_dst = nnx.Linear(out_features, 1, use_bias=False, rngs=rngs)

        self.negative_slope = negative_slope

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

        h = self.linear(graph.nodes)                  # (N,F)

        senders = graph.senders
        receivers = graph.receivers

        h_src = h[senders]                            # (E,F)
        h_dst = h[receivers]                          # (E,F)

        # e_ij = a_l^T Wh_i + a_r^T Wh_j
        e = self.att_src(h_src) + self.att_dst(h_dst) # (E,1)
        e = nnx.leaky_relu(e, self.negative_slope)
        e = e.squeeze(-1)

        alpha = jraph.segment_softmax(e, receivers, num_segments=h.shape[0])
        messages = alpha[:, None] * h_src
        out = jraph.segment_sum(messages, receivers, num_segments=h.shape[0])

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
        self.linear = nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = graph.nodes
        sum_n_node = nodes.shape[0]

        nodes = self.linear(nodes)

        senders   = jnp.concatenate([graph.senders,   jnp.arange(sum_n_node)])
        receivers = jnp.concatenate([graph.receivers, jnp.arange(sum_n_node)])
        
        def count_edges(x: ArrayLike) -> jax.Array:
            return jraph.segment_sum(jnp.ones_like(x), x, num_segments=sum_n_node)
        
        sender_degree   = count_edges(senders)
        receiver_degree = count_edges(receivers)
        
        nodes = nodes * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, jnp.newaxis]
        nodes = jraph.segment_sum(nodes[senders], receivers, num_segments=sum_n_node)
        nodes = nodes * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, jnp.newaxis]

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
        rngs:     nnx.Rngs,
    ):
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
            out_features = out_feat if layer == num_layers - 1 else hidden_feat
            # self.layers.append(GATLayer(in_features, out_features, rngs=rngs))
            # self.layers.append(GCNLayer(in_features, out_features, rngs=rngs))
            self.layers.append(layer_cls(in_features, out_features, rngs=rngs))
            self.layers.append(ReLULayer())
        # self.lstm = nnx.LSTMCell(out_feat, out_feat, rngs=rngs)
        self.lstm = cell_cls(out_feat, out_feat, rngs=rngs)
        self.proj = nnx.Linear(in_time, out_time, rngs=rngs)
        self.carry = self.lstm.initialize_carry((out_feat,))

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

        def node_forward(sequence):
            def step(carry, x_t):
                carry, h_t = self.lstm(carry, x_t)
                return carry, h_t
            
            scan_fn = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            _, h = scan_fn(self.carry, sequence)
            
            y = self.proj(h.T).T
            return y

        # (N,T)
        y = nnx.vmap(node_forward)(x)
        return y