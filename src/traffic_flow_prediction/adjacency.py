from enum import Enum

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class ADJType(Enum):
    correlation = "adj_correlation"
    closeness   = "adj_closeness"


def reindex_cluster_time_matrix(c_t_matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], list[int]]:
    matrix = c_t_matrix.copy()
    nodes_idx = sorted(matrix.columns.tolist())
    mapping = {cluster: i for i, cluster in enumerate(nodes_idx)}
    matrix = matrix[nodes_idx].copy()
    matrix.columns = [mapping[c] for c in matrix.columns]
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix, mapping, nodes_idx


def generate_correlation(
    c_t_matrix: pd.DataFrame,
    k:          int,
    *,
    negative_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    if k < 1 or negative_policy not in ("error", "zero"):
        raise ValueError("k must be positive and negative_policy must be error or zero.")
    matrix = c_t_matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.loc[:, matrix.nunique(dropna=True) > 1]
    if matrix.empty:
        raise ValueError("No nonconstant training nodes for correlation.")
    matrix_gnn, _, nodes_idx = reindex_cluster_time_matrix(matrix)
    corr_values = matrix_gnn.corr(method="pearson", min_periods=2).to_numpy()
    np.fill_diagonal(corr_values, np.nan)
    edges = []
    for i in range(len(nodes_idx)):
        candidates = np.flatnonzero(np.isfinite(corr_values[i]))
        if len(nodes_idx) > 1 and not len(candidates):
            raise ValueError(f"No valid training correlations for node {nodes_idx[i]}.")
        top_idx = candidates[np.argsort(-corr_values[i, candidates], kind="stable")[:k]]
        for j in top_idx:
            correlation = float(corr_values[i, j])
            if correlation < 0 and negative_policy == "error":
                raise ValueError("Negative top-k correlations require an explicit policy; use negative_policy='zero' to preserve the original zero-clipping behavior.")
            weight = max(correlation, 0.) if negative_policy == "zero" else correlation
            edges.append((i, int(j), weight, correlation))
    df_edges = pd.DataFrame(edges, columns=["source", "target", "weight", "raw_correlation"])
    return matrix_gnn, df_edges, nodes_idx


def knn_from_centroids(centroids: pd.DataFrame, k: int) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be positive.")
    centroids = centroids.sort_values("cluster").reset_index(drop=True).copy()
    coords = centroids[["utm_x", "utm_y"]]
    n = len(coords)
    if n == 0:
        raise ValueError("There is no centroids to build the adjacency")
    if n == 1:
        return pd.DataFrame({"source": [0], "target": [0], "distance": [0.0], "weight": [1.0]})
    
    k_eff = min(k, n - 1)
    model = NearestNeighbors(n_neighbors=k_eff+1, metric="euclidean")
    model.fit(coords)
    distances, indices = model.kneighbors(coords)
    
    edges = []
    for i in range(n):
        source = int(centroids.loc[i, "cluster"])
        neighbors = [(dist, j) for dist, j in zip(distances[i], indices[i]) if j != i][:k_eff]
        for dist, j in neighbors:
            target = int(centroids.loc[j, "cluster"])
            edges.append((source, target, float(dist)))
    
    df_edges = pd.DataFrame(edges, columns=["source", "target", "distance"])
    sigma = df_edges["distance"].mean()
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    df_edges["weight"] = np.exp(-df_edges["distance"] / sigma)
    return df_edges


def generate_closeness(
    c_t_matrix: pd.DataFrame,
    k:          int,
    centroids:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    matrix = c_t_matrix.copy()
    matrix_gnn, mapping, nodes_idx = reindex_cluster_time_matrix(matrix)
    
    centroids = centroids[centroids["cluster"].isin(mapping.keys())].copy()
    centroids["cluster"] = centroids["cluster"].map(mapping).astype(int)
    centroids = centroids.sort_values("cluster")
    
    df_edges = knn_from_centroids(centroids, k)
    
    return matrix_gnn, df_edges, nodes_idx
