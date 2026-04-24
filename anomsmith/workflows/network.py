"""Network-shaped anomaly workflows.

Designed to interoperate with organizational communication graphs such as
``org_network_analysis`` (`NetworkAnalyzer._to_edge_list`): undirected edges
aggregated by sender/receiver pair with integer weights (event counts), and a
fixed member roster so isolated nodes still appear in outputs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from anomsmith.constants import (
    DEFAULT_ISOLATION_FOREST_N_ESTIMATORS,
    DEFAULT_OUTLIER_CONTAMINATION,
)
from anomsmith.objects.views import ScoreView
from anomsmith.primitives.detectors.ml import IsolationForestDetector
from anomsmith.primitives.thresholding import ThresholdRule, apply_threshold

logger = logging.getLogger(__name__)


def _numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        c
        for c in features.columns
        if pd.api.types.is_numeric_dtype(features[c])
    ]


def _detect_with_isolation_forest(
    features: pd.DataFrame,
    threshold_rule: ThresholdRule,
    *,
    feature_cols: list[str] | None,
    contamination: float,
    n_estimators: int,
    random_state: int | None,
    min_rows: int,
    too_few_message: str,
    log_action: str,
) -> pd.DataFrame:
    """Fit isolation forest on numeric rows and append score + flag columns."""
    if len(features) < min_rows:
        raise ValueError(too_few_message)

    if feature_cols is None:
        cols = _numeric_feature_columns(features)
    else:
        cols = list(feature_cols)
    if not cols:
        raise ValueError("No numeric feature columns to score.")

    X = features[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("features contains non-finite values in feature_cols.")

    det = IsolationForestDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    det.fit(X)
    if det.model_ is None or not hasattr(det, "scaler_"):
        raise RuntimeError("IsolationForestDetector did not finish fitting.")

    Xs = det.scaler_.transform(X)
    raw = det.model_.decision_function(Xs)
    scores = -np.asarray(raw, dtype=np.float64)
    idx = features.index
    score_view = ScoreView(index=idx, scores=scores)
    label_view = apply_threshold(score_view, threshold_rule)

    out = features.copy()
    out["score"] = score_view.scores
    out["flag"] = label_view.labels
    logger.info(
        "%s: %d rows, %d flagged", log_action, len(out), int(out["flag"].sum())
    )
    return out


def aggregate_undirected_edges(
    communications: pd.DataFrame,
    *,
    sender_col: str = "sender_id",
    receiver_col: str = "receiver_id",
    drop_self_loops: bool = True,
) -> pd.DataFrame:
    """Aggregate communication rows into undirected weighted edges.

    Mirrors the aggregation in ``org_network_analysis`` business logic:
    each unique unordered pair ``(min(a,b), max(a,b))`` gets weight equal to
    the number of rows (communication events) between those endpoints.

    Args:
        communications: One row per event; must include sender/receiver columns.
        sender_col: Column name for the sender endpoint (default ``sender_id``).
        receiver_col: Column name for the receiver endpoint (default ``receiver_id``).
        drop_self_loops: If True, rows where sender equals receiver are skipped.

    Returns:
        DataFrame with columns ``u``, ``v``, ``weight`` (integer counts), sorted by ``u``, ``v``.
    """
    required = {sender_col, receiver_col}
    missing = required - set(communications.columns)
    if missing:
        raise ValueError(f"communications is missing columns: {sorted(missing)}")

    df = communications[[sender_col, receiver_col]].copy()
    df = df.dropna(subset=[sender_col, receiver_col])
    if drop_self_loops:
        df = df[df[sender_col] != df[receiver_col]]

    if df.empty:
        return pd.DataFrame(columns=["u", "v", "weight"])

    u = np.minimum(df[sender_col].values, df[receiver_col].values)
    v = np.maximum(df[sender_col].values, df[receiver_col].values)
    keys = pd.MultiIndex.from_arrays([u, v])
    counts = keys.value_counts().sort_index()
    out = counts.reset_index()
    out.columns = ["u", "v", "weight"]
    out["weight"] = out["weight"].astype(int)
    return out


def _build_neighbor_lists(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
    """Undirected adjacency as neighbor lists and weighted degree (strength)."""
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    strength: np.ndarray = np.zeros(n, dtype=np.float64)
    for a, b, wt in zip(src, dst, w):
        ia, ib = int(a), int(b)
        neighbors[ia].append((ib, float(wt)))
        neighbors[ib].append((ia, float(wt)))
        strength[ia] += float(wt)
        strength[ib] += float(wt)
    return neighbors, strength


def _pagerank_undirected(
    n: int,
    neighbors: list[list[tuple[int, float]]],
    strength: np.ndarray,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Power iteration PageRank on an undirected weighted graph."""
    r: np.ndarray = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    teleport = (1.0 - alpha) / max(n, 1)
    for _ in range(max_iter):
        r_new: np.ndarray = np.full(n, teleport, dtype=np.float64)
        for i in range(n):
            ri = r[i]
            if ri == 0.0:
                continue
            si = strength[i]
            if si <= 0.0:
                r_new += alpha * ri / max(n, 1)
            else:
                for j, wij in neighbors[i]:
                    r_new[j] += alpha * wij / si * ri
        if float(np.linalg.norm(r_new - r, ord=1)) < tol:
            break
        r = r_new
    s = float(r.sum())
    if s > 0:
        r /= s
    return r


def node_features_from_edges(
    edges: pd.DataFrame,
    nodes: pd.Index | list[Any] | np.ndarray,
    *,
    u_col: str = "u",
    v_col: str = "v",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Per-node structural features for anomaly scoring.

    Uses the same edge table shape produced by :func:`aggregate_undirected_edges`
    (``u``, ``v``, ``weight``). Every id in ``nodes`` appears in the index; nodes
    with no incident edges get zero strength, zero distinct-neighbor count, and
    uniform PageRank mass.

    Feature columns:

    - ``weighted_degree``: sum of incident edge weights (communication volume).
    - ``neighbor_count``: number of distinct neighbors.
    - ``pagerank``: undirected PageRank (numpy power iteration; no NetworkX).

    Args:
        edges: Edge list with endpoints and non-negative weights.
        nodes: Complete roster of node identifiers (e.g. all team member ids).
        u_col, v_col, weight_col: Column names in ``edges``.

    Returns:
        DataFrame indexed by node id with numeric feature columns.
    """
    if isinstance(nodes, np.ndarray):
        node_index = pd.Index(nodes)
    elif isinstance(nodes, list):
        node_index = pd.Index(nodes)
    else:
        node_index = nodes

    if node_index.duplicated().any():
        raise ValueError("nodes must be unique")

    id_to_idx = {nid: i for i, nid in enumerate(node_index)}
    n = len(node_index)

    if not edges.empty:
        for col in (u_col, v_col, weight_col):
            if col not in edges.columns:
                raise ValueError(f"edges is missing column {col!r}")
        src = edges[u_col].map(id_to_idx).astype("Int64")
        dst = edges[v_col].map(id_to_idx).astype("Int64")
        mask = src.notna() & dst.notna()
        src_arr = src[mask].astype(np.int64).to_numpy()
        dst_arr = dst[mask].astype(np.int64).to_numpy()
        w_arr = edges.loc[mask, weight_col].astype(float).to_numpy()
    else:
        src_arr = np.array([], dtype=np.int64)
        dst_arr = np.array([], dtype=np.int64)
        w_arr = np.array([], dtype=np.float64)

    neighbors, strength_from_edges = _build_neighbor_lists(n, src_arr, dst_arr, w_arr)
    neighbor_count = np.array(
        [len(neighbors[i]) for i in range(n)], dtype=np.float64
    )
    pr = _pagerank_undirected(n, neighbors, strength_from_edges)

    out = pd.DataFrame(
        {
            "weighted_degree": strength_from_edges,
            "neighbor_count": neighbor_count,
            "pagerank": pr,
        },
        index=node_index.copy(),
    )
    return out


def edge_features_from_edges(
    edges: pd.DataFrame,
    nodes: pd.Index | list[Any] | np.ndarray,
    *,
    u_col: str = "u",
    v_col: str = "v",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Per-edge (dyad) features derived from aggregated undirected weights.

    Rows follow the ``u``, ``v``, ``weight`` table from
    :func:`aggregate_undirected_edges`. Combines each edge weight with endpoint
    strengths from :func:`node_features_from_edges` to highlight unusually heavy
    links relative to endpoint activity.

    Columns:

    - ``weight``: aggregated event count on the dyad.
    - ``share_of_endpoint_volume``: ``2 * weight / (deg(u) + deg(v))`` using
      endpoint ``weighted_degree`` values (each edge's weight is included in
      both degrees).
    - ``log1p_weight``: ``log1p(weight)`` for scale-robust modeling.

    Args:
        edges: Non-empty edge list (typically aggregated counts).
        nodes: Full node roster (same semantics as :func:`node_features_from_edges`).
        u_col, v_col, weight_col: Column names in ``edges``.

    Returns:
        DataFrame indexed by ``MultiIndex`` ``(u, v)`` with numeric feature columns.

    Raises:
        ValueError: If ``edges`` is empty.
    """
    if edges.empty:
        raise ValueError("edge_features_from_edges requires a non-empty edges frame")

    nf = node_features_from_edges(
        edges, nodes, u_col=u_col, v_col=v_col, weight_col=weight_col
    )
    u = edges[u_col].to_numpy()
    v = edges[v_col].to_numpy()
    w = edges[weight_col].astype(float).to_numpy()
    du = nf["weighted_degree"].reindex(u).to_numpy(dtype=np.float64)
    dv = nf["weighted_degree"].reindex(v).to_numpy(dtype=np.float64)
    denom = du + dv
    share = np.where(denom > 0.0, (2.0 * w) / denom, 0.0)
    logw = np.log1p(w)
    idx = pd.MultiIndex.from_arrays([u, v], names=[u_col, v_col])
    return pd.DataFrame(
        {
            "weight": w,
            "share_of_endpoint_volume": share,
            "log1p_weight": logw,
        },
        index=idx,
    )


def node_touch_counts_by_bin(
    communications: pd.DataFrame,
    nodes: pd.Index | list[Any] | np.ndarray,
    *,
    timestamp_col: str = "timestamp",
    sender_col: str = "sender_id",
    receiver_col: str = "receiver_id",
    freq: str = "1D",
    drop_self_loops: bool = True,
) -> pd.DataFrame:
    """Count how often each node sends or receives in each time bin.

    Each communication row increments both the sender and the receiver for the
    floored period bucket (pandas offset string, e.g. ``\"1D\"``, ``\"6H\"``).

    Args:
        communications: Must include timestamp and endpoint columns.
        nodes: Full roster; bins include only these ids (other endpoints dropped).
        timestamp_col: Parseable timestamps (``pd.to_datetime``).
        sender_col, receiver_col: Endpoint identifiers.
        freq: Bin size passed to ``Series.dt.floor``.
        drop_self_loops: If True, rows with sender equal receiver are skipped.

    Returns:
        DataFrame with index = node id, columns = bin start (``datetime64``),
        values = integer touch counts. Missing bins are zero; nodes with no
        events still appear as rows of zeros when listed in ``nodes``.
    """
    if isinstance(nodes, np.ndarray):
        node_index = pd.Index(nodes)
    elif isinstance(nodes, list):
        node_index = pd.Index(nodes)
    else:
        node_index = nodes
    if node_index.duplicated().any():
        raise ValueError("nodes must be unique")

    required = {timestamp_col, sender_col, receiver_col}
    missing = required - set(communications.columns)
    if missing:
        raise ValueError(f"communications is missing columns: {sorted(missing)}")

    df = communications[list(required)].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=False, errors="coerce")
    df = df.dropna(subset=[timestamp_col, sender_col, receiver_col])
    if drop_self_loops:
        df = df[df[sender_col] != df[receiver_col]]

    node_set = set(node_index.tolist())
    if df.empty:
        return pd.DataFrame(0.0, index=node_index.copy(), columns=[])

    df["_bin"] = df[timestamp_col].dt.floor(freq)
    part_s = df[["_bin", sender_col]].rename(columns={sender_col: "_node"})
    part_r = df[["_bin", receiver_col]].rename(columns={receiver_col: "_node"})
    long = pd.concat([part_s, part_r], ignore_index=True)
    long = long[long["_node"].isin(node_set)]
    if long.empty:
        return pd.DataFrame(0.0, index=node_index.copy(), columns=[])

    counts = long.groupby(["_bin", "_node"], observed=True).size()
    wide = counts.unstack(level="_node", fill_value=0)
    wide = wide.reindex(columns=node_index, fill_value=0)
    mat = wide.T
    mat = mat.sort_index(axis=0).sort_index(axis=1)
    mat.index.name = None
    mat.columns.name = None
    return mat.astype(np.float64)


def detect_network_node_anomalies(
    node_features: pd.DataFrame,
    threshold_rule: ThresholdRule,
    *,
    feature_cols: list[str] | None = None,
    contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    n_estimators: int = DEFAULT_ISOLATION_FOREST_N_ESTIMATORS,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Flag structurally unusual nodes using isolation forest on feature rows.

    Fits :class:`~anomsmith.primitives.detectors.ml.IsolationForestDetector` on
    the numeric feature matrix and thresholds anomaly scores. Typical use: pass
    the output of :func:`node_features_from_edges` (options: join extra numeric
    columns before calling).

    Args:
        node_features: Rows are nodes (index = node id); values are features.
        threshold_rule: Rule applied to isolation scores (higher = more anomalous).
        feature_cols: Columns to use; default is all numeric columns in the frame.
        contamination: Passed to ``IsolationForest``.
        n_estimators: Number of trees in the forest.
        random_state: Optional RNG seed.

    Returns:
        DataFrame with original feature columns plus ``score`` and ``flag``
        (1 = anomaly). Index matches ``node_features``.

    Raises:
        ValueError: If fewer than two rows are present (isolation forest requires
            a batch to score relative to).
    """
    return _detect_with_isolation_forest(
        node_features,
        threshold_rule,
        feature_cols=feature_cols,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        min_rows=2,
        too_few_message=(
            "detect_network_node_anomalies requires at least 2 nodes; "
            f"got {len(node_features)}"
        ),
        log_action="detect_network_node_anomalies",
    )


def detect_network_edge_anomalies(
    edge_features: pd.DataFrame,
    threshold_rule: ThresholdRule,
    *,
    feature_cols: list[str] | None = None,
    contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    n_estimators: int = DEFAULT_ISOLATION_FOREST_N_ESTIMATORS,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Flag structurally unusual dyads using isolation forest on edge features.

    Expects a frame such as the output of :func:`edge_features_from_edges`
    (numeric columns only are used by default).

    Raises:
        ValueError: If fewer than two edges are present.
    """
    return _detect_with_isolation_forest(
        edge_features,
        threshold_rule,
        feature_cols=feature_cols,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        min_rows=2,
        too_few_message=(
            "detect_network_edge_anomalies requires at least 2 edges; "
            f"got {len(edge_features)}"
        ),
        log_action="detect_network_edge_anomalies",
    )


def detect_network_temporal_node_anomalies(
    touch_counts_by_bin: pd.DataFrame,
    threshold_rule: ThresholdRule,
    *,
    feature_cols: list[str] | None = None,
    contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    n_estimators: int = DEFAULT_ISOLATION_FOREST_N_ESTIMATORS,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Flag nodes whose time-bin activity vectors look unlike the rest.

    Rows are nodes (index from :func:`node_touch_counts_by_bin`). Columns should
    be numeric bin counts (any column names); by default all numeric columns are
    used as features.

    Raises:
        ValueError: If fewer than two nodes, no numeric columns, or any bin
            column contains non-finite values.
    """
    cols = (
        feature_cols
        if feature_cols is not None
        else _numeric_feature_columns(touch_counts_by_bin)
    )
    if not cols:
        raise ValueError(
            "detect_network_temporal_node_anomalies needs at least one numeric "
            "time-bin column in touch_counts_by_bin (or pass feature_cols)."
        )
    return _detect_with_isolation_forest(
        touch_counts_by_bin,
        threshold_rule,
        feature_cols=feature_cols,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        min_rows=2,
        too_few_message=(
            "detect_network_temporal_node_anomalies requires at least 2 nodes; "
            f"got {len(touch_counts_by_bin)}"
        ),
        log_action="detect_network_temporal_node_anomalies",
    )
