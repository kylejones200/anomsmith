"""Tests for optional NetworkX-backed graph metrics."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("networkx")

from anomsmith.workflows.network import (
    aggregate_undirected_edges,
    node_graph_metrics_networkx,
)


def test_star_hub_has_highest_betweenness() -> None:
    rows = []
    hub = 30
    for sid in [10, 20, 40, 50]:
        rows.append((sid, hub))
        rows.append((hub, sid))
    df = pd.DataFrame(rows, columns=["sender_id", "receiver_id"])
    edges = aggregate_undirected_edges(df)
    members = [10, 20, 30, 40, 50]
    nxm = node_graph_metrics_networkx(edges, members)
    assert nxm.loc[hub, "betweenness_centrality"] == pytest.approx(
        nxm["betweenness_centrality"].max(), rel=1e-5, abs=1e-5
    )


def test_isolated_nodes_zero_centrality() -> None:
    edges = pd.DataFrame({"u": [1], "v": [2], "weight": [1]})
    members = [1, 2, 99]
    nxm = node_graph_metrics_networkx(edges, members)
    assert nxm.loc[99, "betweenness_centrality"] == 0.0
    assert nxm.loc[99, "closeness_centrality"] == 0.0
