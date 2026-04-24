"""Tests for network anomaly workflows (org-style communication graphs)."""

import pandas as pd
import pytest

from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows.network import (
    aggregate_undirected_edges,
    detect_network_edge_anomalies,
    detect_network_node_anomalies,
    detect_network_temporal_node_anomalies,
    edge_features_from_edges,
    node_features_from_edges,
    node_touch_counts_by_bin,
)


class TestAggregateUndirectedEdges:
    def test_counts_match_pair_occurrences(self) -> None:
        comms = pd.DataFrame(
            {
                "sender_id": [1, 2, 1, 1],
                "receiver_id": [2, 1, 2, 2],
            }
        )
        edges = aggregate_undirected_edges(comms)
        assert len(edges) == 1
        assert int(edges["u"].iloc[0]) == 1
        assert int(edges["v"].iloc[0]) == 2
        assert int(edges["weight"].iloc[0]) == 4

    def test_drops_self_loops_by_default(self) -> None:
        comms = pd.DataFrame({"sender_id": [1, 1], "receiver_id": [2, 1]})
        edges = aggregate_undirected_edges(comms)
        assert len(edges) == 1
        assert edges["weight"].iloc[0] == 1


class TestNodeFeaturesFromEdges:
    def test_star_hub_has_highest_weighted_degree(self) -> None:
        # Nodes 0..4; hub 2 like org_network_analysis bottleneck tests
        members = [10, 20, 30, 40, 50]
        comms = []
        for sid in [10, 20, 40, 50]:
            comms.append((sid, 30))
            comms.append((30, sid))
        df = pd.DataFrame(comms, columns=["sender_id", "receiver_id"])
        edges = aggregate_undirected_edges(df)
        feat = node_features_from_edges(edges, members)
        assert feat.loc[30, "weighted_degree"] == feat["weighted_degree"].max()
        assert feat.loc[10, "neighbor_count"] == 1.0

    def test_isolated_member_zero_degree(self) -> None:
        members = [1, 2, 3]
        edges = pd.DataFrame({"u": [1], "v": [2], "weight": [1]})
        feat = node_features_from_edges(edges, members)
        assert feat.loc[3, "weighted_degree"] == 0.0
        assert feat.loc[3, "neighbor_count"] == 0.0


class TestEdgeFeaturesAndEdgeAnomalies:
    def test_edge_features_requires_non_empty_edges(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            edge_features_from_edges(
                pd.DataFrame(columns=["u", "v", "weight"]), nodes=[1, 2]
            )

    def test_share_of_endpoint_volume_bounded(self) -> None:
        edges = aggregate_undirected_edges(
            pd.DataFrame({"sender_id": [1, 1], "receiver_id": [2, 2]})
        )
        ef = edge_features_from_edges(edges, [1, 2])
        assert (ef["share_of_endpoint_volume"] <= 1.0 + 1e-9).all()
        assert (ef["share_of_endpoint_volume"] >= 0.0).all()

    def test_detect_edge_anomalies_min_two_edges(self) -> None:
        ef = pd.DataFrame({"weight": [1.0]}, index=pd.MultiIndex.from_tuples([(1, 2)]))
        rule = ThresholdRule(method="quantile", value=0.0, quantile=0.5)
        with pytest.raises(ValueError, match="at least 2 edges"):
            detect_network_edge_anomalies(ef, rule)


class TestTemporalTouchCounts:
    def test_touch_counts_shape_and_spike(self) -> None:
        comms = pd.DataFrame(
            {
                "sender_id": [1, 1, 1, 2],
                "receiver_id": [2, 2, 2, 3],
                "timestamp": pd.to_datetime(
                    ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-02"]
                ),
            }
        )
        nodes = [1, 2, 3]
        touch = node_touch_counts_by_bin(comms, nodes, freq="1D")
        assert list(touch.index) == nodes
        assert touch.shape[1] == 2
        assert touch.loc[1].sum() >= 3

    def test_temporal_detect_requires_bin_columns(self) -> None:
        wide = pd.DataFrame(0.0, index=[1, 2], columns=[])
        rule = ThresholdRule(method="quantile", value=0.0, quantile=0.5)
        with pytest.raises(ValueError, match="numeric"):
            detect_network_temporal_node_anomalies(wide, rule)


class TestDetectNetworkNodeAnomalies:
    def test_flags_extreme_hub(self) -> None:
        members = list(range(20))
        rows = []
        hub = 0
        for i in range(1, 20):
            for _ in range(3):
                rows.append((hub, i))
                rows.append((i, hub))
        # background clique among non-hub
        for i in range(1, 6):
            for j in range(i + 1, 6):
                rows.append((i, j))
        df = pd.DataFrame(rows, columns=["sender_id", "receiver_id"])
        edges = aggregate_undirected_edges(df)
        feat = node_features_from_edges(edges, members)
        rule = ThresholdRule(method="quantile", value=0.0, quantile=0.95)
        out = detect_network_node_anomalies(
            feat, rule, contamination=0.1, random_state=0
        )
        assert out.loc[hub, "flag"] == 1 or out["score"].idxmax() == hub
        assert out["flag"].sum() >= 1

    def test_requires_two_nodes(self) -> None:
        feat = pd.DataFrame({"weighted_degree": [1.0]}, index=[1])
        rule = ThresholdRule(method="quantile", value=0.0, quantile=0.5)
        with pytest.raises(ValueError, match="at least 2 nodes"):
            detect_network_node_anomalies(feat, rule)
