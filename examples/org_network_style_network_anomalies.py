"""
Example: organizational communication graph → node-level anomalies.

Mirrors the edge construction pattern used in ``org_network_analysis``
(``NetworkAnalyzer._to_edge_list``): each row is one communication event with
``sender_id`` and ``receiver_id``; pairs are aggregated to undirected weights.

Usage (from repo root)::

    python examples/org_network_style_network_anomalies.py

This file is self-contained and does not import the Flask app; copy the
DataFrame-building logic next to your ``CommunicationRepository`` query results.
"""

from __future__ import annotations

import pandas as pd

from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows.network import (
    aggregate_undirected_edges,
    detect_network_node_anomalies,
    node_features_from_edges,
)


def main() -> None:
    # Same column names as org_network_analysis ORM / bulk APIs
    communications = pd.DataFrame(
        {
            "sender_id": [1, 2, 3, 4, 1, 1, 1, 1],
            "receiver_id": [2, 1, 2, 2, 3, 4, 2, 2],
        }
    )
    member_ids = [1, 2, 3, 4, 5]

    edges = aggregate_undirected_edges(communications)
    features = node_features_from_edges(edges, member_ids)

    # Top 15% isolation scores flagged (tune for production)
    rule = ThresholdRule(method="quantile", value=0.0, quantile=0.85)
    result = detect_network_node_anomalies(
        features, rule, contamination=0.12, random_state=0
    )

    flagged = result[result["flag"] == 1].sort_values("score", ascending=False)
    print("Node anomaly scores (higher = more unusual vs team):")
    print(result[["weighted_degree", "neighbor_count", "pagerank", "score", "flag"]])
    print("\nFlagged nodes:")
    print(flagged[["score", "weighted_degree", "neighbor_count"]])


if __name__ == "__main__":
    main()
