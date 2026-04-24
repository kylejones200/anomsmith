Network anomaly workflows
===========================

These workflows support **organizational and communication graphs**: undirected
edge counts from sender/receiver event tables, optional timestamps for temporal
patterns, and isolation-forest scoring at **node**, **edge (dyad)**, or
**temporal profile** granularity.

They are designed to align with patterns such as ``org_network_analysis``
(aggregating ``(min(sender), max(receiver))`` weights and a fixed member roster).

Static graph (snapshot)
-----------------------

1. :func:`~anomsmith.workflows.network.aggregate_undirected_edges` — collapse
   event rows into ``u``, ``v``, ``weight``.
2. :func:`~anomsmith.workflows.network.node_features_from_edges` — per-node
   ``weighted_degree``, ``neighbor_count``, ``pagerank``.
3. :func:`~anomsmith.workflows.network.detect_network_node_anomalies` — isolation
   forest + :class:`~anomsmith.primitives.thresholding.ThresholdRule`.

Dyads (which links are unusual?)
--------------------------------

1. Build aggregated edges as above.
2. :func:`~anomsmith.workflows.network.edge_features_from_edges` — ``weight``,
   ``share_of_endpoint_volume``, ``log1p_weight``.
3. :func:`~anomsmith.workflows.network.detect_network_edge_anomalies`.

Temporal profiles (who spikes when?)
------------------------------------

1. :func:`~anomsmith.workflows.network.node_touch_counts_by_bin` — rows = nodes,
   columns = floored time bins, values = send + receive counts per bin.
2. :func:`~anomsmith.workflows.network.detect_network_temporal_node_anomalies` —
   each node's vector of bin counts is a feature row for isolation forest.

Core paths use **numpy**, **pandas**, and **scikit-learn** only. Optional
**NetworkX** metrics (betweenness, closeness, eigenvector) are available via
``pip install 'anomsmith[network]'`` and :func:`~anomsmith.workflows.network.node_graph_metrics_networkx`.

See also :mod:`anomsmith.workflows.network` in the API reference.
