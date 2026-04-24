"""Property-style tests for network aggregation and PageRank."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from anomsmith.workflows.network import (
    _build_neighbor_lists,
    _pagerank_undirected,
    aggregate_undirected_edges,
    edge_features_from_edges,
)

# CI-friendly example counts (property tests can be slower locally with max_examples)
_SETTINGS = settings(max_examples=60, deadline=None)


def _edges_from_pairs(
    n: int, pairs: list[tuple[int, int]], weights: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One undirected edge per (u,v,w) with u < v in index space."""
    src_list: list[int] = []
    dst_list: list[int] = []
    w_list: list[float] = []
    for (a, b), w in zip(pairs, weights):
        u, v = (a, b) if a < b else (b, a)
        src_list.append(u)
        dst_list.append(v)
        w_list.append(float(w))
    if not src_list:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )
    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(dst_list, dtype=np.int64),
        np.asarray(w_list, dtype=np.float64),
    )


class TestAggregateUndirectedEdgesProperties:
    @_SETTINGS
    @given(
        st.lists(
            st.tuples(st.integers(0, 12), st.integers(0, 12)),
            max_size=80,
        )
    )
    def test_row_permutation_invariant(self, pairs: list[tuple[int, int]]) -> None:
        pairs_nontrivial = [(a, b) for a, b in pairs if a != b]
        if not pairs_nontrivial:
            comms = pd.DataFrame({"sender_id": [], "receiver_id": []})
            assert aggregate_undirected_edges(comms).empty
            return

        df = pd.DataFrame(pairs_nontrivial, columns=["sender_id", "receiver_id"])
        base = aggregate_undirected_edges(df)
        perm = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        permuted = aggregate_undirected_edges(perm)
        pd.testing.assert_frame_equal(
            base.sort_values(["u", "v"]).reset_index(drop=True),
            permuted.sort_values(["u", "v"]).reset_index(drop=True),
        )

    @_SETTINGS
    @given(
        st.lists(
            st.tuples(st.integers(0, 10), st.integers(0, 10)).filter(
                lambda t: t[0] != t[1]
            ),
            min_size=1,
            max_size=60,
        )
    )
    def test_swap_endpoints_invariant(self, pairs: list[tuple[int, int]]) -> None:
        df = pd.DataFrame(pairs, columns=["sender_id", "receiver_id"])
        swapped = pd.DataFrame(
            {"sender_id": df["receiver_id"], "receiver_id": df["sender_id"]}
        )
        pd.testing.assert_frame_equal(
            aggregate_undirected_edges(df).sort_values(["u", "v"]).reset_index(
                drop=True
            ),
            aggregate_undirected_edges(swapped)
            .sort_values(["u", "v"])
            .reset_index(drop=True),
        )

    @_SETTINGS
    @given(
        st.lists(
            st.tuples(st.integers(0, 8), st.integers(0, 8)),
            min_size=1,
            max_size=50,
        ).map(lambda xs: [(a, b) for a, b in xs if a != b])
    )
    def test_total_weight_equals_event_count(self, pairs: list[tuple[int, int]]) -> None:
        df = pd.DataFrame(pairs, columns=["sender_id", "receiver_id"])
        agg = aggregate_undirected_edges(df)
        assert int(agg["weight"].sum()) == len(df)

    @_SETTINGS
    @given(
        st.lists(
            st.tuples(st.integers(0, 9), st.integers(0, 9)),
            max_size=40,
        ).map(lambda xs: [(a, b) for a, b in xs if a != b])
    )
    def test_canonical_endpoints_u_le_v(self, pairs: list[tuple[int, int]]) -> None:
        df = pd.DataFrame(pairs, columns=["sender_id", "receiver_id"])
        agg = aggregate_undirected_edges(df)
        if agg.empty:
            return
        assert (agg["u"] <= agg["v"]).all()


class TestPageRankProperties:
    def test_mass_conservation_single_node_isolated(self) -> None:
        n = 1
        neighbors: list[list[tuple[int, float]]] = [[]]
        strength: np.ndarray = np.zeros(1, dtype=np.float64)
        r = _pagerank_undirected(n, neighbors, strength)
        assert r.shape == (1,)
        assert float(r.sum()) == pytest.approx(1.0)
        assert (r >= 0).all()

    @_SETTINGS
    @given(st.integers(2, 14))
    def test_mass_conservation_random_sparse_graph(self, n: int) -> None:
        rng = np.random.default_rng(n * 31 + 7)
        edges: list[tuple[int, int]] = []
        for _ in range(max(1, n + rng.integers(0, 2 * n))):
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n))
            if a != b:
                edges.append((a, b) if a < b else (b, a))
        pairs = list(dict.fromkeys(edges))
        weights = [int(rng.integers(1, 4)) for _ in pairs]
        src, dst, w = _edges_from_pairs(n, pairs, weights)
        neighbors, strength = _build_neighbor_lists(n, src, dst, w)
        r = _pagerank_undirected(n, neighbors, strength)
        assert float(r.sum()) == pytest.approx(1.0, abs=1e-10, rel=1e-9)
        assert (r >= -1e-15).all()

    @_SETTINGS
    @given(st.integers(2, 12))
    def test_symmetry_complete_graph_uniform(self, n: int) -> None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        weights = [1] * len(pairs)
        src, dst, w = _edges_from_pairs(n, pairs, weights)
        neighbors, strength = _build_neighbor_lists(n, src, dst, w)
        r = _pagerank_undirected(n, neighbors, strength)
        assert np.allclose(r, 1.0 / n, rtol=1e-5, atol=1e-5)

    @_SETTINGS
    @given(st.integers(3, 14))
    def test_symmetry_ring_uniform(self, n: int) -> None:
        pairs = [(i, (i + 1) % n) for i in range(n)]
        pairs = [(min(a, b), max(a, b)) for a, b in pairs]
        pairs = sorted(set(pairs))
        weights = [1] * len(pairs)
        src, dst, w = _edges_from_pairs(n, pairs, weights)
        neighbors, strength = _build_neighbor_lists(n, src, dst, w)
        r = _pagerank_undirected(n, neighbors, strength)
        assert np.allclose(r, 1.0 / n, rtol=1e-4, atol=1e-4)

    def test_symmetry_path_endpoints_equal(self) -> None:
        """Path 0—1—2 on three nodes: endpoints are automorphic → equal PageRank."""
        n = 3
        pairs = [(0, 1), (1, 2)]
        weights = [1, 1]
        src, dst, w = _edges_from_pairs(n, pairs, weights)
        neighbors, strength = _build_neighbor_lists(n, src, dst, w)
        r = _pagerank_undirected(n, neighbors, strength)
        assert float(r[0]) == pytest.approx(float(r[2]), rel=1e-5, abs=1e-5)
        assert float(r[1]) > float(r[0])

    def test_label_permutation_isomorphism(self) -> None:
        """Relabeling nodes permutes PageRank vector consistently."""
        n = 4
        pairs = [(0, 1), (1, 2), (2, 3)]
        weights = [1, 1, 1]
        src, dst, w = _edges_from_pairs(n, pairs, weights)
        neighbors, strength = _build_neighbor_lists(n, src, dst, w)
        r_orig = _pagerank_undirected(n, neighbors, strength)
        perm = np.array([3, 0, 1, 2])
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n)
        pairs_p = []
        for a, b in [(0, 1), (1, 2), (2, 3)]:
            pa, pb = int(perm[a]), int(perm[b])
            pairs_p.append((pa, pb) if pa < pb else (pb, pa))
        src_p, dst_p, w_p = _edges_from_pairs(n, pairs_p, weights)
        nbr_p, str_p = _build_neighbor_lists(n, src_p, dst_p, w_p)
        r_perm = _pagerank_undirected(n, nbr_p, str_p)
        expected = r_orig[inv]
        np.testing.assert_allclose(r_perm, expected, rtol=1e-5, atol=1e-5)


class TestShareOfEndpointVolumeProperties:
    @_SETTINGS
    @given(
        st.lists(
            st.tuples(st.integers(0, 10), st.integers(0, 10)),
            max_size=50,
        )
    )
    def test_share_in_zero_one_for_roster_endpoints(self, pairs: list[tuple[int, int]]) -> None:
        pairs_nontrivial = [(a, b) for a, b in pairs if a != b]
        if not pairs_nontrivial:
            return
        df = pd.DataFrame(pairs_nontrivial, columns=["sender_id", "receiver_id"])
        edges = aggregate_undirected_edges(df)
        if edges.empty:
            return
        nodes = sorted(set(df["sender_id"]) | set(df["receiver_id"]))
        ef = edge_features_from_edges(edges, nodes)
        share = ef["share_of_endpoint_volume"].to_numpy(dtype=np.float64)
        assert (share >= -1e-12).all()
        assert (share <= 1.0 + 1e-9).all()

    @_SETTINGS
    @given(
        st.lists(
            st.tuples(st.integers(0, 6), st.integers(0, 6)),
            min_size=1,
            max_size=30,
        ).map(lambda xs: [(a, b) for a, b in xs if a != b])
    )
    def test_share_invariant_under_row_shuffle(self, pairs: list[tuple[int, int]]) -> None:
        if len(pairs) < 1:
            return
        base = pd.DataFrame(pairs, columns=["sender_id", "receiver_id"])
        edges_a = aggregate_undirected_edges(base)
        nodes = sorted(set(base["sender_id"]) | set(base["receiver_id"]))
        shuffled = base.sample(frac=1.0, random_state=1).reset_index(drop=True)
        edges_b = aggregate_undirected_edges(shuffled)
        ea = edge_features_from_edges(edges_a, nodes).sort_index()
        eb = edge_features_from_edges(edges_b, nodes).sort_index()
        pd.testing.assert_frame_equal(ea, eb)
