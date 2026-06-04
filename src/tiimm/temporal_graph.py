"""
Temporal Graph data structure for TIMM.

Represents a directed graph where each edge carries a sequence of
timestamped interactions. Supports temporal neighbor queries
needed for TRR-set generation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TemporalEdge:
    """A single timestamped interaction on an edge."""
    u: int          # source node
    v: int          # target node
    t: float        # timestamp


class TemporalGraph:
    """
    Directed temporal graph with timestamped edges.

    Stores edges in both forward (outgoing) and backward (incoming)
    adjacency for efficient TRR-set sampling (which walks backwards
    along incoming edges from a target node).

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph (0-indexed).

    Attributes
    ----------
    n : int
        Number of nodes.
    m : int
        Number of temporal edges.
    in_edges : Dict[int, List[TemporalEdge]]
        Incoming edges keyed by target node.
    out_edges : Dict[int, List[TemporalEdge]]
        Outgoing edges keyed by source node.
    """

    def __init__(self, num_nodes: int):
        self.n = num_nodes
        self.m = 0
        self.in_edges: Dict[int, List[TemporalEdge]] = defaultdict(list)
        self.out_edges: Dict[int, List[TemporalEdge]] = defaultdict(list)

    def add_edge(self, u: int, v: int, t: float) -> None:
        """Add a timestamped directed edge u -> v at time t."""
        e = TemporalEdge(u=u, v=v, t=t)
        self.out_edges[u].append(e)
        self.in_edges[v].append(e)
        self.m += 1

    def finalize(self) -> None:
        """
        Sort all edge lists by timestamp and build numpy arrays for performance.
        Must be called after all edges are added and before sampling.
        """
        for node in range(self.n):
            # Sort outgoing by time
            self.out_edges[node].sort(key=lambda e: e.t)
            # Sort incoming by time
            self.in_edges[node].sort(key=lambda e: e.t)

    def get_in_neighbors(
        self, v: int, t_before: float
    ) -> List[Tuple[int, float]]:
        """
        Return (u, t) pairs for edges u -> v where t < t_before.

        Used for backward temporal walks: from target v at time t_before,
        which upstream nodes could have influenced it?
        """
        result = []
        for e in self.in_edges[v]:
            if e.t < t_before:
                result.append((e.u, e.t))
            else:
                break  # edges sorted by time ascending
        return result

    # get_out_neighbors() removed — forward simulation iterates out_edges
    # directly in hawkes_model.py:_simulate_single for efficiency.

    @classmethod
    def from_edge_list(
        cls, edges: List[Tuple[int, int, float]], num_nodes: Optional[int] = None
    ) -> "TemporalGraph":
        """
        Build a TemporalGraph from a list of (u, v, t) tuples.

        Parameters
        ----------
        edges : List[Tuple[int, int, float]]
            List of (source, target, timestamp) triples.
        num_nodes : int, optional
            If not given, inferred from max node id + 1.
        """
        if num_nodes is None:
            max_id = max(max(u, v) for u, v, _ in edges)
            num_nodes = max_id + 1
        g = TemporalGraph(num_nodes)
        for u, v, t in edges:
            g.add_edge(u, v, t)
        g.finalize()
        return g

    def get_temporal_degree_stats(self) -> Dict[str, float]:
        """Return summary statistics for temporal degrees."""
        in_degs = [len(self.in_edges[v]) for v in range(self.n)]
        out_degs = [len(self.out_edges[u]) for u in range(self.n)]
        return {
            "nodes": self.n,
            "temporal_edges": self.m,
            "mean_in_deg": np.mean(in_degs),
            "max_in_deg": np.max(in_degs),
            "mean_out_deg": np.mean(out_degs),
            "max_out_deg": np.max(out_degs),
        }
