"""
Baseline algorithms for temporal influence maximization.

Implements six baselines:
1. Static IMM — ignores temporal information
2. Snapshot IMM — discrete-time windows
3. MC-Hawkes-Greedy — continuous-time Hawkes (no guarantees)
4. CELF — greedy with lazy evaluation
5. DegreeDiscount — heuristic based on weighted degree
6. Random — random seed selection (sanity check)
"""

import numpy as np
import time
from typing import Set, List, Tuple, Optional, Dict
from collections import defaultdict
from .temporal_graph import TemporalGraph
from .hawkes_model import ExponentialHawkesKernel, HawkesDiffusion
from .trr_sets import TRRSetGenerator


class StaticIMM:
    """
    Static IMM (Tang et al., SIGMOD 2015) applied to temporal graphs.

    Aggregates all temporal edges into a static weighted graph
    (weight = number of interactions), then runs standard IMM.

    This baseline answers: "How much do we lose by ignoring time?"
    """

    def __init__(self, graph: TemporalGraph, seed: Optional[int] = None):
        self.graph = graph
        self.rng = np.random.default_rng(seed)

    def run(self, k: int, epsilon: float = 0.1) -> Tuple[List[int], Dict]:
        """
        Run static IMM on the aggregated static graph.

        Parameters
        ----------
        k : int
            Seed budget.
        epsilon : float
            Approximation error.

        Returns
        -------
        seeds : List[int]
            Selected seeds.
        stats : Dict
            Execution statistics.
        """
        t_start = time.time()

        # Build static graph with edge probabilities.
        # Probability that at least one activation occurs across
        # `count` temporal interactions, each with base prob p0.
        # Formula: p = 1 - (1 - p0)^count, capped at 0.95.
        n = self.graph.n
        base_p0 = 0.02
        edge_counts = defaultdict(int)

        for u in range(n):
            for edge in self.graph.out_edges[u]:
                key = (u, edge.v)
                edge_counts[key] += 1

        static_probs = {}
        for key, count in edge_counts.items():
            static_probs[key] = min(1.0 - (1.0 - base_p0) ** count, 0.95)

        # Simple RIS-based selection
        # (Simplified — full IMM would use martingale bounds)
        num_rr_sets = min(max(1000, n * 2), 5000)
        rr_sets = self._generate_static_rr_sets(static_probs, num_rr_sets)

        seeds, _ = self._greedy_max_cover(rr_sets, k)

        elapsed = time.time() - t_start
        return seeds, {
            "algorithm": "Static-IMM",
            "num_seeds": len(seeds),
            "num_rr_sets": num_rr_sets,
            "time_s": elapsed,
        }

    def _generate_static_rr_sets(
        self, edge_probs: dict, num_sets: int
    ) -> List[Set[int]]:
        """Generate static RR-sets from weighted graph."""
        n = self.graph.n
        rr_sets = []

        for _ in range(num_sets):
            target = self.rng.integers(0, n)
            R = {target}
            frontier = [target]

            while frontier:
                v = frontier.pop()
                for edge in self.graph.in_edges[v]:
                    key = (edge.u, v)
                    p = edge_probs.get(key, 0.1)
                    if self.rng.random() < p:
                        if edge.u not in R:
                            R.add(edge.u)
                            frontier.append(edge.u)

            rr_sets.append(R)

        return rr_sets

    def _greedy_max_cover(
        self, rr_sets: List[Set[int]], k: int
    ) -> Tuple[List[int], List[float]]:
        """Greedy max-coverage on RR-sets."""
        theta = len(rr_sets)
        covered = np.zeros(theta, dtype=bool)

        node_to_sets = {}
        for i, R in enumerate(rr_sets):
            for node in R:
                node_to_sets.setdefault(node, []).append(i)

        seeds, gains = [], []
        for _ in range(k):
            best_node, best_new = -1, 0
            for node, idxs in node_to_sets.items():
                new = sum(1 for i in idxs if not covered[i])
                if new > best_new:
                    best_new, best_node = new, node

            if best_node == -1:
                break
            seeds.append(best_node)
            gains.append(self.graph.n * best_new / theta)
            for i in node_to_sets[best_node]:
                covered[i] = True

        return seeds, gains


class SnapshotIMM:
    """
    Snapshot-based temporal IM (Ohsaka et al., AAAI 2016).

    Divides the timeline into discrete windows, builds a static
    aggregated graph for each window independently, runs RR-set
    sampling per window, and combines seed scores with exponential
    recency weighting.

    This is a faithful approximation of the original snapshot method:
    each window captures the graph state during that time interval.
    Seeds are selected based on influence scores aggregated across
    all windows, with recent windows weighted higher.
    """

    def __init__(
        self,
        graph: TemporalGraph,
        num_windows: int = 10,
        decay_lambda: float = 0.5,
        base_p0: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.graph = graph
        self.num_windows = num_windows
        self.decay_lambda = decay_lambda
        self.base_p0 = base_p0
        self.rng = np.random.default_rng(seed)

    def run(self, k: int) -> Tuple[List[int], Dict]:
        """
        Run snapshot-based IM with per-window RIS and recency-weighted
        seed scoring.

        1. Partition edges into `num_windows` equal-length time windows.
        2. For each window, build a static graph from edges within that
           window and generate RR-sets.
        3. Compute per-node influence scores per window.
        4. Aggregate scores across windows with exponential decay
           (recent windows weighted higher).
        5. Select top-k nodes by aggregated score.
        """
        t_start = time.time()
        n = self.graph.n

        # Find time range
        all_times = []
        for u in range(n):
            for e in self.graph.out_edges[u]:
                all_times.append(e.t)
        if not all_times:
            return [], {"error": "no edges"}

        t_min, t_max = min(all_times), max(all_times)
        if t_max <= t_min:
            t_max = t_min + 1.0
        window_span = (t_max - t_min) / self.num_windows

        # Partition edges by window
        window_edges: Dict[int, Dict[Tuple[int, int], int]] = {
            w: defaultdict(int) for w in range(self.num_windows)
        }
        for u in range(n):
            for e in self.graph.out_edges[u]:
                w = min(int((e.t - t_min) / window_span), self.num_windows - 1)
                window_edges[w][(u, e.v)] += 1

        # Per-window node scores (accumulated)
        node_scores = np.zeros(n, dtype=np.float64)
        num_rr_per_window = max(200, min(1000, (n * 3) // self.num_windows))

        for w in range(self.num_windows):
            edge_counts = window_edges[w]
            if not edge_counts:
                continue

            # Generate RR-sets for this window
            rr_sets = self._generate_rr_sets_from_counts(edge_counts, num_rr_per_window)

            # Compute per-node coverage
            node_cov = np.zeros(n, dtype=np.float64)
            for R in rr_sets:
                for node in R:
                    node_cov[node] += 1.0
            if len(rr_sets) > 0:
                node_cov /= len(rr_sets)

            # Exponential decay: weight = exp(-decay_lambda * (num_windows - 1 - w))
            recency_weight = np.exp(-self.decay_lambda * (self.num_windows - 1 - w))
            node_scores += node_cov * recency_weight

        # Select top-k nodes by aggregated score
        top_k = np.argpartition(-node_scores, min(k, n) - 1)[:k]
        seeds = list(top_k[np.argsort(-node_scores[top_k])])

        elapsed = time.time() - t_start
        return seeds, {
            "algorithm": "Snapshot-IMM",
            "num_windows": self.num_windows,
            "window_span": round(window_span, 3),
            "num_seeds": len(seeds),
            "rr_per_window": num_rr_per_window,
            "time_s": elapsed,
        }

    def _generate_rr_sets_from_counts(
        self, edge_counts: Dict[Tuple[int, int], int], num_sets: int
    ) -> List[Set[int]]:
        """Generate RR-sets from window-specific edge count dict."""
        n = self.graph.n
        rr_sets = []
        for _ in range(num_sets):
            target = self.rng.integers(0, n)
            R = {target}
            frontier = [target]
            while frontier:
                v = frontier.pop()
                for e in self.graph.in_edges[v]:
                    cnt = edge_counts.get((e.u, v), 0)
                    if cnt <= 0:
                        continue
                    # p = 1 - (1 - base_p0)^count, capped at 0.95
                    p = min(1.0 - (1.0 - self.base_p0) ** cnt, 0.95)
                    if self.rng.random() < p:
                        if e.u not in R:
                            R.add(e.u)
                            frontier.append(e.u)
            rr_sets.append(R)
        return rr_sets


class MCHawkesGreedy:
    """
    MC-Hawkes-Greedy: direct Monte Carlo greedy baseline under the same
    Hawkes diffusion model.

    Uses forward Monte Carlo with the Hawkes model to evaluate
    influence, then applies naive greedy selection. No approximation
    guarantees, but captures temporal dynamics.

    Key difference from TIMM: no TRR-sets, no submodularity proof,
    no (1-1/e-ε) guarantee. Pure Monte Carlo + greedy.

    Complexity: O(k·n·MC) — feasible only on small graphs (n ≤ ~2000).
    """

    def __init__(
        self,
        graph: TemporalGraph,
        kernel: ExponentialHawkesKernel,
        horizon: float,
        mc_samples: int = 200,
        seed: Optional[int] = None,
    ):
        self.graph = graph
        self.kernel = kernel
        self.horizon = horizon
        self.mc_samples = mc_samples
        self.rng = np.random.default_rng(seed)

    def run(self, k: int) -> Tuple[List[int], Dict]:
        """
        Naive greedy selection with Hawkes Monte Carlo evaluation.

        Direct MC greedy baseline under the Hawkes diffusion model:
        - Round 1: evaluate spread({u}) for every node, pick best.
        - Round i: for each unselected node u, evaluate
          spread(selected ∪ {u}), pick max marginal gain.

        No CELF acceleration; this keeps the baseline close to the standard
        greedy selection procedure used in influence maximization.
        O(k·n·MC) complexity.
        """
        t_start = time.time()
        n = self.graph.n

        diffuser = HawkesDiffusion(
            self.graph, self.kernel, self.horizon, rng=self.rng
        )

        selected: Set[int] = set()
        seeds: List[int] = []
        total_evaluations = 0

        for round_idx in range(k):
            best_node = -1
            best_gain = -1.0

            if round_idx == 0:
                # First round: evaluate spread for each singleton seed
                for u in range(n):
                    sp = diffuser.forward_cascade({u}, self.mc_samples)
                    total_evaluations += 1
                    if sp > best_gain:
                        best_gain = sp
                        best_node = u
            else:
                # Subsequent rounds: evaluate marginal gain for each candidate
                current_spread = diffuser.forward_cascade(
                    selected, self.mc_samples
                )
                total_evaluations += 1

                for u in range(n):
                    if u in selected:
                        continue
                    sp = diffuser.forward_cascade(
                        selected | {u}, self.mc_samples
                    )
                    total_evaluations += 1
                    gain = sp - current_spread
                    if gain > best_gain:
                        best_gain = gain
                        best_node = u

            if best_node == -1:
                break
            seeds.append(best_node)
            selected.add(best_node)

        elapsed = time.time() - t_start
        return seeds, {
            "algorithm": "MC-Hawkes-Greedy",
            "num_seeds": len(seeds),
            "mc_samples": self.mc_samples,
            "total_evaluations": total_evaluations,
            "time_s": elapsed,
        }


class CELF:
    """
    CELF (Leskovec et al., KDD 2007): Cost-Effective Lazy Forward.

    Proper lazy greedy selection exploiting submodularity:
    1. Initially evaluate f({u}) for all nodes, store in max-heap.
    2. Pop the node with highest marginal gain.
    3. Re-evaluate its gain against the CURRENT seed set.
    4. If it's still the highest, select it. Otherwise, push it
       back with the updated (lower) gain and repeat.
    5. Continue until k seeds are selected.

    This achieves ~700× speedup over naive greedy in practice
    by avoiding re-evaluation of nodes whose marginal gain
    hasn't changed.
    """

    def __init__(self, graph: TemporalGraph, seed: Optional[int] = None):
        self.graph = graph
        self.rng = np.random.default_rng(seed)

    def run(self, k: int, mc_samples: int = 200) -> Tuple[List[int], Dict]:
        """CELF selection with proper lazy evaluation on static graph."""
        import heapq
        t_start = time.time()
        n = self.graph.n

        # Static edge probabilities
        edge_probs = {}
        for u in range(n):
            for e in self.graph.out_edges[u]:
                key = (u, e.v)
                edge_probs[key] = min(edge_probs.get(key, 0) + 0.1, 0.99)

        def simulate_static(seeds: Set[int]) -> float:
            """Static IC model Monte Carlo."""
            total = 0.0
            for _ in range(mc_samples):
                active = set(seeds)
                newly_active = list(seeds)
                while newly_active:
                    u = newly_active.pop()
                    for e in self.graph.out_edges[u]:
                        if e.v not in active:
                            p = edge_probs.get((u, e.v), 0.1)
                            if self.rng.random() < p:
                                active.add(e.v)
                                newly_active.append(e.v)
                total += len(active)
            return total / mc_samples

        seeds = []
        current_set: Set[int] = set()
        evaluations = 0
        heap_version = 0  # incremented each time a seed is selected

        # Phase 1: evaluate marginal gain of each singleton
        # Store as (-gain, node, version_when_evaluated) in max-heap
        heap = []
        for u in range(n):
            gain = simulate_static({u})
            evaluations += 1
            # version=heap_version means "fresh against current seed set"
            heapq.heappush(heap, (-gain, u, heap_version))

        # Phase 2: CELF lazy selection with correct version tracking
        while len(seeds) < k and heap:
            # Pop the node with highest (possibly stale) gain
            neg_gain, u, when_evaluated = heapq.heappop(heap)

            if when_evaluated == heap_version:
                # Gain was computed against the CURRENT seed set → select it
                seeds.append(u)
                current_set.add(u)
                heap_version += 1  # seed set changed → all other entries are now stale
                continue

            # Gain is stale — re-evaluate against current seed set
            if not current_set:
                new_gain = simulate_static({u})
            else:
                base = simulate_static(current_set)
                evaluations += 1
                new_with = simulate_static(current_set | {u})
                new_gain = new_with - base
            evaluations += 1

            # Push back with the updated gain and the CURRENT version
            heapq.heappush(heap, (-new_gain, u, heap_version))

        elapsed = time.time() - t_start
        return seeds, {
            "algorithm": "CELF",
            "num_seeds": len(seeds),
            "mc_samples": mc_samples,
            "evaluations": evaluations,
            "time_s": elapsed,
        }


class DegreeDiscount:
    """
    DegreeDiscount (Chen et al., KDD 2009): heuristic seed selection.

    Iteratively picks the node with highest (weighted) out-degree,
    then discounts its neighbors' effective degree. Very fast,
    no theoretical guarantee, but often close to greedy quality
    on static graphs.

    Complexity: O(k·log n + m).
    """

    def __init__(self, graph: TemporalGraph, seed: Optional[int] = None):
        self.graph = graph
        self.rng = np.random.default_rng(seed)

    def run(self, k: int) -> Tuple[List[int], Dict]:
        """
        DegreeDiscount seed selection on aggregated temporal graph.

        Edge weights are edge occurrence counts (sum of temporal
        interactions between each (u,v) pair).
        """
        t_start = time.time()
        n = self.graph.n

        # Build weighted out-degree from temporal edges
        out_degree = np.zeros(n, dtype=np.float64)
        neighbors: Dict[int, Set[int]] = defaultdict(set)
        edge_weight: Dict[Tuple[int, int], float] = defaultdict(float)

        for u in range(n):
            for e in self.graph.out_edges[u]:
                edge_weight[(u, e.v)] += 1.0
                neighbors[u].add(e.v)

        for u in range(n):
            out_degree[u] = sum(
                edge_weight[(u, v)] for v in neighbors[u]
            )

        # DegreeDiscount selection
        selected: Set[int] = set()
        seeds: List[int] = []
        dd = out_degree.copy()  # discounted degree
        total_degree = out_degree.sum()

        for _ in range(k):
            # Pick unselected node with highest discounted degree
            best_node = -1
            best_dd = -1.0
            for u in range(n):
                if u not in selected and dd[u] > best_dd:
                    best_dd = dd[u]
                    best_node = u

            if best_node == -1:
                break
            seeds.append(best_node)
            selected.add(best_node)

            # Discount neighbors: dd[v] -= 2 * edge_weight(best_node, v)
            for v in neighbors[best_node]:
                if v not in selected:
                    w = edge_weight[(best_node, v)]
                    dd[v] = max(0.0, dd[v] - 2.0 * w)

        elapsed = time.time() - t_start
        return seeds, {
            "algorithm": "DegreeDiscount",
            "num_seeds": len(seeds),
            "time_s": elapsed,
        }


class TemporalDegree:
    """
    Temporal weighted degree baseline.

    Scores nodes by time-remaining weighted out-degree:
        score(v) = Σ_{v→w} exp(-β · max(0, H − t_{v→w}))
    where H is the influence horizon. Edges with more remaining time
    (smaller t) get higher weight — they have longer to propagate.
    β matches the Hawkes kernel.

    Lightweight heuristic — single pass over edges, no sampling needed.
    """

    def __init__(self, graph: TemporalGraph, beta: float = 1.0,
                 horizon: float = 100.0, seed: Optional[int] = None):
        self.graph = graph
        self.beta = beta
        self.horizon = horizon
        self.seed = seed

    def run(self, k: int) -> Tuple[List[int], Dict]:
        import time as _time
        t_start = _time.time()
        n = self.graph.n
        H = self.horizon

        # Compute temporal degree: weight edges by remaining cascade time.
        # Earlier edges (smaller t) → more remaining time → higher weight.
        scores = np.zeros(n, dtype=np.float64)
        for v in range(n):
            for e in self.graph.out_edges[v]:
                remaining = H - e.t
                if remaining > 0:
                    scores[v] += np.exp(-self.beta * remaining)

        # Top-k by score
        if k >= n:
            seeds = list(range(n))
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            seeds = list(top_indices[:k])

        elapsed = _time.time() - t_start
        return seeds, {
            "algorithm": "TemporalDegree",
            "num_seeds": len(seeds),
            "time_s": elapsed,
            "horizon": float(H),
        }


class RandomBaseline:
    """
    Random baseline: selects k nodes uniformly at random.

    Sanity check — any non-trivial algorithm should outperform this.
    """

    def __init__(self, graph: TemporalGraph, seed: Optional[int] = None):
        self.graph = graph
        self.rng = np.random.default_rng(seed)

    def run(self, k: int) -> Tuple[List[int], Dict]:
        """Select k nodes uniformly at random without replacement."""
        t_start = time.time()
        n = self.graph.n
        k = min(k, n)
        seeds = list(self.rng.choice(n, size=k, replace=False))
        elapsed = time.time() - t_start
        return seeds, {
            "algorithm": "Random",
            "num_seeds": len(seeds),
            "time_s": elapsed,
        }
