"""
TIMM: Temporal IMM — Influence Maximization on Continuous-Time Networks
      with (1-1/e-ε) Approximation Guarantees.

Algorithm overview:
1. Estimate lower bound on OPT via sampling
2. Compute required TRR-set count θ via martingale bound
3. Generate θ TRR-sets
4. Greedy max-coverage on TRR-sets → output seed set

Complexity:
- Time: O(θ · L) where L is average temporal walk length
- Space: O(θ · avg(|R|)) where |R| is TRR-set size
"""

import numpy as np
import time
from typing import Set, List, Tuple, Optional, Dict
from .temporal_graph import TemporalGraph
from .hawkes_model import ExponentialHawkesKernel
from .trr_sets import TRRSetGenerator
from .martingale import (
    compute_trr_set_bound,
    compute_trr_set_bound_refined,
    estimate_opt_lower_bound,
)


class TIMM:
    """
    Temporal IMM algorithm.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph with timestamped edges.
    kernel : ExponentialHawkesKernel
        Hawkes process kernel (exponential).
    horizon : float
        Time horizon for influence spread.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        graph: TemporalGraph,
        kernel: ExponentialHawkesKernel,
        horizon: float,
        seed: Optional[int] = None,
    ):
        self.graph = graph
        self.kernel = kernel
        self.horizon = horizon
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Internal state
        self.trr_generator = TRRSetGenerator(graph, kernel, horizon, seed=seed)
        self.trr_sets: List[Set[int]] = []
        self.theta: int = 0
        self.selected_seeds: List[int] = []
        self.opt_lower_bound: float = 0.0
        self.stats: Dict = {}
        self._eval_counter: int = 0  # ensures evaluate() uses independent RNGs

    def run(
        self,
        k: int,
        epsilon: float = 0.1,
        delta: float = 0.01,
        verbose: bool = True,
    ) -> Tuple[List[int], Dict]:
        """
        Run TIMM to select k seed nodes.

        Parameters
        ----------
        k : int
            Seed budget (number of seeds to select).
        epsilon : float
            Approximation error (default 0.1 = 10% error).
        delta : float
            Confidence parameter (default 0.01 = 99% confidence).
        verbose : bool
            Print progress information.

        Returns
        -------
        seeds : List[int]
            Selected seed nodes.
        stats : Dict
            Execution statistics.
        """
        n = self.graph.n
        t_start = time.time()

        # ── Phase 1: Estimate OPT lower bound ──
        if verbose:
            print(f"[TIMM] Phase 1: Estimating OPT lower bound...")
        sample_theta = min(
            compute_trr_set_bound(n, k, epsilon=0.5, delta=delta),
            10000,
        )
        sample_trr = self.trr_generator.generate_batch(sample_theta)
        self.opt_lower_bound = estimate_opt_lower_bound(n, k, sample_trr)
        t_phase1 = time.time()

        if verbose:
            print(f"  OPT lower bound: {self.opt_lower_bound:.2f}")
            print(f"  Sample TRR-sets: {sample_theta}")
            print(f"  Time: {t_phase1 - t_start:.2f}s")

        # ── Phase 2: Compute TRR-set count ──
        if verbose:
            print(f"[TIMM] Phase 2: Computing TRR-set count...")
        self.theta = compute_trr_set_bound_refined(
            n, k, epsilon, delta, self.opt_lower_bound
        )
        if verbose:
            print(f"  Required TRR-sets: {self.theta}")

        # ── Phase 3: Generate TRR-sets ──
        if verbose:
            print(f"[TIMM] Phase 3: Generating {self.theta} TRR-sets...")
        self.trr_sets = self.trr_generator.generate_batch_optimized(self.theta)
        t_phase3 = time.time()

        trr_stats = self.trr_generator.get_trr_set_stats(self.trr_sets)
        if verbose:
            print(f"  TRR-set stats: mean_size={trr_stats['mean_size']:.1f}, "
                  f"max_size={trr_stats['max_size']}")
            print(f"  Time: {t_phase3 - t_phase1:.2f}s")

        # ── Phase 4: Greedy max-coverage ──
        if verbose:
            print(f"[TIMM] Phase 4: Greedy max-coverage (k={k})...")
        self.selected_seeds, gains = self.trr_generator.greedy_max_cover(
            self.trr_sets, k
        )
        t_phase4 = time.time()

        if verbose:
            print(f"  Seeds selected: {len(self.selected_seeds)}")
            print(f"  Time: {t_phase4 - t_phase3:.2f}s")

        # ── Statistics ──
        total_time = time.time() - t_start
        self.stats = {
            "n": n,
            "m": self.graph.m,
            "k": k,
            "epsilon": epsilon,
            "delta": delta,
            "horizon": self.horizon,
            "opt_lower_bound": float(self.opt_lower_bound),
            "theta": self.theta,
            "trr_sets_generated": len(self.trr_sets),
            "trr_set_stats": trr_stats,
            "num_seeds": len(self.selected_seeds),
            "phase1_time_s": t_phase1 - t_start,
            "phase3_time_s": t_phase3 - t_phase1,
            "phase4_time_s": t_phase4 - t_phase3,
            "total_time_s": total_time,
            "estimated_influence": (
                self.trr_generator.estimate_influence(
                    set(self.selected_seeds), self.trr_sets
                )
                if self.trr_sets
                else 0.0
            ),
        }

        if verbose:
            print(f"[TIMM] Done. Total time: {total_time:.2f}s")
            print(f"  Estimated influence: {self.stats['estimated_influence']:.2f}")

        return self.selected_seeds, self.stats

    def evaluate(
        self, seeds: List[int], num_simulations: int = 100
    ) -> float:
        """
        Evaluate influence spread via Monte Carlo simulation.

        Parameters
        ----------
        seeds : List[int]
            Seed set to evaluate.
        num_simulations : int
            Number of Monte Carlo cascade simulations.

        Returns
        -------
        float
            Estimated influence spread.
        """
        from .hawkes_model import HawkesDiffusion

        # Use a derived seed so repeated evaluate() calls are independent.
        # Without this, eval_multiple() in experiments would get identical
        # results on every call, making std ≡ 0.
        eval_seed = (self.seed if self.seed is not None else 42) + self._eval_counter
        self._eval_counter += 1
        sim = HawkesDiffusion(
            self.graph,
            self.kernel,
            self.horizon,
            rng=np.random.default_rng(eval_seed),
        )
        return sim.forward_cascade(set(seeds), num_simulations)


def tiimm_example():
    """Quick example demonstrating TIMM on synthetic data."""
    from .temporal_graph import TemporalGraph

    # Build a small temporal graph
    n = 100
    edges = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        u = rng.integers(0, n)
        v = rng.integers(0, n)
        if u != v:
            edges.append((u, v, rng.exponential(10.0)))

    g = TemporalGraph.from_edge_list(edges, num_nodes=n)
    kernel = ExponentialHawkesKernel(alpha=0.5, beta=0.1, mu=0.01)
    horizon = 20.0

    timm = TIMM(g, kernel, horizon, seed=42)
    seeds, stats = timm.run(k=10, epsilon=0.1, verbose=True)

    # Evaluate
    spread = timm.evaluate(seeds, num_simulations=50)
    print(f"\nMonte Carlo verified spread: {spread:.2f}")

    return seeds, stats


if __name__ == "__main__":
    tiimm_example()
