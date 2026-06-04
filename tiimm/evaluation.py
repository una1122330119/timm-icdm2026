"""
Experiment harness and evaluation metrics for TIMM.

Runs head-to-head comparisons between TIMM and baselines,
ablation studies, and parameter sensitivity analyses.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

from .temporal_graph import TemporalGraph
from .hawkes_model import ExponentialHawkesKernel, HawkesDiffusion
from .trr_sets import TRRSetGenerator
from .timms import TIMM
from .baselines import (
    StaticIMM, SnapshotIMM, MCHawkesGreedy, CELF,
    DegreeDiscount, RandomBaseline,
)


@dataclass
class ExperimentResult:
    """Single experiment run result."""
    algorithm: str
    k: int
    seeds: List[int]
    influence_spread: float
    time_s: float
    memory_mb: float
    extra: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """
    Run a complete experiment suite comparing TIMM against baselines.

    Parameters
    ----------
    graph : TemporalGraph
    kernel : ExponentialHawkesKernel
    horizon : float
    mc_eval_samples : int
        Monte Carlo samples for final evaluation (should be high, e.g. 10000).
    seed : int
    """

    def __init__(
        self,
        graph: TemporalGraph,
        kernel: ExponentialHawkesKernel,
        horizon: float,
        mc_eval_samples: int = 1000,
        seed: int = 42,
    ):
        self.graph = graph
        self.kernel = kernel
        self.horizon = horizon
        self.mc_eval_samples = mc_eval_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.results: List[ExperimentResult] = []

    def run_all(
        self,
        k_values: List[int] = None,
        epsilon: float = 0.1,
    ) -> List[ExperimentResult]:
        """Run all algorithms across all k values."""
        if k_values is None:
            k_values = [10, 20, 50]

        for k in k_values:
            print(f"\n{'='*60}")
            print(f"k = {k}")
            print(f"{'='*60}")

            # ── TIMM ──
            print("\n[TIMM]")
            timm = TIMM(self.graph, self.kernel, self.horizon, seed=self.seed)
            seeds, stats = timm.run(k=k, epsilon=epsilon, verbose=True)
            spread = timm.evaluate(seeds, num_simulations=self.mc_eval_samples)
            self.results.append(ExperimentResult(
                algorithm="TIMM",
                k=k,
                seeds=seeds,
                influence_spread=spread,
                time_s=stats["total_time_s"],
                memory_mb=0.0,
                extra=stats,
            ))
            print(f"  Spread: {spread:.2f}")

            # ── Static IMM ──
            print("\n[Static IMM]")
            simm = StaticIMM(self.graph, seed=self.seed)
            s_seeds, s_stats = simm.run(k=k)
            s_spread = timm.evaluate(s_seeds, num_simulations=self.mc_eval_samples)
            self.results.append(ExperimentResult(
                algorithm="Static-IMM",
                k=k,
                seeds=s_seeds,
                influence_spread=s_spread,
                time_s=s_stats["time_s"],
                memory_mb=0.0,
                extra=s_stats,
            ))
            print(f"  Spread: {s_spread:.2f}")

            # ── Snapshot IMM ──
            print("\n[Snapshot IMM]")
            snap = SnapshotIMM(self.graph, seed=self.seed)
            n_seeds, n_stats = snap.run(k=k)
            n_spread = timm.evaluate(n_seeds, num_simulations=self.mc_eval_samples)
            self.results.append(ExperimentResult(
                algorithm="Snapshot-IMM",
                k=k,
                seeds=n_seeds,
                influence_spread=n_spread,
                time_s=n_stats["time_s"],
                memory_mb=0.0,
                extra=n_stats,
            ))
            print(f"  Spread: {n_spread:.2f}")

            # ── MC-Hawkes-Greedy ──
            print("\n[MC-Hawkes-Greedy]")
            mc_hawkes = MCHawkesGreedy(
                self.graph, self.kernel, self.horizon,
                mc_samples=200, seed=self.seed,
            )
            mc_hawkes_seeds, mc_hawkes_stats = mc_hawkes.run(k=k)
            mc_hawkes_spread = timm.evaluate(
                mc_hawkes_seeds, num_simulations=self.mc_eval_samples
            )
            self.results.append(ExperimentResult(
                algorithm="MC-Hawkes-Greedy",
                k=k,
                seeds=mc_hawkes_seeds,
                influence_spread=mc_hawkes_spread,
                time_s=mc_hawkes_stats["time_s"],
                memory_mb=0.0,
                extra=mc_hawkes_stats,
            ))
            print(f"  Spread: {w_spread:.2f}")

            # ── CELF ──
            print("\n[CELF]")
            celf = CELF(self.graph, seed=self.seed)
            c_seeds, c_stats = celf.run(k=k, mc_samples=200)
            c_spread = timm.evaluate(c_seeds, num_simulations=self.mc_eval_samples)
            self.results.append(ExperimentResult(
                algorithm="CELF",
                k=k,
                seeds=c_seeds,
                influence_spread=c_spread,
                time_s=c_stats["time_s"],
                memory_mb=0.0,
                extra=c_stats,
            ))
            print(f"  Spread: {c_spread:.2f}")

        return self.results

    def run_ablation_trr_count(
        self, k: int = 20, epsilon: float = 0.1
    ) -> List[ExperimentResult]:
        """
        Ablation: TRR-set count — with vs without martingale bound.

        Compares:
        - TIMM (full) with martingale-determined theta
        - theta/2 (half the TRR-sets)
        - theta*2 (double the TRR-sets)
        """
        n = self.graph.n

        # Get the base theta from TIMM
        timm = TIMM(self.graph, self.kernel, self.horizon, seed=self.seed)
        _, stats = timm.run(k=k, epsilon=epsilon, verbose=False)
        base_theta = stats["theta"]

        results = []
        trr_gen = TRRSetGenerator(
            self.graph, self.kernel, self.horizon, seed=self.seed
        )

        for label, theta in [("θ/2", base_theta // 2),
                              ("θ (TIMM)", base_theta),
                              ("2θ", base_theta * 2)]:
            print(f"\n[Ablation] {label}: θ = {theta}")
            t0 = time.time()
            trr_sets = trr_gen.generate_batch(theta)
            seeds, _ = trr_gen.greedy_max_cover(trr_sets, k)
            elapsed = time.time() - t0
            spread = timm.evaluate(seeds, num_simulations=self.mc_eval_samples)
            results.append(ExperimentResult(
                algorithm=f"Ablation-TRR-{label}",
                k=k,
                seeds=seeds,
                influence_spread=spread,
                time_s=elapsed,
                memory_mb=0.0,
                extra={"theta": theta},
            ))
            print(f"  Spread: {spread:.2f}, Time: {elapsed:.2f}s")

        return results

    def run_ablation_temporal_vs_static_rr(
        self, k: int = 20, epsilon: float = 0.1
    ) -> List[ExperimentResult]:
        """
        Ablation: Temporal RR-sets vs static RR-sets.

        Compares seed quality when using the same number of
        TRR-sets vs static RR-sets (ignoring temporal information).
        """
        n = self.graph.n
        theta = 2000  # fixed for controlled comparison

        trr_gen = TRRSetGenerator(
            self.graph, self.kernel, self.horizon, seed=self.seed
        )

        # Temporal RR-sets
        print("\n[Ablation] Temporal RR-sets")
        trr_sets = trr_gen.generate_batch(theta)
        t_seeds, _ = trr_gen.greedy_max_cover(trr_sets, k)

        # Static RR-sets (ignoring time)
        print("[Ablation] Static RR-sets")
        static_rr_sets = self._generate_static_rr_sets(theta)
        s_seeds, _ = trr_gen.greedy_max_cover(static_rr_sets, k)

        # Evaluate both
        timm = TIMM(self.graph, self.kernel, self.horizon, seed=self.seed)
        t_spread = timm.evaluate(t_seeds, num_simulations=self.mc_eval_samples)
        s_spread = timm.evaluate(s_seeds, num_simulations=self.mc_eval_samples)

        results = [
            ExperimentResult(
                algorithm="Ablation-Temporal-RR", k=k,
                seeds=t_seeds, influence_spread=t_spread,
                time_s=0.0, memory_mb=0.0,
                extra={"theta": theta},
            ),
            ExperimentResult(
                algorithm="Ablation-Static-RR", k=k,
                seeds=s_seeds, influence_spread=s_spread,
                time_s=0.0, memory_mb=0.0,
                extra={"theta": theta},
            ),
        ]

        print(f"  Temporal RR spread: {t_spread:.2f}")
        print(f"  Static RR spread:    {s_spread:.2f}")
        print(f"  Gain from temporal:  {(t_spread - s_spread) / max(s_spread, 1) * 100:.1f}%")

        return results

    def _generate_static_rr_sets(self, num_sets: int) -> List[set]:
        """Generate static RR-sets (all edges equal, ignoring timestamps)."""
        n = self.graph.n
        rr_sets = []
        for _ in range(num_sets):
            target = self.rng.integers(0, n)
            R = {target}
            frontier = [target]
            while frontier:
                v = frontier.pop()
                for e in self.graph.in_edges[v]:
                    if self.rng.random() < 0.1:  # fixed probability
                        if e.u not in R:
                            R.add(e.u)
                            frontier.append(e.u)
            rr_sets.append(R)
        return rr_sets

    def run_parameter_sensitivity(
        self,
        k: int = 20,
        param_name: str = "epsilon",
        param_values: List[float] = None,
    ) -> List[ExperimentResult]:
        """
        Parameter sensitivity analysis.

        Parameters
        ----------
        param_name : str
            One of: "epsilon", "alpha", "beta"
        param_values : List[float]
            Values to sweep.
        """
        if param_values is None:
            if param_name == "epsilon":
                param_values = [0.01, 0.05, 0.1, 0.2]
            elif param_name == "alpha":
                param_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            elif param_name == "beta":
                param_values = [0.01, 0.05, 0.1, 0.5, 1.0]
            else:
                raise ValueError(f"Unknown param: {param_name}")

        results = []
        for val in param_values:
            print(f"\n[Sensitivity] {param_name} = {val}")

            if param_name == "epsilon":
                kernel = self.kernel
                timm = TIMM(self.graph, kernel, self.horizon, seed=self.seed)
                seeds, stats = timm.run(k=k, epsilon=val, verbose=False)
                spread = timm.evaluate(seeds, self.mc_eval_samples)
                results.append(ExperimentResult(
                    algorithm=f"Sens-ε={val}", k=k,
                    seeds=seeds, influence_spread=spread,
                    time_s=stats["total_time_s"], memory_mb=0.0,
                    extra={param_name: val, "theta": stats["theta"]},
                ))

            elif param_name in ("alpha", "beta"):
                kw = {"alpha": self.kernel.alpha, "beta": self.kernel.beta}
                kw[param_name] = val
                kernel = ExponentialHawkesKernel(**kw)
                timm = TIMM(self.graph, kernel, self.horizon, seed=self.seed)
                seeds, stats = timm.run(k=k, verbose=False)
                spread = timm.evaluate(seeds, self.mc_eval_samples)
                results.append(ExperimentResult(
                    algorithm=f"Sens-{param_name}={val}", k=k,
                    seeds=seeds, influence_spread=spread,
                    time_s=stats["total_time_s"], memory_mb=0.0,
                    extra={param_name: val},
                ))

            print(f"  Spread: {spread:.2f}")

        return results

    def summary(self) -> Dict:
        """Generate summary statistics of all results."""
        if not self.results:
            return {"error": "no results"}

        df = {}
        for r in self.results:
            if r.algorithm not in df:
                df[r.algorithm] = []
            df[r.algorithm].append(r.influence_spread)

        summary = {}
        tiimm_spread = np.mean(df.get("TIMM", [0]))

        for algo, spreads in df.items():
            mean_s = np.mean(spreads)
            summary[algo] = {
                "mean_spread": mean_s,
                "std_spread": np.std(spreads),
                "relative_to_timm": mean_s / max(tiimm_spread, 1),
            }

        return summary

    def print_summary(self):
        """Pretty-print results summary."""
        summ = self.summary()
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Algorithm':<25} {'Spread':>10} {'vs TIMM':>10}")
        print(f"{'-'*45}")
        for algo, s in summ.items():
            print(f"{algo:<25} {s['mean_spread']:>10.2f} {s['relative_to_timm']:>9.2%}")
        print(f"{'='*70}")

    def to_json(self, path: str = "experiment_results.json"):
        """Save results to JSON."""
        data = [asdict(r) for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Results saved to {path}")
