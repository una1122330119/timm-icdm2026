"""
Hawkes Process Diffusion Model for Temporal IM.

Models influence spread as a self-exciting point process with
exponential kernel φ(t) = α * exp(-β * t).

Key property enabling the submodularity proof:
The exponential kernel is memoryless — the future rate depends
only on the current state, not the full history.
"""

import numpy as np
from typing import Set, Dict, Tuple, Optional, List
import heapq
from collections import deque
from .temporal_graph import TemporalGraph


class ExponentialHawkesKernel:
    """
    Exponential Hawkes kernel: φ(Δt) = α * exp(-β * Δt) for Δt ≥ 0.

    Parameters
    ----------
    alpha : float
        Branching ratio — expected number of secondary activations
        triggered by one activation. Must be in (0, 1) for stability.
    beta : float
        Decay rate — controls how fast influence decays over time.
        Higher β = faster decay.
    mu : float, optional
        Baseline intensity (exogenous events). Default 0.0.
    """

    def __init__(self, alpha: float, beta: float, mu: float = 0.0):
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.alpha = alpha
        self.beta = beta
        self.mu = mu

    def intensity(self, delta_t: float) -> float:
        """Kernel value at lag Δt."""
        if delta_t < 0:
            return 0.0
        return self.alpha * np.exp(-self.beta * delta_t)

    def sample_waiting_time(self, rng: np.random.Generator) -> float:
        """
        Sample waiting time Δt from the exponential kernel.

        Uses the inverse-CDF method for the Hawkes kernel:
        F(Δt) = 1 - exp(-(α/β)·(1 - exp(-β·Δt)))

        Note: this method is available for standalone sampling but
        _sample_activation_delay() in HawkesDiffusion is preferred
        for cascade simulation as it handles horizon truncation.
        """
        # Total probability mass over infinite horizon:
        #   p_max = 1 - exp(-∫₀^∞ α·e^{-βs} ds) = 1 - exp(-α/β)
        p_max = -np.expm1(-self.alpha / self.beta)  # 1 - exp(-α/β)
        u = rng.random()
        if u >= p_max or u <= 1e-10:
            return float("inf")
        # Inverse CDF: Δt = -ln(1 + (β/α)·ln(1-u)) / β
        # Let F(d) = u = 1 - exp(-(α/β)(1 - e^{-βd}))
        #   => d = -ln(1 + (β/α)·ln(1-u)) / β
        cumul = -np.log1p(-u)  # -ln(1-u)
        inner = 1.0 - cumul * self.beta / self.alpha
        if inner <= 1e-10:
            return float("inf")  # numerical saturation → no activation
        return -np.log(inner) / self.beta

    def survival_prob(self, delta_t: float) -> float:
        """
        Probability that no activation has occurred by time Δt.

        P(T > Δt) = exp(-∫₀^Δt φ(s) ds)
                   = exp(-α/β · (1 - exp(-β·Δt)))
        """
        # Use expm1 for numerical stability when beta*delta_t is small
        integral = (self.alpha / self.beta) * (-np.expm1(-self.beta * delta_t))
        return np.exp(-integral)

    def cumulative_intensity(self, delta_t: float) -> float:
        """∫₀^Δt φ(s) ds = α/β · (1 - exp(-β·Δt))"""
        if delta_t <= 0:
            return 0.0
        return (self.alpha / self.beta) * (-np.expm1(-self.beta * delta_t))


class HawkesDiffusion:
    """
    Simulate influence spread under the Hawkes temporal diffusion model.

    The process:
    1. Seed set S is activated at time 0.
    2. When node u activates at time t_u, for each outgoing neighbor v:
       - v activates with probability proportional to φ(t - t_u)
       - The activation time t_v > t_u is sampled from the kernel
    3. Process continues until time horizon T.
    """

    def __init__(
        self,
        graph: TemporalGraph,
        kernel: ExponentialHawkesKernel,
        horizon: float,
        rng: Optional[np.random.Generator] = None,
    ):
        self.graph = graph
        self.kernel = kernel
        self.horizon = horizon
        self.rng = rng if rng is not None else np.random.default_rng()

    def forward_cascade(
        self, seeds: Set[int], num_simulations: int = 1000
    ) -> float:
        """
        Monte Carlo estimate of expected temporal influence f_t(S).

        Parameters
        ----------
        seeds : Set[int]
            Initial seed set.
        num_simulations : int
            Number of Monte Carlo runs.

        Returns
        -------
        float
            Estimated expected number of activated nodes.
        """
        total_activated = 0.0
        for _ in range(num_simulations):
            total_activated += self._simulate_single(seeds)
        return total_activated / num_simulations

    def _simulate_single(self, seeds: Set[int]) -> int:
        """
        Run a single cascade simulation.

        Uses a priority queue (min-heap by activation time).

        Correct temporal semantics:
        - A seed is activated at time 0.
        - For an activated node u at time t_u, each outgoing edge (u→v, t_e):
          The influence can only flow after both u is activated AND the
          interaction at t_e has occurred. Effective start = max(t_u, t_e).
          The activation delay is sampled from the Hawkes kernel starting
          at effective_start.
        """
        activated: Dict[int, float] = {}  # node -> activation time
        heap: List[Tuple[float, int, int]] = []  # (time, tiebreaker, node)

        # Initialize with seeds at time 0
        for s in seeds:
            activated[s] = 0.0

        # Seed nodes try to activate their neighbors
        tie = 0
        for s in seeds:
            for edge in self.graph.out_edges[s]:
                # Effective start = max(seed activation time, edge timestamp)
                effective_start = max(0.0, edge.t)
                delay = self._sample_activation_delay(effective_start)
                arrival = effective_start + delay
                if delay < float("inf") and arrival <= self.horizon:
                    heapq.heappush(heap, (arrival, tie, edge.v))
                    tie += 1

        while heap:
            t_arrival, _, v = heapq.heappop(heap)
            if v in activated:
                continue
            activated[v] = t_arrival

            # v tries to activate its neighbors
            for edge in self.graph.out_edges[v]:
                # Effective start = max(v's activation time, edge timestamp)
                effective_start = max(t_arrival, edge.t)
                delay = self._sample_activation_delay(effective_start)
                arrival = effective_start + delay
                if delay < float("inf") and arrival <= self.horizon:
                    heapq.heappush(heap, (arrival, tie, edge.v))
                    tie += 1

        return len(activated)

    def _sample_activation_delay(self, from_time: float) -> float:
        """
        Fast inverse-CDF sampling for exponential Hawkes kernel.

        Samples a delay Δt ≥ 0 such that the activation occurs at
        from_time + Δt, with probability governed by the Hawkes kernel
        integrated over [from_time, horizon].

        Returns float("inf") if no activation occurs within the horizon.

        Parameters
        ----------
        from_time : float
            Time from which delay is measured (effective start time).
        """
        remaining = self.horizon - from_time
        if remaining <= 0:
            return float("inf")

        a, b = self.kernel.alpha, self.kernel.beta
        # Cumulative intensity over [0, remaining]: ∫₀^r α·e^{-βs} ds
        cumul_max = (a / b) * (1.0 - np.exp(-b * remaining))
        p_max = -np.expm1(-cumul_max)  # 1 - exp(-cumul_max)

        u = self.rng.random()
        if u >= p_max:
            return float("inf")

        # Inverse CDF: F(d) = 1 - exp(-(α/β)(1 - e^{-βd}))
        # Given F(d) = u, solve for d:
        #   d = -ln(1 + (β/α)·ln(1-u)) / β
        cumul = -np.log1p(-u)  # -ln(1-u)
        inner = 1.0 - cumul * b / a
        if inner <= 1e-10:
            return remaining  # saturate at remaining
        delay = -np.log(inner) / b
        return delay if delay <= remaining else remaining


# TemporalInfluenceEstimator removed — TRRSetGenerator.estimate_influence()
# in trr_sets.py provides the same functionality with unbiased estimation.
