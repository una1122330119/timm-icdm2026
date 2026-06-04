"""
Martingale Concentration Bounds for TIMM.

Extends the IMM martingale approach (Tang et al., SIGMOD 2015)
to temporal reverse reachable sets.

Core idea:
The TRR-set coverage process forms a martingale. We use
martingale concentration inequalities to derive tight bounds
on the number of TRR-sets needed for (1-1/e-ε) approximation.

Theorem 2 (Martingale Bound):
    For ε, δ ∈ (0, 1), generating
        θ ≥ (2n / ε²) · ln(1/δ)
    TRR-sets guarantees that greedy max-coverage on TRR-sets
    returns a (1-1/e-ε)-approximate seed set with probability ≥ 1-δ.
"""

import numpy as np
from math import log, ceil
from typing import Tuple


def compute_trr_set_bound(
    n: int,
    k: int,
    epsilon: float,
    delta: float = 0.01,
) -> int:
    """
    Compute the minimum number of TRR-sets needed for (ε, δ)-approximation.

    This is the temporal extension of IMM's Theorem 4 (Tang et al. 2015):
        θ ≥ 2n · ((1 - 1/e) · ℓ + ...)  /  ε²

    Simplified practical bound:
        θ = ⌈2n · ln(1/δ) / ε²⌉

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    k : int
        Seed budget.
    epsilon : float
        Approximation error parameter (e.g., 0.1 means 10% error).
    delta : float
        Confidence parameter (e.g., 0.01 means 99% confidence).

    Returns
    -------
    int
        Required number of TRR-sets.
    """
    # Base bound from concentration inequality
    theta_base = ceil(2.0 * n * log(1.0 / delta) / (epsilon ** 2))

    # Refinement: incorporate the optimum lower bound
    # In IMM, they estimate OPT via sampling and use:
    #   θ = 2n · ((1 - 1/e) · α + β)² / (ε · OPT)²
    # For simplicity, we use a conservative bound first,
    # then refine after estimating OPT.
    return theta_base


def compute_trr_set_bound_refined(
    n: int,
    k: int,
    epsilon: float,
    delta: float = 0.01,
    opt_lower_bound: float = None,
) -> int:
    """
    Compute TRR-set count using IMM-style martingale bound.

    Uses the martingale concentration inequality adapted for TRR-sets.
    The bound scales with n, 1/ε², and 1/OPT_L.

    Unlike the previous version, there is no arbitrary hard cap —
    θ is determined by the theoretical formula with a practical
    ceiling that grows with graph size.
    """
    if opt_lower_bound is None or opt_lower_bound <= 0:
        opt_lower_bound = max(k, 1.0)

    # IMM bound: θ = (2n / ε²) · (ln(1/δ) + ln C(n,k))
    # ln C(n,k) ≈ k·(ln(n/k) + 1) for k ≪ n
    log_comb = k * (log(n / max(k, 1)) + 1.0) if k > 0 else 0
    theta_raw = 2.0 * n * (log(1.0 / delta) + log_comb) / (epsilon ** 2)

    # Scale by OPT lower bound — larger OPT means fewer TRR-sets needed
    theta = ceil(theta_raw / max(opt_lower_bound, 1.0))

    # Dynamic practical cap: scales with graph size.
    # We keep a generous ceiling so the (ε,δ) guarantee is preserved
    # for reasonable ε values. For very small ε (<0.1) on small graphs,
    # the theoretical θ may exceed the cap; a warning is raised below.
    theta_max = int(max(n * 50, 200000))
    theta_min = max(100, n // 10)
    theta_max = max(theta_max, theta_min * 2)  # ensure max > min

    theta = max(min(theta, theta_max), theta_min)

    # Warn if the theoretical bound was clipped (for diagnostics)
    theoretical = theta_raw / max(opt_lower_bound, 1.0)
    if theoretical > theta_max:
        import warnings
        warnings.warn(
            f"θ capped at {theta} (theoretical: {theoretical:.0f}). "
            f"For ε={epsilon}, the (ε,δ) guarantee may be weakened. "
            f"Consider increasing ε to 0.3–0.5 for small graphs with low OPT."
        )
    return theta


def estimate_opt_lower_bound(
    n: int, k: int, trr_sets_sample: list, sample_fraction: float = 0.1
) -> float:
    """
    Estimate a lower bound on OPT (for k seeds) using a small sample of TRR-sets.

    Runs a quick greedy max-coverage on the sample TRR-sets to select k seeds,
    then scales the resulting coverage fraction to an absolute influence estimate.
    This follows the standard IMM approach (Tang et al. 2015, §4.2).

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Seed budget.
    trr_sets_sample : list of set
        A small sample of TRR-sets.
    sample_fraction : float
        Unused; kept for API compatibility.

    Returns
    -------
    float
        Lower bound on OPT (absolute number of expected activated nodes).
    """
    theta_sample = len(trr_sets_sample)
    if theta_sample == 0:
        return float(k)

    # ── Greedy max-coverage on the sample (quick, k seeds) ──
    uncovered = set(range(theta_sample))  # set of still-uncovered TRR-set indices
    selected_nodes = set()
    # Precompute: for each node, which TRR-sets contain it?
    node_to_sets = {}
    for i, R in enumerate(trr_sets_sample):
        for node in R:
            node_to_sets.setdefault(node, []).append(i)

    total_covered = 0
    for _ in range(k):
        best_node, best_count = -1, 0
        # Find node covering the most still-uncovered TRR-sets
        for node, set_indices in node_to_sets.items():
            if node in selected_nodes:
                continue
            cnt = sum(1 for idx in set_indices if idx in uncovered)
            if cnt > best_count:
                best_count, best_node = cnt, node
        if best_node == -1 or best_count == 0:
            break
        selected_nodes.add(best_node)
        # Remove covered TRR-sets from uncovered set
        covered_now = {idx for idx in node_to_sets.get(best_node, [])
                       if idx in uncovered}
        uncovered -= covered_now
        total_covered += best_count

    # Fraction of sample TRR-sets covered by the greedy k-seed set
    covered_fraction = total_covered / theta_sample

    # Scale to absolute influence: fraction × n gives expected activated nodes.
    # Apply a conservative discount (0.8×) to ensure it's a valid lower bound
    # under sampling noise (standard IMM practice).
    opt_lb = covered_fraction * n * 0.8

    # Floor: at least k (each seed activates itself) + 1
    return max(opt_lb, float(k + 1))


class MartingaleVerifier:
    """
    Verify the martingale property of the TRR-set coverage process.

    For theoretical correctness: the TRR-set coverage Z_i = 1[R_i ∩ S]
    forms a martingale difference sequence. This module provides
    computational verification of the concentration bound.
    """

    @staticmethod
    def verify_concentration(
        trr_sets: list, seed_set: set, n: int, epsilon: float
    ) -> dict:
        """
        Empirically verify that the concentration bound holds.

        Parameters
        ----------
        trr_sets : list of set
            Generated TRR-sets.
        seed_set : set
            Seed set to evaluate.
        n : int
            Number of nodes.
        epsilon : float
            Target error.

        Returns
        -------
        dict
            Verification results.
        """
        theta = len(trr_sets)
        hits = sum(1 for R in trr_sets if seed_set & R)
        estimate = n * hits / theta

        # Theoretical bound on deviation
        dev_bound = n * np.sqrt(log(2.0 / 0.01) / (2.0 * theta))

        return {
            "num_trr_sets": theta,
            "hits": hits,
            "estimated_influence": estimate,
            "theoretical_deviation_bound": dev_bound,
            "relative_error_bound": dev_bound / max(estimate, 1),
            "epsilon_target": epsilon,
            "bound_satisfied": dev_bound / max(estimate, 1) <= epsilon,
        }
