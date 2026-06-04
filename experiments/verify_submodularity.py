#!/usr/bin/env python3
"""
Numerical verification of submodularity for §4.

Generates Figure X: empirical evidence of diminishing returns.
Shows that marginal gain Δ_S(v) = f(S ∪ {v}) - f(S) decreases as |S| grows.

Usage:
    python experiments/verify_submodularity.py
"""

import sys, os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tiimm import (
    generate_synthetic_graph,
    ExponentialHawkesKernel,
    HawkesDiffusion,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 150,
})


def verify_diminishing_returns(
    graph,
    kernel,
    horizon: float,
    mc_samples: int = 200,
    seed: int = 42,
):
    """
    Empirically verify: Δ_S(v) decreases as |S| grows.

    For each |S| ∈ {0, 1, 2, ..., max_k}, we:
    1. Randomly sample S of size |S|
    2. Randomly pick v ∉ S
    3. Estimate f(S), f(S∪{v}) via Monte Carlo
    4. Compute Δ_S(v) = f(S∪{v}) - f(S)
    5. Plot Δ vs |S|
    """
    n = graph.n
    max_k = min(20, n - 2)
    rng = np.random.default_rng(seed)

    diffuser = HawkesDiffusion(graph, kernel, horizon, rng=rng)

    all_nodes = np.arange(n)
    marginal_gains = []
    set_sizes = list(range(max_k + 1))

    for k_size in set_sizes:
        gains_for_k = []
        num_trials = 5  # multiple trials per size for robustness

        for _ in range(num_trials):
            # Random S of size k
            S = set(rng.choice(all_nodes, size=k_size, replace=False))

            # Random v not in S
            remaining = list(set(all_nodes) - S)
            if not remaining:
                continue
            v = int(rng.choice(remaining))

            # Estimate f(S) and f(S∪{v})
            f_S = diffuser.forward_cascade(S, mc_samples)
            f_Sv = diffuser.forward_cascade(S | {v}, mc_samples)
            delta = f_Sv - f_S
            gains_for_k.append(delta)

        if gains_for_k:
            marginal_gains.append((k_size, np.mean(gains_for_k), np.std(gains_for_k)))

    return marginal_gains


def plot_diminishing_returns(marginal_gains, save_path="fig_diminishing_returns.pdf"):
    """Plot marginal gain vs seed set size."""
    ks = [x[0] for x in marginal_gains]
    means = [x[1] for x in marginal_gains]
    stds = [x[2] for x in marginal_gains]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: marginal gain vs |S|
    ax = axes[0]
    ax.errorbar(ks, means, yerr=stds, fmt="o-", capsize=3,
                color="#2166AC", markersize=5, linewidth=1.5,
                label=r"$\Delta_S(v) = f(S \cup \{v\}) - f(S)$")

    # Fit a decreasing trend line
    z = np.polyfit(ks, means, 1)
    trend = np.poly1d(z)
    ax.plot(ks, trend(ks), "--", color="#B2182B", linewidth=1.2,
            alpha=0.7, label=f"Linear trend (slope={z[0]:.3f})")

    ax.set_xlabel("Seed set size |S|")
    ax.set_ylabel(r"Marginal gain $\Delta_S(v)$")
    ax.set_title("Empirical Verification of Diminishing Returns")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, max(ks) + 0.5)

    # Right: normalized marginal gain
    ax = axes[1]
    max_gain = means[0] if means[0] > 0 else 1
    norm_means = [m / max_gain for m in means]
    norm_stds = [s / max_gain for s in stds]

    ax.errorbar(ks, norm_means, yerr=norm_stds, fmt="s-", capsize=3,
                color="#4DAF4A", markersize=5, linewidth=1.5,
                label=r"Normalized $\Delta_S(v)$")

    # Theoretical (1 - |S|/n) curve for reference
    graph_n = len(ks)  # approximate
    ax.plot(ks, [1.0 - k/max(ks) for k in ks], "--", color="gray",
            linewidth=1, alpha=0.5, label=r"Reference: $1 - |S|/n$")

    ax.set_xlabel("Seed set size |S|")
    ax.set_ylabel(r"Normalized marginal gain")
    ax.set_title("Normalized Diminishing Returns")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close()


def verify_submodularity_inequality(
    graph, kernel, horizon, num_trials=100, mc_samples=100, seed=42
):
    """
    Directly test: f(S∪{v}) - f(S) ≥ f(T∪{v}) - f(T) for S ⊆ T.

    Counts the fraction of trials where submodularity holds.
    """
    n = graph.n
    rng = np.random.default_rng(seed)
    diffuser = HawkesDiffusion(graph, kernel, horizon, rng=rng)

    violations = 0
    deltas = []

    for trial in range(num_trials):
        # Random S, T with S ⊂ T
        all_nodes = np.arange(n)
        S_size = rng.integers(1, n // 4)
        T_size = rng.integers(S_size + 1, n // 2)

        S = set(rng.choice(all_nodes, size=S_size, replace=False))
        remaining = list(set(all_nodes) - S)
        extra = set(rng.choice(remaining, size=T_size - S_size, replace=False))
        T = S | extra

        # Random v not in T
        remaining = list(set(all_nodes) - T)
        if not remaining:
            continue
        v = int(rng.choice(remaining))

        f_S = diffuser.forward_cascade(S, mc_samples)
        f_Sv = diffuser.forward_cascade(S | {v}, mc_samples)
        f_T = diffuser.forward_cascade(T, mc_samples)
        f_Tv = diffuser.forward_cascade(T | {v}, mc_samples)

        delta_S = f_Sv - f_S
        delta_T = f_Tv - f_T

        if delta_S < delta_T - 1e-6:  # tolerance for MC noise
            violations += 1

        deltas.append((S_size, T_size, delta_S, delta_T))

    violation_rate = violations / num_trials
    return violation_rate, deltas


def main():
    print("=" * 60)
    print("Submodularity Verification for TIMM")
    print("=" * 60)

    # Build graph — increased to n=500 for statistical power (was n=80)
    graph = generate_synthetic_graph(n=500, m=5000, seed=42)
    kernel = ExponentialHawkesKernel(alpha=0.5, beta=0.1)
    horizon = 30.0

    print(f"Graph: {graph.n} nodes, {graph.m} edges")
    print(f"Kernel: α={kernel.alpha}, β={kernel.beta}")

    # 1. Diminishing returns (MC=1000 for reliable estimates)
    print("\n[1/2] Testing diminishing returns (MC=1000)...")
    marginal_gains = verify_diminishing_returns(
        graph, kernel, horizon, mc_samples=1000, seed=42
    )

    # 2. Direct inequality test (500 trials, MC=1000 each)
    print("\n[2/2] Testing f(S∪{v}) - f(S) ≥ f(T∪{v}) - f(T) (500 trials, MC=1000)...")
    violation_rate, deltas = verify_submodularity_inequality(
        graph, kernel, horizon, num_trials=500, mc_samples=1000, seed=42
    )
    plot_diminishing_returns(marginal_gains)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Violation rate (MC noise): {violation_rate:.2%}")
    print(f"(Expect ~0% if submodularity holds; small non-zero from MC variance)")

    if violation_rate < 0.05:
        print("\n✅ Empirical evidence strongly supports submodularity.")
    elif violation_rate < 0.10:
        print("\n⚠️ Marginal — increase MC samples for cleaner verification.")
    else:
        print("\n❌ WARNING: High violation rate — proof may have a gap.")

    # Show a few examples
    print(f"\nExample marginal gains (S_size, T_size, Δ_S, Δ_T):")
    for size_s, size_t, ds, dt in deltas[:5]:
        arrow = "≥" if ds >= dt - 1e-6 else "<?"
        print(f"  |S|={size_s:3d}  |T|={size_t:3d}  Δ_S={ds:7.2f}  Δ_T={dt:7.2f}  {arrow}")

    print(f"\nFigure: fig_diminishing_returns.pdf")


if __name__ == "__main__":
    main()


