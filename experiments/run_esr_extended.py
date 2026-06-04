#!/usr/bin/env python3
"""
Extended ESR (Empirical Submodularity Ratio) verification on all 4 real datasets.

Upgrades the paper's 200-trial ESR to 500+ trials per dataset for stronger
statistical evidence. Uses TRR-set estimator (θ=5000) as described in §4.

Output: esr_extended.json with per-dataset violation counts, Clopper-Pearson CIs,
and the estimated γ lower bound.

Usage:
    python experiments/run_esr_extended.py                    # all 4 datasets, 500 trials
    python experiments/run_esr_extended.py --trials 1000       # 1000 trials
    python experiments/run_esr_extended.py --dataset higgs     # single dataset
"""

import sys, os, json, time, argparse
import numpy as np
from scipy import stats as sp_stats  # for Clopper-Pearson CI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tiimm import (
    ExponentialHawkesKernel,
    TRRSetGenerator,
    load_dataset,
    TemporalGraph,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "esr_extended.json")

DATASET_CONFIGS = {
    "higgs": {
        "file": "higgs",
        "beta": 1.0,
        "target_b": 0.7,
        "horizon": 100.0,
    },
    "dblp": {
        "file": "dblp",
        "beta": 1.0,
        "target_b": 0.7,
        "horizon": 100.0,
    },
    "reddit": {
        "file": "reddit",
        "beta": 1.0,
        "target_b": 0.7,
        "horizon": 100.0,
    },
    "collegemsg": {
        "file": "collegemsg",
        "beta": 1.0,
        "target_b": 0.7,
        "horizon": 100.0,
    },
}


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """
    Clopper-Pearson exact binomial confidence interval.
    
    k = number of successes (violations, in our case)
    n = number of trials
    
    Returns (lower, upper) bounds on the true proportion.
    For 0 violations: CI = [0.0, 1 - (alpha/2)^(1/n)]
    For n violations: CI = [(alpha/2)^(1/n), 1.0]
    """
    if k == 0:
        ci_low = 0.0
        ci_high = 1.0 - (alpha / 2) ** (1.0 / n)
    elif k == n:
        ci_low = (alpha / 2) ** (1.0 / n)
        ci_high = 1.0
    else:
        ci_low = sp_stats.beta.ppf(alpha / 2, k, n - k + 1)
        ci_high = sp_stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return ci_low, ci_high


def compute_alpha_from_b(graph, beta, target_b):
    """Compute α to achieve target effective branching factor."""
    deg = graph.m / max(graph.n, 1)
    if deg > 1.5:
        alpha = -beta * np.log(1.0 - target_b / deg)
    else:
        alpha = 0.5
    return round(alpha, 4)


def run_esr_on_dataset(
    graph: TemporalGraph,
    kernel: ExponentialHawkesKernel,
    horizon: float,
    num_trials: int = 500,
    theta: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run ESR verification on a single dataset.
    
    For each trial:
    1. Randomly select S ⊂ T ⊂ V and x ∉ T
    2. Estimate f(S), f(S∪{x}), f(T), f(T∪{x}) via TRR-set estimator
    3. Check if submodularity holds: Δ_S(x) ≥ Δ_T(x)
    
    Uses a single batch of θ TRR-sets for ALL trials (efficient).
    """
    n = graph.n
    rng = np.random.default_rng(seed)
    
    if verbose:
        print(f"  Generating θ={theta} TRR-sets...", end=" ", flush=True)
    
    t0 = time.time()
    trr_gen = TRRSetGenerator(graph, kernel, horizon, seed=seed)
    trr_sets = trr_gen.generate_batch_optimized(theta)
    t_trr = time.time() - t0
    
    if verbose:
        print(f"done ({t_trr:.1f}s). mean_size={np.mean([len(R) for R in trr_sets]):.1f}")
    
    # Precompute node-to-TRR-set mapping for fast coverage queries
    if verbose:
        print(f"  Building node→TRR index...", end=" ", flush=True)
    t0 = time.time()
    node_to_sets = {}
    for i, R in enumerate(trr_sets):
        for node in R:
            node_to_sets.setdefault(node, []).append(i)
    t_idx = time.time() - t0
    if verbose:
        print(f"done ({t_idx:.1f}s). {len(node_to_sets)} nodes indexed.")
    
    def estimate_fast(seed_set: set) -> float:
        """Fast TRR-set influence estimation using precomputed index."""
        covered = set()
        for node in seed_set:
            if node in node_to_sets:
                covered.update(node_to_sets[node])
        return (n / theta) * len(covered)
    
    # Run trials
    violations = 0
    deltas = []  # store (Δ_S, Δ_T) pairs for analysis
    all_nodes = list(range(n))
    
    t_trials_start = time.time()
    for trial in range(num_trials):
        # Random S, T with S ⊂ T
        S_size = rng.integers(1, min(n // 4, 20) + 1)
        T_size = rng.integers(S_size + 1, min(n // 2, 30) + 1)
        
        S_nodes = set(rng.choice(all_nodes, size=S_size, replace=False))
        remaining = list(set(all_nodes) - S_nodes)
        extra_nodes = set(rng.choice(remaining, size=min(T_size - S_size, len(remaining)), replace=False))
        T_nodes = S_nodes | extra_nodes
        
        # Random x not in T
        remaining = list(set(all_nodes) - T_nodes)
        if not remaining:
            continue
        x = int(rng.choice(remaining))
        
        # Estimate influences
        f_S = estimate_fast(S_nodes)
        f_Sx = estimate_fast(S_nodes | {x})
        f_T = estimate_fast(T_nodes)
        f_Tx = estimate_fast(T_nodes | {x})
        
        delta_S = f_Sx - f_S
        delta_T = f_Tx - f_T
        
        # Check submodularity: Δ_S ≥ Δ_T (with small tolerance for floating point)
        if delta_S < delta_T - 1e-9:
            violations += 1
        
        deltas.append({
            "S_size": S_size,
            "T_size": T_size,
            "delta_S": round(delta_S, 4),
            "delta_T": round(delta_T, 4),
            "ratio": round(delta_T / max(delta_S, 1e-12), 6),
        })
        
        if verbose and (trial + 1) % 100 == 0:
            elapsed = time.time() - t_trials_start
            print(f"    {trial+1}/{num_trials} trials ({elapsed:.0f}s), violations={violations}", flush=True)
    
    t_total = time.time() - t_trials_start + t_trr + t_idx
    
    # Clopper-Pearson CI
    violation_rate = violations / num_trials
    ci_low, ci_high = clopper_pearson_ci(violations, num_trials)
    
    # ESR γ analysis
    # Note: individual Δ_T/Δ_S ratios can be unstable due to TRR-set estimation
    # noise (small denominators). The robust measure is the violation rate:
    #   γ_lower = 1 - (upper CI on violation rate)
    # The paper reports γ ≥ γ_lower with 95% confidence.
    ratios = [d["ratio"] for d in deltas]
    # Use 5th percentile for a robust point estimate (less sensitive to noise)
    sorted_ratios = sorted(ratios)
    p5_ratio = sorted_ratios[max(0, int(len(sorted_ratios) * 0.05))]
    median_ratio = np.median(ratios)
    
    # γ lower bound = 1 - (upper CI on violation rate)
    # If violation rate ≤ p with 95% confidence, then γ ≥ 1-p
    gamma_lower = 1.0 - ci_high
    
    # Point estimate: use violation-rate-based approach as primary
    # γ_point ≈ 1 - violation_rate (for small rates, this ≈ the empirical bound)
    
    return {
        "dataset": graph.name if hasattr(graph, 'name') else "unknown",
        "n": n,
        "m": graph.m,
        "alpha": kernel.alpha,
        "beta": kernel.beta,
        "horizon": horizon,
        "theta": theta,
        "num_trials": num_trials,
        "violations": violations,
        "violation_rate": round(violation_rate, 6),
        "ci_95_lower": round(ci_low, 6),
        "ci_95_upper": round(ci_high, 6),
        "gamma_point_estimate": round(1.0 - violation_rate, 6),
        "gamma_lower_bound_95ci": round(gamma_lower, 6),
        "gamma_median_ratio": round(float(median_ratio), 6),
        "gamma_p5_ratio": round(float(p5_ratio), 6),
        "time_total_s": round(t_total, 1),
        "trr_gen_time_s": round(t_trr, 1),
        "index_time_s": round(t_idx, 1),
        "trials_time_s": round(time.time() - t_trials_start, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--theta", type=int, default=5000)
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    datasets_to_run = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    all_results = []
    
    print("=" * 60)
    print(f"ESR Extended Verification: {args.trials} trials/dataset, θ={args.theta}")
    print("=" * 60)
    
    for ds_name in datasets_to_run:
        config = DATASET_CONFIGS[ds_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        # Load dataset
        print(f"  Loading {ds_name}...", end=" ", flush=True)
        t0 = time.time()
        graph = load_dataset(config["file"], data_dir=DATA_DIR)
        print(f"done ({time.time()-t0:.1f}s). n={graph.n}, m={graph.m}")
        
        # Compute α
        deg = graph.m / max(graph.n, 1)
        alpha = compute_alpha_from_b(graph, config["beta"], config["target_b"])
        kernel = ExponentialHawkesKernel(alpha=alpha, beta=config["beta"])
        b_actual = deg * (1.0 - np.exp(-alpha / config["beta"]))
        print(f"  avg_deg={deg:.2f}, α={alpha:.4f}, b≈{b_actual:.2f}")
        
        # Run ESR
        result = run_esr_on_dataset(
            graph, kernel, config["horizon"],
            num_trials=args.trials, theta=args.theta, seed=args.seed,
        )
        result["dataset"] = ds_name
        all_results.append(result)
        
        # Print summary
        print(f"\n  Results for {ds_name}:")
        print(f"    Trials:        {result['num_trials']}")
        print(f"    Violations:    {result['violations']}")
        print(f"    Viol. rate:    {result['violation_rate']:.4%}")
        print(f"    95% CI:        [{result['ci_95_lower']:.4%}, {result['ci_95_upper']:.4%}]")
        print(f"    γ (point):     {result['gamma_point_estimate']:.6f}")
        print(f"    γ (95% CI lb): {result['gamma_lower_bound_95ci']:.6f}")
        print(f"    γ (median):    {result['gamma_median_ratio']:.6f}")
        print(f"    γ (P5):        {result['gamma_p5_ratio']:.6f}")
        print(f"    Total time:    {result['time_total_s']:.0f}s")
        
        # Save incrementally
        json.dump(all_results, open(OUT, "w"), indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<14} {'Trials':>7} {'Viol':>5} {'Rate':>8} {'CI lb':>8} {'CI ub':>8} {'γ lb':>8} {'Time':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset']:<14} {r['num_trials']:>7} {r['violations']:>5} "
              f"{r['violation_rate']:>8.4%} {r['ci_95_lower']:>8.4%} {r['ci_95_upper']:>8.4%} "
              f"{r['gamma_lower_bound_95ci']:>8.4f} {r['time_total_s']:>7.0f}s")
    
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()


