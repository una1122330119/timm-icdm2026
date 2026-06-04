#!/usr/bin/env python3
"""
TRR Estimator Calibration — §4/§5 empirical validation.

For each real dataset:
  1. Generate θ=5000 TRR-sets.
  2. Randomly sample 100 seed sets across k ∈ {1, 5, 10, 20, 50}.
  3. For each seed set, compare:
       - TRR-based influence estimate (n/θ · Σ 1[R∩S≠∅])
       - High-MC forward simulation (10 000 cascades) as ground truth.
  4. Report: Spearman rank correlation, bias (mean, std), RMSE, scatter plot.

This experiment directly addresses the concern "how accurate is the
TRR estimator compared to ground-truth Monte Carlo?" and provides
evidence for Lemma 2's practical reliability.

Usage:
    python experiments/calibration_trr.py                     # all datasets
    python experiments/calibration_trr.py --dataset higgs      # single
    python experiments/calibration_trr.py --quick              # 50 seeds, MC=2000
"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from tiimm import (
    ExponentialHawkesKernel,
    TRRSetGenerator,
    TIMM,
    HawkesDiffusion,
    load_dataset,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "calibration_results.json")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42

DATASET_CONFIGS = {
    "higgs": {
        "beta": 1.0,
        "horizon": 100.0,
    },
    "reddit": {
        "beta": 1.0,
        "horizon": 100.0,
    },
    "dblp": {
        "beta": 1.0,
        "horizon": 100.0,
    },
    "collegemsg": {
        "beta": 1.0,
        "horizon": 100.0,
    },
}


def compute_alpha(graph, beta, target_b=0.7):
    """Compute α for target effective branching factor."""
    deg = graph.m / max(graph.n, 1)
    if deg > 1e-6:
        alpha = -beta * np.log(1.0 - target_b / deg)
    else:
        alpha = 0.5
    return max(0.01, min(0.99, alpha))


def run_calibration(
    graph,
    kernel,
    horizon: float,
    num_seedsets: int = 100,
    theta: int = 5000,
    mc_ground_truth: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run TRR estimator calibration on a single dataset.

    Returns dict with:
      - dataset info
      - per-seed-set (TRR_est, MC_est, k) records
      - aggregate stats: Spearman ρ, bias, RMSE, R²
    """
    n = graph.n
    rng = np.random.default_rng(seed)

    # ── Generate TRR-sets (shared across all seed sets) ──
    if verbose:
        print(f"  Generating θ={theta} TRR-sets...", end=" ", flush=True)
    t0 = time.time()
    trr_gen = TRRSetGenerator(graph, kernel, horizon, seed=seed)
    trr_sets = trr_gen.generate_batch_optimized(theta)
    t_trr = time.time() - t0
    if verbose:
        mean_sz = np.mean([len(R) for R in trr_sets])
        print(f"done ({t_trr:.1f}s, mean|R|={mean_sz:.1f})")

    # ── Fast TRR estimate via precomputed node→TRR index ──
    if verbose:
        print(f"  Building node→TRR index...", end=" ", flush=True)
    t0 = time.time()
    node_to_sets = {}
    for i, R in enumerate(trr_sets):
        for node in R:
            node_to_sets.setdefault(node, []).append(i)
    t_idx = time.time() - t0
    if verbose:
        print(f"done ({t_idx:.1f}s)")

    def trr_estimate(seed_set: set) -> float:
        covered = set()
        for node in seed_set:
            if node in node_to_sets:
                covered.update(node_to_sets[node])
        return (n / theta) * len(covered)

    # ── MC ground truth simulator ──
    diffuser = HawkesDiffusion(
        graph, kernel, horizon,
        rng=np.random.default_rng(seed + 99999),
    )

    def mc_estimate(seed_set: set) -> float:
        return diffuser.forward_cascade(seed_set, mc_ground_truth)

    # ── Generate random seed sets ──
    ks = [1, 5, 10, 20, 50]
    seeds_per_k = max(1, num_seedsets // len(ks))
    all_nodes = list(range(n))

    records = []
    if verbose:
        print(f"  Running {num_seedsets} calibration points...", flush=True)

    t_start = time.time()
    for k in ks:
        for rep in range(seeds_per_k):
            actual_k = min(k, n - 1)
            seed_set = set(rng.choice(all_nodes, size=actual_k, replace=False))

            trr_val = trr_estimate(seed_set)
            mc_val = mc_estimate(seed_set)

            records.append({
                "k": actual_k,
                "TRR_estimate": round(trr_val, 2),
                "MC_estimate": round(mc_val, 2),
                "abs_error": round(abs(trr_val - mc_val), 2),
                "rel_error": round(abs(trr_val - mc_val) / max(mc_val, 1e-6), 4),
            })

    t_total = time.time() - t_start + t_trr + t_idx

    # ── Aggregate statistics ──
    trr_vals = np.array([r["TRR_estimate"] for r in records])
    mc_vals = np.array([r["MC_estimate"] for r in records])

    # Spearman rank correlation
    rho, rho_p = sp_stats.spearmanr(trr_vals, mc_vals)

    # Pearson R²
    pearson_r, pearson_p = sp_stats.pearsonr(trr_vals, mc_vals)

    # Bias: mean(TRR - MC)
    biases = trr_vals - mc_vals
    mean_bias = float(np.mean(biases))
    std_bias = float(np.std(biases))

    # RMSE
    rmse = float(np.sqrt(np.mean(biases ** 2)))

    # Normalized RMSE (relative to mean MC spread)
    nrmse = rmse / max(float(np.mean(mc_vals)), 1e-6)

    # Per-k breakdown
    per_k = {}
    for k in sorted(set(r["k"] for r in records)):
        k_recs = [r for r in records if r["k"] == k]
        k_trr = np.array([r["TRR_estimate"] for r in k_recs])
        k_mc = np.array([r["MC_estimate"] for r in k_recs])
        k_biases = k_trr - k_mc
        per_k[str(k)] = {
            "n": len(k_recs),
            "mean_TRR": round(float(np.mean(k_trr)), 2),
            "mean_MC": round(float(np.mean(k_mc)), 2),
            "mean_bias": round(float(np.mean(k_biases)), 2),
            "rmse": round(float(np.sqrt(np.mean(k_biases ** 2))), 2),
        }

    result = {
        "dataset": graph.name if hasattr(graph, 'name') else "unknown",
        "n": n,
        "m": graph.m,
        "alpha": kernel.alpha,
        "beta": kernel.beta,
        "horizon": horizon,
        "theta": theta,
        "mc_ground_truth": mc_ground_truth,
        "num_seedsets": len(records),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rho_p),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_r2": round(float(pearson_r ** 2), 4),
        "mean_bias": round(mean_bias, 2),
        "std_bias": round(std_bias, 2),
        "rmse": round(rmse, 2),
        "nrmse": round(nrmse, 4),
        "time_total_s": round(t_total, 1),
        "per_k": per_k,
        "records": records,
    }

    if verbose:
        print(f"  Spearman rho = {rho:.4f}  (p = {rho_p:.2e})")
        print(f"  Pearson  r = {pearson_r:.4f}  (R^2 = {pearson_r**2:.4f})")
        print(f"  Bias = {mean_bias:+.1f} +/- {std_bias:.1f}")
        print(f"  RMSE = {rmse:.1f}  (NRMSE = {nrmse:.3f})")
        print(f"  Total time: {t_total:.0f}s")

    return result


def plot_calibration(results: list, save_path: str = None):
    """Generate calibration scatter plot for all datasets."""
    n_ds = len(results)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 5),
                              squeeze=False)
    fig.suptitle("TRR Estimator Calibration vs. Ground-Truth Monte Carlo",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax_idx, res in enumerate(results):
        ax = axes[0, ax_idx]
        trr = np.array([r["TRR_estimate"] for r in res["records"]])
        mc = np.array([r["MC_estimate"] for r in res["records"]])
        ks = np.array([r["k"] for r in res["records"]])

        # Color by k
        unique_ks = sorted(set(ks))
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(unique_ks)))
        for ki, k in enumerate(unique_ks):
            mask = ks == k
            ax.scatter(trr[mask], mc[mask], c=[colors[ki]], s=18,
                       alpha=0.7, edgecolors="white", linewidth=0.3,
                       label=f"k={k}")

        # Identity line
        lims = [min(trr.min(), mc.min()) * 0.9, max(trr.max(), mc.max()) * 1.1]
        ax.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.7)

        # Annotation
        rho = res["spearman_rho"]
        rmse = res["rmse"]
        ax.text(0.05, 0.95,
                f"ρ = {rho:.4f}\nRMSE = {rmse:.1f}\nn = {res['num_seedsets']}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#ccc"))

        ds_name = res["dataset"].upper() if res["dataset"] != "collegemsg" else "C.Msg"
        ax.set_title(f"{ds_name}  (n={res['n']//1000}k)", fontsize=11)
        ax.set_xlabel("TRR Estimate")
        ax.set_ylabel("MC Ground Truth (10K sims)")
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right",
                      markerscale=0.7, framealpha=0.8)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(FIG_DIR, "fig_calibration.pdf")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Figure saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 50 seeds, MC=2000")
    parser.add_argument("--theta", type=int, default=5000)
    parser.add_argument("--mc", type=int, default=10000)
    parser.add_argument("--num-seedsets", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.num_seedsets = 50
        args.mc = 2000
        print("[Quick mode] 50 seeds, MC=2000")

    datasets_to_run = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    all_results = []

    print("=" * 60)
    print("TRR Estimator Calibration")
    print(f"  θ={args.theta}, MC={args.mc}, seedsets={args.num_seedsets}")
    print("=" * 60)

    for ds_name in datasets_to_run:
        config = DATASET_CONFIGS[ds_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load
        print(f"  Loading...", end=" ", flush=True)
        t0 = time.time()
        graph = load_dataset(ds_name, data_dir=DATA_DIR)
        print(f"n={graph.n}, m={graph.m} ({time.time()-t0:.1f}s)")

        # Kernel
        alpha = compute_alpha(graph, config["beta"])
        kernel = ExponentialHawkesKernel(alpha=round(alpha, 4), beta=config["beta"])
        b_actual = (graph.m / max(graph.n, 1)) * (1.0 - np.exp(-alpha / config["beta"]))
        print(f"  α={alpha:.4f}, β={config['beta']}, b≈{b_actual:.2f}")

        # Run calibration
        result = run_calibration(
            graph, kernel, config["horizon"],
            num_seedsets=args.num_seedsets,
            theta=args.theta,
            mc_ground_truth=args.mc,
            seed=args.seed + hash(ds_name) % 10000,
        )
        result["dataset"] = ds_name
        all_results.append(result)

        # Save incrementally
        json.dump(all_results, open(OUT, "w"), indent=2)

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'ρ':>7} {'R²':>7} {'Bias':>8} {'±Std':>7} {'RMSE':>7} {'NRMSE':>7}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['dataset']:<12} {r['spearman_rho']:>7.4f} {r['pearson_r2']:>7.4f} "
              f"{r['mean_bias']:>+7.1f} {r['std_bias']:>7.1f} {r['rmse']:>7.1f} "
              f"{r['nrmse']:>7.4f}")
    print("-" * 60)

    # ── Plot ──
    plot_calibration(all_results)

    print(f"\nResults: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
