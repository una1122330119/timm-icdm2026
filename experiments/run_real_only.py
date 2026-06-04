#!/usr/bin/env python3
"""
Real-dataset experiments for TIMM — ICDM 2026 (FIXED VERSION).

Runs TIMM + baselines on all available real temporal networks:
  - CollegeMsg (n≈1.9K): small enough for MC-Hawkes-Greedy comparison
  - Reddit (n≈56K): medium, good for general comparison
  - DBLP (n≈317K): large citation network
  - Higgs (n≈457K): very large, retweet cascades
  - SuperUser (n≈194K): Q&A temporal interactions (optional, slow)

Changes from original:
  - MC_EVAL raised to 1000 for statistical reliability
  - N_RUNS=5 for mean ± std on evaluation
  - Proper kernel: α=0.35, β=1.0 (Moderate), tested to avoid saturation
  - MC-Hawkes-Greedy comparison on CollegeMsg (small enough to run)
  - All algorithms evaluated on the same temporal Hawkes model
"""

import sys, os, json, time, argparse
import numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tiimm import (
    ExponentialHawkesKernel, TIMM, StaticIMM, SnapshotIMM,
    MCHawkesGreedy, CELF, DegreeDiscount, TemporalDegree, RandomBaseline,
    load_dataset,
)

# ── Configuration ─────────────────────────────────────────────────────
SEED = 42
N_RUNS = 5          # independent evaluation runs for mean ± std
MC_EVAL = 1000       # MC samples for final evaluation
EPSILON = 0.3

# Kernel is auto-tuned per dataset below.
# Target effective branching b = 0.7 (subcritical) for all datasets
# to avoid cascade saturation and ensure seed selection matters.
def auto_tune_kernel(g, beta=1.0):
    deg = g.m / max(g.n, 1)
    if deg > 0.7:
        alpha = -beta * np.log(1.0 - 0.7 / deg)
    else:
        alpha = 0.5  # very sparse: moderate α, accept lower coverage
    return ExponentialHawkesKernel(alpha=round(alpha, 4), beta=beta)

K_VALUES = [10, 20, 50]

# Real datasets with per-dataset horizon
REAL_DATASETS = [
    ("collegemsg", 50.0, True),   # (name, horizon, run_mc_hawkes)
    ("reddit",     50.0, False),
    ("dblp",       50.0, False),
    ("higgs",      50.0, False),
]


def evaluate_multiple(timm, seeds, n_runs=N_RUNS):
    """Evaluate spread with multiple MC runs → mean ± std."""
    spreads = []
    for _ in range(n_runs):
        spreads.append(timm.evaluate(seeds, num_simulations=MC_EVAL))
    return np.mean(spreads), np.std(spreads)


def main():
    parser = argparse.ArgumentParser(description="TIMM Real-Dataset Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 eval run, MC_EVAL=200")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip Higgs and DBLP (slow on large graphs)")
    args = parser.parse_args()

    n_runs = 1 if args.quick else N_RUNS
    mc_eval = 200 if args.quick else MC_EVAL

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    out_path = os.path.join(REPO_ROOT, "results", "real_results.json")
    results = []

    def record(ds, algo, k, spread, std_spread, elapsed, **extra):
        r = {"dataset": ds, "algo": algo, "k": k,
             "spread": round(spread, 2),
             "std": round(std_spread, 2),
             "time_s": round(elapsed, 1)}
        r.update(extra)
        results.append(r)
        json.dump(results, open(out_path, "w"), indent=2)
        return r

    for ds_name, horizon, run_wc in REAL_DATASETS:
        if args.skip_large and ds_name in ("higgs", "dblp"):
            print(f"\nSkipping {ds_name} (--skip-large)")
            continue

        print(f"\n{'='*60}")
        print(f"Loading {ds_name}...")
        try:
            g = load_dataset(ds_name, data_dir=data_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  n={g.n}, m={g.m}, avg_deg={g.m/max(g.n,1):.1f}")
        KERNEL = auto_tune_kernel(g)
        b_actual = (g.m/max(g.n,1)) * (1.0 - np.exp(-KERNEL.alpha))
        print(f"  Kernel: α={KERNEL.alpha:.4f}, β={KERNEL.beta:.1f} → b≈{b_actual:.2f}")

        for k in K_VALUES:
            if k > g.n:
                continue
            print(f"\n  k={k}:", end=" ", flush=True)

            # ── TIMM ──
            t0 = time.time()
            timm = TIMM(g, KERNEL, horizon, seed=SEED)
            seeds, st = timm.run(k=k, epsilon=EPSILON, verbose=False)
            elapsed = time.time() - t0
            sp, std = evaluate_multiple(timm, seeds, n_runs)
            record(ds_name, "TIMM", k, sp, std, elapsed,
                   theta=st.get("theta", 0),
                   opt_lb=round(st.get("opt_lower_bound", 0), 1))
            print(f"TIMM={sp:.1f}±{std:.1f}({elapsed:.0f}s)", end=" ", flush=True)

            # ── StaticIMM ──
            t0 = time.time()
            si = StaticIMM(g, seed=SEED+1)
            s_seeds, _ = si.run(k=k)
            elapsed = time.time() - t0
            sp, std = evaluate_multiple(timm, s_seeds, n_runs)
            record(ds_name, "StaticIMM", k, sp, std, elapsed)
            print(f"Static={sp:.1f}±{std:.1f}", end=" ", flush=True)

            # ── SnapshotIMM ──
            t0 = time.time()
            snap = SnapshotIMM(g, seed=SEED+3)
            n_seeds, _ = snap.run(k=k)
            elapsed = time.time() - t0
            sp, std = evaluate_multiple(timm, n_seeds, n_runs)
            record(ds_name, "SnapshotIMM", k, sp, std, elapsed)
            print(f"Snap={sp:.1f}", end=" ", flush=True)

            # ── TemporalDegree ──
            t0 = time.time()
            td = TemporalDegree(g, beta=KERNEL.beta, horizon=horizon, seed=SEED)
            td_seeds, _ = td.run(k=k)
            elapsed = time.time() - t0
            sp, std = evaluate_multiple(timm, td_seeds, n_runs)
            record(ds_name, "TemporalDegree", k, sp, std, elapsed)
            print(f"TempDeg={sp:.1f}", end=" ", flush=True)

            # ── DegreeDiscount ──
            t0 = time.time()
            dd = DegreeDiscount(g, seed=SEED)
            dd_seeds, _ = dd.run(k=k)
            elapsed = time.time() - t0
            sp, std = evaluate_multiple(timm, dd_seeds, n_runs)
            record(ds_name, "DegreeDiscount", k, sp, std, elapsed)
            print(f"DegDis={sp:.1f}", end=" ", flush=True)

            # ── Random ──
            t0 = time.time()
            rd = RandomBaseline(g, seed=SEED)
            r_seeds, _ = rd.run(k=k)
            elapsed = time.time() - t0
            sp, std = evaluate_multiple(timm, r_seeds, n_runs)
            record(ds_name, "Random", k, sp, std, elapsed)
            print(f"Rand={sp:.1f}", end="", flush=True)

            # ── MC-Hawkes-Greedy (only on small graphs) ──
            if run_wc and g.n <= 2000:
                try:
                    mc_hawkes = MCHawkesGreedy(g, KERNEL, horizon, mc_samples=200, seed=SEED)
                    mc_hawkes_seeds, mc_hawkes_stats = mc_hawkes.run(k=k)
                    # Single MC eval for MC-Hawkes-Greedy (too expensive for multiple)
                    from tiimm.hawkes_model import HawkesDiffusion
                    diff = HawkesDiffusion(g, KERNEL, horizon,
                                           rng=np.random.default_rng(SEED + 1))
                    mc_hawkes_sp = diff.forward_cascade(set(mc_hawkes_seeds), mc_eval)
                    elapsed_mc_hawkes = mc_hawkes_stats["time_s"]
                    record(ds_name, "MCHawkesGreedy", k, mc_hawkes_sp, 0.0, elapsed_mc_hawkes)
                    print(f" MC-Hawkes={mc_hawkes_sp:.1f}({elapsed_mc_hawkes:.0f}s)", end="", flush=True)
                except Exception as e:
                    print(f" MC-Hawkes=fail({e})", end="", flush=True)

            # ── CELF (on small graphs) ──
            if g.n <= 2000:
                try:
                    celf = CELF(g, seed=SEED)
                    c_seeds, c_stats = celf.run(k=k, mc_samples=100)
                    elapsed_c = c_stats["time_s"]
                    sp_c, std_c = evaluate_multiple(timm, c_seeds, n_runs)
                    record(ds_name, "CELF", k, sp_c, std_c, elapsed_c)
                    print(f" CELF={sp_c:.1f}", end="", flush=True)
                except Exception as e:
                    print(f" CELF=fail", end="", flush=True)

            print()  # newline

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY (mean spread)")
    print(f"{'='*60}")
    datasets_done = sorted(set(r["dataset"] for r in results))
    for ds in datasets_done:
        print(f"\n  {ds}:")
        row = [r for r in results if r["dataset"] == ds]
        algos = sorted(set(r["algo"] for r in row))
        for algo in algos:
            vals = "  ".join(
                f"k={r['k']}:{r['spread']:>8.1f}"
                for r in sorted(row, key=lambda x: x["k"])
                if r["algo"] == algo
            )
            print(f"    {algo:<18} {vals}")

    print(f"\nResults saved to: {out_path}")
    print(f"Total entries: {len(results)}")
    print("DONE")


if __name__ == "__main__":
    main()


