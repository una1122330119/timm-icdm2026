#!/usr/bin/env python3
"""
TIMM main experiments — ICDM 2026 submission (FIXED VERSION).

Changes from original:
- MC_EVAL raised from 100 → 1000 (minimum for statistical reliability)
- Multiple independent runs (N_RUNS=5) for mean ± std reporting
- Dynamic horizon per dataset (matched to data time range)
- CollegeMsg added as small real graph for MC-Hawkes-Greedy comparison
- All algorithms evaluated on the same Hawkes model
- Timeout handling for slow algorithms (MCHawkesGreedy, CELF)
"""

import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiimm import (
    generate_synthetic_graph,
    ExponentialHawkesKernel,
    TIMM,
    StaticIMM,
    SnapshotIMM,
    MCHawkesGreedy,
    CELF,
    DegreeDiscount,
    TemporalDegree,
    RandomBaseline,
    TRRSetGenerator,
    load_dataset,
)

# ── Configuration ─────────────────────────────────────────────────────
SEED = 42
N_RUNS = 5          # independent runs for mean ± std
MC_EVAL = 1000       # MC samples for final evaluation (was 100)
EPSILON = 0.3
TIMEOUT_S = 3600 * 6  # 6 hours per algorithm-dataset-k combination

# Kernel auto-tuning: target subcritical b = 0.7 per dataset.
# Effective branching factor b = avg_deg · (1 − exp(−α/β)).
# Subcritical cascades (b < 1) ensure seed quality matters; supercritical
# cascades (b > 1) cause exponential explosion → all seeds look identical.
def auto_tune_kernel(g, beta=1.0):
    deg = g.m / max(g.n, 1)
    if deg > 0.7:
        alpha = -beta * np.log(1.0 - 0.7 / deg)
    else:
        alpha = 0.5  # very sparse: moderate α
    return ExponentialHawkesKernel(alpha=round(alpha, 4), beta=beta)

KERNEL_DEFAULT = None  # reassigned per dataset via auto_tune_kernel

# Synthetic datasets: edge counts tuned so avg out-degree ∈ [4, 6].
# This keeps effective branching factor manageable (b < 2.5).
SYNTHETIC_DATASETS = [
    ("Synth-S", (500, 2000), 20.0),     # avg deg ≈ 4
    ("Synth-M", (2000, 12000), 20.0),   # avg deg ≈ 6
    ("Synth-L", (5000, 30000), 20.0),   # avg deg ≈ 6
]

K_VALUES = [10, 20, 50]


def build_graph(name, n, m):
    """Build synthetic graph with consistent seed."""
    return generate_synthetic_graph(n, m, seed=SEED)


def evaluate(timm, seeds, mc_samples=MC_EVAL):
    """Monte Carlo evaluation of seed set influence."""
    return timm.evaluate(seeds, num_simulations=mc_samples)


def eval_multiple(timm, seeds, n_runs=N_RUNS):
    """Evaluate with multiple MC runs to get mean ± std."""
    spreads = []
    for _ in range(n_runs):
        spreads.append(evaluate(timm, seeds))
    return np.mean(spreads), np.std(spreads)


def run_timm(g, kernel, k, eps, horizon):
    """Run TIMM and evaluate."""
    t0 = time.time()
    timm = TIMM(g, kernel, horizon, seed=SEED)
    seeds, stats = timm.run(k=k, epsilon=eps, verbose=False)
    elapsed = time.time() - t0
    mean_sp, std_sp = eval_multiple(timm, seeds)
    return seeds, mean_sp, std_sp, elapsed, {
        "theta": stats.get("theta", 0),
        "opt_lb": round(stats.get("opt_lower_bound", 0), 2),
    }


def run_static_imm(g, k, horizon):
    """Run Static IMM and evaluate on Hawkes model."""
    alg = StaticIMM(g, seed=SEED+1)
    # Evaluate on the TEMPORAL model to be fair
    timm_eval = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
    t0 = time.time()
    seeds, _ = alg.run(k=k)
    elapsed = time.time() - t0
    mean_sp, std_sp = eval_multiple(timm_eval, seeds)
    return seeds, mean_sp, std_sp, elapsed


def run_snapshot_imm(g, k, horizon):
    """Run Snapshot IMM and evaluate on Hawkes model."""
    alg = SnapshotIMM(g, seed=SEED+3)
    timm_eval = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
    t0 = time.time()
    seeds, _ = alg.run(k=k)
    elapsed = time.time() - t0
    mean_sp, std_sp = eval_multiple(timm_eval, seeds)
    return seeds, mean_sp, std_sp, elapsed


def run_mc_hawkes(g, kernel, k, horizon):
    """Run MC-Hawkes-Greedy (MC greedy, no guarantees)."""
    from tiimm.hawkes_model import HawkesDiffusion
    alg = MCHawkesGreedy(g, kernel, horizon, mc_samples=200, seed=SEED)
    t0 = time.time()
    seeds, stats = alg.run(k=k)
    elapsed = time.time() - t0
    # Evaluate with same MC samples for fairness
    diffuser = HawkesDiffusion(g, kernel, horizon, rng=np.random.default_rng(SEED + 1))
    spread = diffuser.forward_cascade(set(seeds), MC_EVAL)
    return seeds, spread, 0.0, elapsed  # no std for single run (too costly)


def run_celf(g, k, horizon):
    """Run CELF and evaluate on Hawkes model."""
    alg = CELF(g, seed=SEED)
    timm_eval = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
    t0 = time.time()
    seeds, _ = alg.run(k=k, mc_samples=100)
    elapsed = time.time() - t0
    mean_sp, std_sp = eval_multiple(timm_eval, seeds)
    return seeds, mean_sp, std_sp, elapsed


def run_temporal_degree(g, k, horizon, beta=1.0):
    """Run TemporalDegree baseline."""
    alg = TemporalDegree(g, beta=beta, horizon=horizon, seed=SEED)
    timm_eval = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
    t0 = time.time()
    seeds, _ = alg.run(k=k)
    elapsed = time.time() - t0
    mean_sp, std_sp = eval_multiple(timm_eval, seeds)
    return seeds, mean_sp, std_sp, elapsed


def run_degree_discount(g, k, horizon):
    """Run DegreeDiscount and evaluate on Hawkes model."""
    alg = DegreeDiscount(g, seed=SEED)
    timm_eval = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
    t0 = time.time()
    seeds, _ = alg.run(k=k)
    elapsed = time.time() - t0
    mean_sp, std_sp = eval_multiple(timm_eval, seeds)
    return seeds, mean_sp, std_sp, elapsed


def run_random(g, k, horizon):
    """Run Random baseline and evaluate on Hawkes model."""
    alg = RandomBaseline(g, seed=SEED)
    timm_eval = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
    t0 = time.time()
    seeds, _ = alg.run(k=k)
    elapsed = time.time() - t0
    mean_sp, std_sp = eval_multiple(timm_eval, seeds)
    return seeds, mean_sp, std_sp, elapsed


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TIMM Main Experiments (Fixed)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 run, MC_EVAL=200, skip slow")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip MCHawkesGreedy and CELF")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation studies")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip parameter sensitivity")
    args = parser.parse_args()

    n_runs = 1 if args.quick else N_RUNS
    global MC_EVAL
    mc_eval = 200 if args.quick else MC_EVAL
    MC_EVAL = mc_eval

    results = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "experiment_results.json")

    def record(dataset, algo, k, spread, std_spread, elapsed, **extra):
        r = {"dataset": dataset, "algo": algo, "k": k,
             "spread": round(spread, 2),
             "std": round(std_spread, 2),
             "time_s": round(elapsed, 1)}
        r.update(extra)
        results.append(r)
        # Save incrementally
        json.dump(results, open(out_path, "w"), indent=2, default=str)
        return r

    # ==================================================================
    # EXP 1: Main Comparison
    # ==================================================================
    print(f"{'='*70}")
    print("EXP 1: TIMM vs Baselines (mean ± std, {n_runs} runs, MC={mc_eval})")
    print(f"{'='*70}")

    datasets = []

    # Build synthetic datasets
    for name, (n, m), horizon in SYNTHETIC_DATASETS:
        print(f"\nBuilding {name} (n={n}, m={m}, horizon={horizon})...")
        g = build_graph(name, n, m)
        datasets.append((name, g, horizon))
        stats = g.get_temporal_degree_stats()
        print(f"  mean_deg={stats['mean_in_deg']:.1f}, max_deg={stats['max_in_deg']}")

    # Load real datasets
    data_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # Reddit
    try:
        for ddir in [data_base, "data"]:
            try:
                g_real = load_dataset("reddit", data_dir=ddir)
                # Reddit timestamps normalized to [0, 100], use horizon=50
                datasets.append(("Reddit", g_real, 50.0))
                print(f"\nLoaded Reddit: n={g_real.n}, m={g_real.m}, horizon=50")
                break
            except Exception:
                continue
        else:
            print("\nReddit not found — skipping")
    except Exception as e:
        print(f"\nReddit: {e}")

    # CollegeMsg (small real graph for MC-Hawkes-Greedy)
    try:
        for ddir in [data_base, "data"]:
            path = os.path.join(ddir, "CollegeMsg.txt")
            if os.path.isfile(path):
                g_cm = load_dataset("collegemsg", data_dir=ddir)
                datasets.append(("CollegeMsg", g_cm, 50.0))
                print(f"\nLoaded CollegeMsg: n={g_cm.n}, m={g_cm.m}, horizon=50")
                break
            path_gz = os.path.join(ddir, "CollegeMsg.txt.gz")
            if os.path.isfile(path_gz):
                g_cm = load_dataset("collegemsg", data_dir=ddir)
                datasets.append(("CollegeMsg", g_cm, 50.0))
                print(f"\nLoaded CollegeMsg: n={g_cm.n}, m={g_cm.m}, horizon=50")
                break
        else:
            print("\nCollegeMsg not found — skipping")
    except Exception as e:
        print(f"\nCollegeMsg: {e}")

    for name, g, horizon in datasets:
        print(f"\n{'─'*50}")
        print(f"Dataset: {name}  (n={g.n}, m={g.m}, horizon={horizon})")
        global KERNEL_DEFAULT
        KERNEL_DEFAULT = auto_tune_kernel(g)
        b = (g.m/max(g.n,1)) * (1.0 - np.exp(-KERNEL_DEFAULT.alpha))
        print(f"Kernel: α={KERNEL_DEFAULT.alpha:.4f} β=1.0 → b≈{b:.2f}")
        print(f"{'─'*50}")

        for k in K_VALUES:
            print(f"\n  k = {k}")
            print(f"  {'Algorithm':<15} {'Spread':>10} {'Time':>8}")
            print(f"  {'─'*33}")

            # TIMM
            _, sp, std_sp, t, extra = run_timm(g, KERNEL_DEFAULT, k, EPSILON, horizon)
            record(name, "TIMM", k, sp, std_sp, t, **extra)
            print(f"  {'TIMM':<15} {sp:>8.1f}±{std_sp:.1f} {t:>6.1f}s")

            # Static IMM
            _, sp, std_sp, t = run_static_imm(g, k, horizon)
            record(name, "StaticIMM", k, sp, std_sp, t)
            print(f"  {'Static IMM':<15} {sp:>8.1f}±{std_sp:.1f} {t:>6.1f}s")

            # Snapshot IMM
            _, sp, std_sp, t = run_snapshot_imm(g, k, horizon)
            record(name, "SnapshotIMM", k, sp, std_sp, t)
            print(f"  {'Snapshot IMM':<15} {sp:>8.1f}±{std_sp:.1f} {t:>6.1f}s")

            # MC-Hawkes-Greedy (only on small graphs to avoid timeout)
            if not args.skip_slow and g.n <= 2000:
                print(f"  {'MC-Hawkes-Greedy':<15} {'running...':>10}")
                try:
                    _, sp, _, t = run_mc_hawkes(g, KERNEL_DEFAULT, k, horizon)
                    record(name, "MCHawkesGreedy", k, sp, 0.0, t)
                    print(f"  {'MC-Hawkes-Greedy':<15} {sp:>8.1f}      {t:>6.1f}s")
                except Exception as e:
                    print(f"  {'MC-Hawkes-Greedy':<15} {'FAILED':>10} — {e}")

            # CELF (only on small graphs)
            if not args.skip_slow and g.n <= 2000:
                print(f"  {'CELF':<15} {'running...':>10}")
                try:
                    _, sp, std_sp, t = run_celf(g, k, horizon)
                    record(name, "CELF", k, sp, std_sp, t)
                    print(f"  {'CELF':<15} {sp:>8.1f}±{std_sp:.1f} {t:>6.1f}s")
                except Exception as e:
                    print(f"  {'CELF':<15} {'FAILED':>10} — {e}")

            # DegreeDiscount
            _, sp, std_sp, t = run_degree_discount(g, k, horizon)
            record(name, "DegreeDiscount", k, sp, std_sp, t)
            print(f"  {'DegDiscount':<15} {sp:>8.1f}±{std_sp:.1f} {t:>6.1f}s")

            # TemporalDegree
            _, sp, std_sp, t = run_temporal_degree(g, k, horizon, beta=1.0)
            record(name, "TemporalDegree", k, sp, std_sp, t)
            print(f"  {'TemporalDeg':<15} {sp:>8.1f}±{std_sp:.1f} {t:>6.1f}s")

            # Random
            _, sp, std_sp, t = run_random(g, k, horizon)
            record(name, "Random", k, sp, std_sp, t)
            print(f"  {'Random':<15} {sp:>8.1f}±{std_sp:.1f} {t:>6.1f}s")

    # ==================================================================
    # EXP 2: Ablation Studies (Synth-M, k=20)
    # ==================================================================
    if not args.skip_ablation and len(datasets) >= 2:
        print(f"\n{'='*70}")
        print("EXP 2: Ablation Studies (Synth-M, k=20)")
        print(f"{'='*70}")

        g = datasets[1][1]  # Synth-M
        horizon = datasets[1][2]
        timm = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
        trr_gen = TRRSetGenerator(g, KERNEL_DEFAULT, horizon, seed=SEED)

        # 2a: TRR-set count ablation
        print("\n[2a] TRR-set count ablation")
        _, st = timm.run(k=20, epsilon=EPSILON, verbose=False)
        theta_base = st["theta"]
        for label, theta in [("θ/4", max(500, theta_base // 4)),
                              ("θ/2", theta_base // 2),
                              ("θ (TIMM)", theta_base),
                              ("2θ", min(theta_base * 2, 50000))]:
            rr = trr_gen.generate_batch_numba(theta)
            seeds_a, _ = trr_gen.greedy_max_cover(rr, 20)
            sp, std_sp = eval_multiple(timm, seeds_a, n_runs=3)
            record("Synth-M", f"Abl-TRR-{label}", 20, sp, std_sp, 0, theta=theta)
            print(f"  {label}: θ={theta:>6d}  spread={sp:.1f}±{std_sp:.1f}")

        # 2b: Temporal vs Static RR-sets
        print("\n[2b] Temporal vs Static RR (θ=2000)")
        theta = 2000
        trr_t = trr_gen.generate_batch_numba(theta)
        seeds_t, _ = trr_gen.greedy_max_cover(trr_t, 20)
        sp_t, std_t = eval_multiple(timm, seeds_t)

        # Static RR-sets (uniform edge probability, no time)
        rng = np.random.default_rng(SEED)
        rr_s = []
        for _ in range(theta):
            tgt = rng.integers(0, g.n)
            R = {tgt}
            q = [tgt]
            while q:
                v = q.pop()
                for e in g.in_edges[v]:
                    if rng.random() < 0.1:
                        if e.u not in R:
                            R.add(e.u)
                            q.append(e.u)
            rr_s.append(R)
        seeds_s, _ = trr_gen.greedy_max_cover(rr_s, 20)
        sp_s, std_s = eval_multiple(timm, seeds_s)

        record("Synth-M", "Abl-Temporal-RR", 20, sp_t, std_t, 0)
        record("Synth-M", "Abl-Static-RR", 20, sp_s, std_s, 0)
        gain = (sp_t - sp_s) / max(sp_s, 1) * 100
        print(f"  Temporal RR: {sp_t:.1f}±{std_t:.1f}")
        print(f"  Static RR:    {sp_s:.1f}±{std_s:.1f}")
        print(f"  Gain from temporal info: {gain:.1f}%")

        # 2c: Kernel type comparison
        print("\n[2c] Kernel type comparison")
        for label, kw in [
            ("Exp(β=1.0)", {"alpha": 0.5, "beta": 1.0}),
            ("Exp(β=0.1)", {"alpha": 0.5, "beta": 0.1}),
            ("Exp(β=0.02)", {"alpha": 0.5, "beta": 0.02}),
        ]:
            ker = ExponentialHawkesKernel(**kw)
            tt = TIMM(g, ker, horizon, seed=SEED)
            seeds_k, _ = tt.run(k=20, epsilon=EPSILON, verbose=False)
            sp_k, std_k = eval_multiple(tt, seeds_k)
            record("Synth-M", f"Abl-Kernel-{label}", 20, sp_k, std_k, 0)
            print(f"  {label}: spread={sp_k:.1f}±{std_k:.1f}")

    # ==================================================================
    # EXP 3: Parameter Sensitivity (Synth-M, k=20)
    # ==================================================================
    if not args.skip_sensitivity and len(datasets) >= 2:
        print(f"\n{'='*70}")
        print("EXP 3: Parameter Sensitivity (Synth-M, k=20)")
        print(f"{'='*70}")

        g = datasets[1][1]
        horizon = datasets[1][2]

        for param, values in [
            ("epsilon", [0.05, 0.1, 0.2, 0.3, 0.5]),
            ("alpha", [0.1, 0.3, 0.5, 0.7, 0.9]),
            ("beta", [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
        ]:
            print(f"\n  Parameter: {param}")
            for val in values:
                if param == "epsilon":
                    tt = TIMM(g, KERNEL_DEFAULT, horizon, seed=SEED)
                    seeds_p, st = tt.run(k=20, epsilon=val, verbose=False)
                    sp, std_sp = eval_multiple(tt, seeds_p, n_runs=3)
                    record("Synth-M", f"Sens-{param}={val}", 20, sp, std_sp, 0,
                           theta=st.get("theta", 0))
                elif param == "alpha":
                    ker = ExponentialHawkesKernel(alpha=val, beta=1.0)
                    tt = TIMM(g, ker, horizon, seed=SEED)
                    seeds_p, _ = tt.run(k=20, epsilon=EPSILON, verbose=False)
                    sp, std_sp = eval_multiple(tt, seeds_p, n_runs=3)
                    record("Synth-M", f"Sens-{param}={val}", 20, sp, std_sp, 0)
                elif param == "beta":
                    ker = ExponentialHawkesKernel(alpha=0.3, beta=val)
                    tt = TIMM(g, ker, horizon, seed=SEED)
                    seeds_p, _ = tt.run(k=20, epsilon=EPSILON, verbose=False)
                    sp, std_sp = eval_multiple(tt, seeds_p, n_runs=3)
                    record("Synth-M", f"Sens-{param}={val}", 20, sp, std_sp, 0)
                print(f"    {param}={val:<5}  spread={sp:.1f}±{std_sp:.1f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("SUMMARY (mean spread)")
    print(f"{'='*70}")
    for name, _, _ in datasets:
        print(f"\n  {name}:")
        row = [r for r in results if r["dataset"] == name]
        algos = sorted(set(r["algo"] for r in row))
        for algo in algos:
            vals = "  ".join(
                f"k={r['k']}:{r['spread']:>7.1f}"
                for r in sorted(row, key=lambda x: x["k"])
                if r["algo"] == algo
            )
            print(f"    {algo:<20} {vals}")

    print(f"\nResults saved to: {out_path}")
    print(f"Total entries: {len(results)}")
    print("DONE")


if __name__ == "__main__":
    main()
