#!/usr/bin/env python3
"""
SnapshotIMM Window Sensitivity Sweep — §6.3 empirical validation.

Demonstrates that SnapshotIMM's performance depends critically on
window size choice, and that even the best-of-grid window underperforms
TIMM. Tests: num_windows ∈ {5, 10, 20, 50, adaptive}.

Adaptive mode: sets num_windows = ceil(T_max / median_inter_event_time)
based on the actual temporal edge distribution.

Usage:
    python experiments/snapshot_window_sweep.py
    python experiments/snapshot_window_sweep.py --dataset higgs
"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiimm import (
    ExponentialHawkesKernel,
    TIMM,
    SnapshotIMM,
    RandomBaseline,
    load_dataset,
)
from tiimm.hawkes_model import HawkesDiffusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "snapshot_window_results.json")

SEED = 42
MC_EVAL = 200
EPS = 0.3
K_VALUES = [10, 20, 50]
WINDOW_SIZES = [5, 10, 20, 50]
BETA = 1.0

DATASET_CONFIGS = {
    "higgs": {"beta": 1.0, "horizon": 100.0},
    "dblp": {"beta": 1.0, "horizon": 100.0},
    "reddit": {"beta": 1.0, "horizon": 100.0},
    "collegemsg": {"beta": 1.0, "horizon": 100.0},
}


def compute_alpha(graph, beta, target_b=0.7):
    deg = graph.m / max(graph.n, 1)
    if deg > 1e-6:
        alpha = -beta * np.log(1.0 - target_b / deg)
    else:
        alpha = 0.5
    return max(0.01, min(0.99, alpha))


def compute_adaptive_windows(graph, base=10):
    """Adaptive window count based on inter-event time distribution."""
    all_times = []
    for u in range(graph.n):
        for e in graph.out_edges[u]:
            all_times.append(e.t)
    if len(all_times) < 2:
        return base
    all_times = sorted(all_times)
    gaps = np.diff(all_times)
    median_gap = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1.0
    t_span = max(all_times) - min(all_times)
    n_windows = max(5, min(100, int(np.ceil(t_span / max(median_gap * 10, 1e-6)))))
    return n_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--mc", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    all_results = []

    print("=" * 60)
    print("SnapshotIMM Window Sensitivity Sweep")
    print(f"  Windows: {WINDOW_SIZES} + adaptive")
    print(f"  MC eval: {args.mc}")
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

        alpha = compute_alpha(graph, config["beta"])
        kernel = ExponentialHawkesKernel(alpha=round(alpha, 4), beta=config["beta"])
        b_actual = (graph.m / max(graph.n, 1)) * (1.0 - np.exp(-alpha / config["beta"]))
        print(f"  alpha={alpha:.4f}, beta={config['beta']}, b={b_actual:.2f}")

        horizon = config["horizon"]
        diffuser = HawkesDiffusion(
            graph, kernel, horizon,
            rng=np.random.default_rng(args.seed + 999),
        )

        # Compute adaptive window count
        adaptive_windows = compute_adaptive_windows(graph)
        all_windows = WINDOW_SIZES + [adaptive_windows]
        window_labels = [f"w={w}" for w in WINDOW_SIZES] + [f"w={adaptive_windows} (adapt)"]

        # Run TIMM once as reference (same kernel, horizon, k)
        timm = TIMM(graph, kernel, horizon, seed=args.seed)

        for k in K_VALUES:
            print(f"\n  k={k}:")

            # TIMM reference
            t0 = time.time()
            t_seeds, t_stats = timm.run(k=k, epsilon=EPS, verbose=False)
            t_spread = diffuser.forward_cascade(set(t_seeds), args.mc)
            t_time = time.time() - t0
            print(f"    TIMM:         spread={t_spread:.1f}  time={t_time:.1f}s", flush=True)

            all_results.append({
                "dataset": ds_name, "algo": "TIMM", "k": k,
                "spread": round(t_spread, 1), "time_s": round(t_time, 1),
                "window": "N/A",
            })

            # Random baseline
            rb = RandomBaseline(graph, seed=args.seed)
            r_seeds, _ = rb.run(k=k)
            r_spread = diffuser.forward_cascade(set(r_seeds), args.mc)
            all_results.append({
                "dataset": ds_name, "algo": "Random", "k": k,
                "spread": round(r_spread, 1), "time_s": 0,
                "window": "N/A",
            })

            # SnapshotIMM with each window size
            best_snap_spread = 0
            best_snap_window = None

            for nw, label in zip(all_windows, window_labels):
                t0 = time.time()
                snap = SnapshotIMM(graph, num_windows=nw, seed=args.seed + 7)
                s_seeds, s_stats = snap.run(k=k)
                s_spread = diffuser.forward_cascade(set(s_seeds), args.mc)
                s_time = time.time() - t0

                if s_spread > best_snap_spread:
                    best_snap_spread = s_spread
                    best_snap_window = label

                print(f"    Snap({label:>12s}): spread={s_spread:.1f}  time={s_time:.1f}s", flush=True)

                all_results.append({
                    "dataset": ds_name, "algo": "SnapshotIMM", "k": k,
                    "spread": round(s_spread, 1), "time_s": round(s_time, 1),
                    "window": label,
                })

            gap = (t_spread - best_snap_spread) / max(best_snap_spread, 1) * 100
            print(f"    Best Snap:    spread={best_snap_spread:.1f} ({best_snap_window})")
            print(f"    TIMM vs best Snap: {gap:+.1f}%", flush=True)

            # Save incrementally
            json.dump(all_results, open(OUT, "w"), indent=2)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SNAPSHOT WINDOW SUMMARY")
    print(f"{'='*70}")

    for ds_name in datasets_to_run:
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        timm_results = {r["k"]: r for r in ds_results if r["algo"] == "TIMM"}
        snap_results = [r for r in ds_results if r["algo"] == "SnapshotIMM"]

        print(f"\n{ds_name}:")
        for k in K_VALUES:
            t = timm_results.get(k, {})
            snaps_k = [r for r in snap_results if r["k"] == k]
            if snaps_k:
                best = max(snaps_k, key=lambda r: r["spread"])
                worst = min(snaps_k, key=lambda r: r["spread"])
                print(f"  k={k:2d}: TIMM={t.get('spread',0):.0f}  "
                      f"Snap(best={best['window']})={best['spread']:.0f}  "
                      f"Snap(worst)={worst['spread']:.0f}  "
                      f"gap={(t.get('spread',0)-best['spread'])/max(best['spread'],1)*100:+.0f}%")

    print(f"\nSaved: {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
