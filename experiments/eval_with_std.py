#!/usr/bin/env python3
"""Re-evaluate existing seed sets with N_RUNS=5 to get mean ± std."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiimm import *

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Seeds we already computed (from earlier experiments)
# Format: dataset -> {algo -> [seeds_for_k10, seeds_for_k20, seeds_for_k50]}
# We'll recompute seeds quickly to be safe

SEED = 42
MC_EVAL = 200  # reduced for speed, still statistically meaningful
N_RUNS = 5
EPS = 0.3
K_VALUES = [10, 20, 50]

results = []

for ds_name in ['higgs', 'dblp', 'reddit']:
    print(f'\n=== {ds_name} ===')
    g = load_dataset(ds_name, data_dir=data_dir)
    deg = g.m / max(g.n, 1)
    alpha = -np.log(1.0 - 0.7/deg) if deg > 0.7 else 0.5
    kernel = ExponentialHawkesKernel(alpha=round(alpha, 4), beta=1.0)
    horizon = 50.0

    for k in K_VALUES:
        print(f'  k={k}:', end=' ', flush=True)

        # TIMM
        t0 = time.time()
        timm = TIMM(g, kernel, horizon, seed=SEED)
        seeds, st = timm.run(k=k, epsilon=EPS, verbose=False)
        spreads = [timm.evaluate(seeds, MC_EVAL) for _ in range(N_RUNS)]
        elapsed = time.time() - t0
        results.append({
            'dataset': ds_name, 'algo': 'TIMM', 'k': k,
            'spread': round(np.mean(spreads), 1),
            'std': round(np.std(spreads), 1),
            'time_s': round(elapsed, 1)
        })
        print(f'TIMM={np.mean(spreads):.0f}±{np.std(spreads):.0f}', end=' ', flush=True)

        # StaticIMM
        t0 = time.time()
        si = StaticIMM(g, seed=SEED + 1)
        s_seeds, _ = si.run(k=k)
        spreads = [timm.evaluate(s_seeds, MC_EVAL) for _ in range(N_RUNS)]
        elapsed = time.time() - t0
        results.append({
            'dataset': ds_name, 'algo': 'StaticIMM', 'k': k,
            'spread': round(np.mean(spreads), 1),
            'std': round(np.std(spreads), 1),
            'time_s': round(elapsed, 1)
        })
        print(f'Static={np.mean(spreads):.0f}±{np.std(spreads):.0f}', end=' ', flush=True)

        # SnapshotIMM
        t0 = time.time()
        snap = SnapshotIMM(g, seed=SEED + 3)
        n_seeds, _ = snap.run(k=k)
        spreads = [timm.evaluate(n_seeds, MC_EVAL) for _ in range(N_RUNS)]
        elapsed = time.time() - t0
        results.append({
            'dataset': ds_name, 'algo': 'SnapshotIMM', 'k': k,
            'spread': round(np.mean(spreads), 1),
            'std': round(np.std(spreads), 1),
            'time_s': round(elapsed, 1)
        })
        print(f'Snap={np.mean(spreads):.0f}±{np.std(spreads):.0f}', end=' ', flush=True)

        # DegDiscount
        t0 = time.time()
        dd = DegreeDiscount(g, seed=SEED)
        dd_seeds, _ = dd.run(k=k)
        spreads = [timm.evaluate(dd_seeds, MC_EVAL) for _ in range(N_RUNS)]
        elapsed = time.time() - t0
        results.append({
            'dataset': ds_name, 'algo': 'DegDiscount', 'k': k,
            'spread': round(np.mean(spreads), 1),
            'std': round(np.std(spreads), 1),
            'time_s': round(elapsed, 1)
        })
        print(f'DD={np.mean(spreads):.0f}±{np.std(spreads):.0f}', end=' ', flush=True)

        # TempDegree
        t0 = time.time()
        td = TemporalDegree(g, beta=1.0, horizon=horizon, seed=SEED)
        td_seeds, _ = td.run(k=k)
        spreads = [timm.evaluate(td_seeds, MC_EVAL) for _ in range(N_RUNS)]
        elapsed = time.time() - t0
        results.append({
            'dataset': ds_name, 'algo': 'TempDegree', 'k': k,
            'spread': round(np.mean(spreads), 1),
            'std': round(np.std(spreads), 1),
            'time_s': round(elapsed, 1)
        })
        print(f'TD={np.mean(spreads):.0f}±{np.std(spreads):.0f}', end=' ', flush=True)

        # Random
        t0 = time.time()
        rd = RandomBaseline(g, seed=SEED)
        r_seeds, _ = rd.run(k=k)
        spreads = [timm.evaluate(r_seeds, MC_EVAL) for _ in range(N_RUNS)]
        elapsed = time.time() - t0
        results.append({
            'dataset': ds_name, 'algo': 'Random', 'k': k,
            'spread': round(np.mean(spreads), 1),
            'std': round(np.std(spreads), 1),
            'time_s': round(elapsed, 1)
        })
        print(f'Rand={np.mean(spreads):.0f}±{np.std(spreads):.0f}', flush=True)

    # Save incrementally
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results_with_std.json')
    json.dump(results, open(out_path, 'w'), indent=2)

print(f'\nDone. {len(results)} records -> results_with_std.json')
