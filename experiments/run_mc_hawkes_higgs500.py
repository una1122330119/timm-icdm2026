#!/usr/bin/env python3
"""MC-Hawkes-Greedy on Higgs-500 induced subgraph — fills real-subgraph gap."""
import sys, os, json, time
import numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tiimm import (
    ExponentialHawkesKernel, TIMM, MCHawkesGreedy, load_dataset,
    TemporalGraph,
)
from tiimm.hawkes_model import HawkesDiffusion

SEED = 42
MC_MC_HAWKES = 30  # reduced because MC greedy runs 500*n*MC cascades
MC_EVAL = 1000    # final comparison MC
EPS = 0.3
K = 10
BETA = 1.0
HORIZON = 100.0

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results",
    "mc_hawkes_higgs500.json",
)

print("=" * 60)
print("MC-Hawkes-Greedy on Higgs-500 Real Subgraph")
print("=" * 60)

# 1. Load Higgs
print("\n[1] Loading Higgs...", flush=True)
g_full = load_dataset("higgs", data_dir=DATA_DIR)
print(f"    Full: n={g_full.n}, m={g_full.m}")

# 2. Top-500 by out-degree
print("[2] Selecting top-500 nodes by out-degree...", flush=True)
out_deg = np.zeros(g_full.n, dtype=int)
for u in range(g_full.n):
    out_deg[u] = len(g_full.out_edges[u])
top500 = set(np.argpartition(-out_deg, 499)[:500])
print(f"    Top-500 degree range: [{out_deg[list(top500)].min()}, {out_deg[list(top500)].max()}]")

# 3. Build induced subgraph
print("[3] Building induced subgraph...", flush=True)
node_map = {old: new for new, old in enumerate(sorted(top500))}
edges_sub = []
for u_old in top500:
    u_new = node_map[u_old]
    for e in g_full.out_edges[u_old]:
        if e.v in top500:
            edges_sub.append((u_new, node_map[e.v], e.t))

g = TemporalGraph.from_edge_list(edges_sub, num_nodes=500)
print(f"    Subgraph: n={g.n}, m={g.m}")

# 4. Kernel
deg = g.m / max(g.n, 1)
alpha = -BETA * np.log(1.0 - 0.7 / deg) if deg > 0.7 else 0.5
alpha = round(alpha, 4)
kernel = ExponentialHawkesKernel(alpha=alpha, beta=BETA)
b_actual = deg * (1.0 - np.exp(-alpha / BETA))
print(f"    alpha={alpha}, beta={BETA}, b={b_actual:.2f}")

# 5. MC-Hawkes-Greedy
print(f"\n[4] MC-Hawkes-Greedy k={K} (MC={MC_MC_HAWKES})...", flush=True)
t0 = time.time()
mc_hawkes = MCHawkesGreedy(g, kernel, HORIZON, mc_samples=MC_MC_HAWKES, seed=SEED)
mc_hawkes_seeds, mc_hawkes_stats = mc_hawkes.run(k=K)
mc_hawkes_time = time.time() - t0
print(f"    Done: {mc_hawkes_time:.1f}s, {mc_hawkes_stats['total_evaluations']} evals")
print(f"    Seeds: {mc_hawkes_seeds}")

# 6. TIMM
print(f"\n[5] TIMM k={K} (epsilon={EPS})...", flush=True)
t0 = time.time()
timm = TIMM(g, kernel, HORIZON, seed=SEED)
t_seeds, t_stats = timm.run(k=K, epsilon=EPS, verbose=False)
t_time = time.time() - t0
print(f"    Done: {t_time:.1f}s, theta={t_stats['theta']}")
print(f"    Seeds: {t_seeds}")

# 7. Evaluate both on same MC
print(f"\n[6] MC evaluation (MC={MC_EVAL})...", flush=True)
diff = HawkesDiffusion(g, kernel, HORIZON, rng=np.random.default_rng(SEED + 999))
mc_hawkes_spread = diff.forward_cascade(set(mc_hawkes_seeds), MC_EVAL)
t_spread = diff.forward_cascade(set(t_seeds), MC_EVAL)

# 8. StaticIMM + Random baselines
from tiimm import StaticIMM, RandomBaseline
si = StaticIMM(g, seed=SEED + 1)
s_seeds, _ = si.run(k=K)
s_spread = diff.forward_cascade(set(s_seeds), MC_EVAL)

rb = RandomBaseline(g, seed=SEED)
r_seeds, _ = rb.run(k=K)
r_spread = diff.forward_cascade(set(r_seeds), MC_EVAL)

# 9. Report
overlap = len(set(mc_hawkes_seeds) & set(t_seeds))
print(f"\n{'='*60}")
print("RESULTS: Higgs-500 Real Subgraph")
print(f"{'='*60}")
print(f"  MC-Hawkes-Greedy:  spread={mc_hawkes_spread:.1f}  time={mc_hawkes_time:.0f}s  seeds={mc_hawkes_seeds}")
print(f"  TIMM:         spread={t_spread:.1f}  time={t_time:.1f}s  seeds={t_seeds}")
print(f"  StaticIMM:    spread={s_spread:.1f}")
print(f"  Random:       spread={r_spread:.1f}")
print(f"  Seed overlap: {overlap}/{K}")
print(f"  TIMM vs MC-Hawkes-Greedy gap:   {(t_spread-mc_hawkes_spread)/max(mc_hawkes_spread,1)*100:+.1f}%")
print(f"  TIMM vs MC-Hawkes-Greedy speed: {mc_hawkes_time/max(t_time,0.01):.0f}x")
print(f"  TIMM vs Static:   {(t_spread-s_spread)/max(s_spread,1)*100:+.1f}%")

result = {
    "dataset": "Higgs-500 (real subgraph)",
    "n": g.n, "m": g.m,
    "alpha": alpha, "beta": BETA, "b": round(b_actual, 2),
    "k": K,
    "mc_hawkes_spread": round(mc_hawkes_spread, 1),
    "mc_hawkes_time_s": round(mc_hawkes_time, 1),
    "mc_hawkes_evals": mc_hawkes_stats["total_evaluations"],
    "mc_hawkes_seeds": mc_hawkes_seeds,
    "TIMM_spread": round(t_spread, 1),
    "TIMM_time_s": round(t_time, 1),
    "TIMM_theta": t_stats["theta"],
    "TIMM_seeds": t_seeds,
    "StaticIMM_spread": round(s_spread, 1),
    "Random_spread": round(r_spread, 1),
    "seed_overlap": overlap,
    "gap_pct": round((t_spread-mc_hawkes_spread)/max(mc_hawkes_spread,1)*100, 1),
    "speedup": round(mc_hawkes_time/max(t_time, 0.01)),
}
json.dump(result, open(OUT, "w"), indent=2)
print(f"\nSaved: {OUT}")
print("DONE")


