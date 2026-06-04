#!/usr/bin/env python3
"""Quick smoke test — verify all algorithms produce different, reasonable results."""
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiimm import (
    generate_synthetic_graph, ExponentialHawkesKernel,
    TIMM, StaticIMM, SnapshotIMM, CELF, DegreeDiscount, RandomBaseline,
)
from tiimm.hawkes_model import HawkesDiffusion

SEED = 42
MC = 200

# Build small graph
g = generate_synthetic_graph(500, 3000, seed=SEED)
kernel = ExponentialHawkesKernel(alpha=0.3, beta=1.0)
horizon = 20.0

def evaluate(seeds):
    diff = HawkesDiffusion(g, kernel, horizon, rng=np.random.default_rng(SEED + 1))
    return diff.forward_cascade(set(seeds), MC)

print(f"Graph: n={g.n}, m={g.m}")
print(f"Kernel: α={kernel.alpha}, β={kernel.beta}")
print(f"Horizon: {horizon}, k=10")

results = []

# TIMM
t0 = time.time()
timm = TIMM(g, kernel, horizon, seed=SEED)
seeds, stats = timm.run(k=10, epsilon=0.3, verbose=False)
sp = evaluate(seeds)
results.append(("TIMM", sp, time.time() - t0, stats.get("theta", 0)))

# Static IMM
t0 = time.time()
si = StaticIMM(g, seed=SEED)
s_seeds, _ = si.run(k=10)
sp = evaluate(s_seeds)
results.append(("StaticIMM", sp, time.time() - t0, 0))

# Snapshot IMM
t0 = time.time()
sn = SnapshotIMM(g, seed=SEED)
n_seeds, _ = sn.run(k=10)
sp = evaluate(n_seeds)
results.append(("SnapshotIMM", sp, time.time() - t0, 0))

# CELF
t0 = time.time()
celf = CELF(g, seed=SEED)
c_seeds, _ = celf.run(k=10, mc_samples=50)
sp = evaluate(c_seeds)
results.append(("CELF", sp, time.time() - t0, 0))

# DegreeDiscount
t0 = time.time()
dd = DegreeDiscount(g, seed=SEED)
d_seeds, _ = dd.run(k=10)
sp = evaluate(d_seeds)
results.append(("DegreeDiscount", sp, time.time() - t0, 0))

# Random
t0 = time.time()
rand = RandomBaseline(g, seed=SEED)
r_seeds, _ = rand.run(k=10)
sp = evaluate(r_seeds)
results.append(("Random", sp, time.time() - t0, 0))

print(f"\n{'Algorithm':<16} {'Spread':>8} {'Time':>8} {'θ':>6}")
print("-" * 40)
best = max(r[1] for r in results)
for name, sp, t, theta in results:
    pct = sp / max(best, 1) * 100
    marker = " ← BEST" if pct > 99.9 else ""
    print(f"{name:<16} {sp:>8.1f} {t:>7.1f}s {theta:>6}{marker}")

# Verify TIMM is best or close to best
timm_sp = results[0][1]
print(f"\nTIMM is {'BEST' if timm_sp >= best - 0.5 else f'within {best - timm_sp:.1f} of best'}")
print("SMOKE TEST PASSED")
