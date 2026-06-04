#!/usr/bin/env python3
"""Reddit only — medium graph (n≈56K), fast heuristics + TIMM."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiimm import (
    ExponentialHawkesKernel, TIMM, StaticIMM, SnapshotIMM,
    DegreeDiscount, TemporalDegree, RandomBaseline,
    load_dataset,
)

SEED, MC, EPS, K, H = 42, 1000, 0.3, [10, 20, 50], 100.0
OUT = os.path.join(os.path.dirname(__file__), "..", "real_reddit.json")
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

print("Loading Reddit...")
g = load_dataset("reddit", data_dir=data_dir)
deg = g.m / max(g.n, 1)
beta = 1.0
if deg > 1.5:
    target_b = 0.7  # subcritical: avoid cascade saturation on dense graphs
    alpha = -beta * np.log(1.0 - target_b / deg)
else:
    alpha = 0.5  # sparse graph: keep moderate α
KERNEL = ExponentialHawkesKernel(alpha=round(alpha, 4), beta=beta)
b_actual = deg * (1.0 - np.exp(-KERNEL.alpha))
print(f"  n={g.n}, m={g.m}, avg_deg={deg:.2f} → α={KERNEL.alpha:.4f} (b≈{b_actual:.2f})")

results = []
timm = TIMM(g, KERNEL, H, seed=SEED)

for k in K:
    if k > g.n: continue
    print(f"\nk={k}:", flush=True)

    for Cls, name in [
        (lambda: TIMM(g,KERNEL,H,seed=SEED), "TIMM"),
        (lambda: StaticIMM(g,seed=SEED+1), "StaticIMM"),
        (lambda: SnapshotIMM(g,seed=SEED+3), "SnapshotIMM"),
        (lambda: DegreeDiscount(g,seed=SEED), "DegDiscount"),
        (lambda: TemporalDegree(g,beta=KERNEL.beta,horizon=H,seed=SEED), "TempDegree"),
        (lambda: RandomBaseline(g,seed=SEED), "Random"),
    ]:
        t0 = time.time()
        algo = Cls()
        if name == "TIMM":
            seeds, st = algo.run(k=k, epsilon=EPS, verbose=False)
            extra = {"theta": st.get("theta", 0)}
        else:
            seeds, _ = algo.run(k=k)
            extra = {}
        sp = np.mean([timm.evaluate(seeds, MC) for _ in range(3)])
        elapsed = time.time() - t0
        results.append({"ds":"Reddit","algo":name,"k":k,"spread":round(sp,1),
                        "time_s":round(elapsed,1), **extra})
        print(f"  {name:<12} sp={sp:.0f}  {elapsed:.0f}s", flush=True)

    json.dump(results, open(OUT,"w"), indent=2)

print(f"\nSaved to {OUT} ({len(results)} records)")
