#!/usr/bin/env python3
"""CollegeMsg only — small graph, can run MC-Hawkes-Greedy + CELF."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiimm import (
    ExponentialHawkesKernel, TIMM, StaticIMM, SnapshotIMM,
    MCHawkesGreedy, CELF, DegreeDiscount, TemporalDegree, RandomBaseline,
    load_dataset,
)

SEED, MC, EPS, K, H = 42, 1000, 0.3, [10, 20, 50], 100.0
# Kernel auto-tuned per dataset: target effective branching b ≈ 0.7 (subcritical).
#   b = avg_deg · (1 − exp(−α/β))  →  α = −β · ln(1 − 0.7/avg_deg)
# Subcritical cascades ensure seed selection matters (avoid saturation).
# (alpha, beta) set below after loading the graph.
OUT = os.path.join(os.path.dirname(__file__), "..", "real_collegemsg.json")
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

print("Loading CollegeMsg...")
g = load_dataset("collegemsg", data_dir=data_dir)
deg = g.m / max(g.n, 1)
beta = 1.0
if deg > 1.5:
    target_b = 0.7  # subcritical: avoid cascade saturation on dense graphs
    alpha = -beta * np.log(1.0 - target_b / deg)
else:
    alpha = 0.5  # sparse graph: keep moderate α
KERNEL = ExponentialHawkesKernel(alpha=round(alpha, 4), beta=beta)
b_actual = deg * (1.0 - np.exp(-KERNEL.alpha))
print(f"  n={g.n}, m={g.m}, avg_deg={deg:.1f} → α={KERNEL.alpha:.4f} (b≈{b_actual:.2f})")

results = []
timm = TIMM(g, KERNEL, H, seed=SEED)

for k in K:
    print(f"\nk={k}:", flush=True)

    # TIMM
    t0 = time.time(); seeds, st = timm.run(k=k, epsilon=EPS, verbose=False)
    sp = np.mean([timm.evaluate(seeds, MC) for _ in range(3)])
    results.append({"ds":"CollegeMsg","algo":"TIMM","k":k,"spread":round(sp,1),
                    "time_s":round(time.time()-t0,1),"theta":st.get("theta",0)})
    print(f"  TIMM         sp={sp:.0f}  {time.time()-t0:.0f}s", flush=True)

    # StaticIMM
    t0 = time.time(); ss,_ = StaticIMM(g,seed=SEED+1).run(k=k)
    sp = np.mean([timm.evaluate(ss, MC) for _ in range(3)])
    results.append({"ds":"CollegeMsg","algo":"StaticIMM","k":k,"spread":round(sp,1),
                    "time_s":round(time.time()-t0,1)})
    print(f"  StaticIMM    sp={sp:.0f}  {time.time()-t0:.0f}s", flush=True)

    # SnapshotIMM
    t0 = time.time(); ns,_ = SnapshotIMM(g,seed=SEED+3).run(k=k)
    sp = np.mean([timm.evaluate(ns, MC) for _ in range(3)])
    results.append({"ds":"CollegeMsg","algo":"SnapshotIMM","k":k,"spread":round(sp,1),
                    "time_s":round(time.time()-t0,1)})
    print(f"  SnapshotIMM  sp={sp:.0f}  {time.time()-t0:.0f}s", flush=True)

    # MC-Hawkes-Greedy (MC=200 — expensive)
    try:
        mc_hawkes = MCHawkesGreedy(g, KERNEL, H, mc_samples=200, seed=SEED)
        t0 = time.time(); mc_hawkes_seeds,_ = mc_hawkes.run(k=k)
        from tiimm.hawkes_model import HawkesDiffusion
        diff = HawkesDiffusion(g, KERNEL, H, rng=np.random.default_rng(SEED+1))
        sp = diff.forward_cascade(set(mc_hawkes_seeds), MC)
        results.append({"ds":"CollegeMsg","algo":"MCHawkesGreedy","k":k,"spread":round(sp,1),
                        "time_s":round(time.time()-t0,1)})
        print(f"  MC-Hawkes-Greedy    sp={sp:.0f}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  MC-Hawkes-Greedy    FAIL: {e}")

    # CELF
    try:
        t0 = time.time(); cs,_ = CELF(g,seed=SEED).run(k=k, mc_samples=100)
        sp = np.mean([timm.evaluate(cs, MC) for _ in range(3)])
        results.append({"ds":"CollegeMsg","algo":"CELF","k":k,"spread":round(sp,1),
                        "time_s":round(time.time()-t0,1)})
        print(f"  CELF         sp={sp:.0f}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  CELF         FAIL: {e}")

    # Fast heuristics
    for Cls, name in [(DegreeDiscount,"DegDiscount"),(lambda g,seed: TemporalDegree(g,beta=KERNEL.beta,horizon=H,seed=seed),"TempDegree"),(RandomBaseline,"Random")]:
        t0 = time.time(); ds,_ = Cls(g,seed=SEED).run(k=k)
        sp = np.mean([timm.evaluate(ds, MC) for _ in range(3)])
        results.append({"ds":"CollegeMsg","algo":name,"k":k,"spread":round(sp,1),
                        "time_s":round(time.time()-t0,1)})
        print(f"  {name:<12} sp={sp:.0f}  {time.time()-t0:.0f}s", flush=True)

    json.dump(results, open(OUT,"w"), indent=2)

print(f"\nSaved to {OUT} ({len(results)} records)")
