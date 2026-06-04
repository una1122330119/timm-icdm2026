#!/usr/bin/env python3
"""MC-Hawkes-Greedy comparison on Synth-S (n=500) — fast enough to complete."""
import sys, os, time
import numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tiimm import *
from tiimm.hawkes_model import HawkesDiffusion

SEED = 42
MC_EVAL = 1000

# Synth-S with controlled parameters
g = generate_synthetic_graph(n=500, m=2000, seed=SEED)
# Use subcritical kernel (b<1) so MC-Hawkes-Greedy can complete in reasonable time
kernel = ExponentialHawkesKernel(alpha=0.15, beta=1.0)
horizon = 20.0
deg = g.m / max(g.n, 1)
b = deg * (1.0 - np.exp(-kernel.alpha))
print(f'Synth-S: n={g.n}  m={g.m}  deg={deg:.1f}  a=0.15  b={b:.2f}')
assert b < 1.0, f'Need subcritical kernel, got b={b:.2f}'

for k in [10, 20]:
    print(f'\n--- k={k} ---')

    # MC-Hawkes-Greedy (MC=100 for speed, subcritical cascade)
    print('  MC-Hawkes-Greedy (MC=100)...', end=' ', flush=True)
    mc_hawkes = MCHawkesGreedy(g, kernel, horizon, mc_samples=100, seed=SEED)
    t0 = time.time()
    mc_hawkes_seeds, mc_hawkes_st = mc_hawkes.run(k=k)
    mc_hawkes_time = time.time() - t0
    print(f'{mc_hawkes_time:.1f}s  evals={mc_hawkes_st["total_evaluations"]}')

    # TIMM
    print('  TIMM...', end=' ', flush=True)
    timm = TIMM(g, kernel, horizon, seed=SEED)
    t_seeds, t_st = timm.run(k=k, epsilon=0.3, verbose=False)
    ti_time = t_st['total_time_s']
    print(f'{ti_time:.1f}s  theta={t_st["theta"]}')

    # StaticIMM
    print('  StaticIMM...', end=' ', flush=True)
    si = StaticIMM(g, seed=SEED + 1)
    s_seeds, _ = si.run(k=k)

    # Random
    rd = RandomBaseline(g, seed=SEED)
    r_seeds, _ = rd.run(k=k)

    # Evaluate all with same evaluator
    diff = HawkesDiffusion(g, kernel, horizon, rng=np.random.default_rng(SEED + 99))
    mc_hawkes_sp = diff.forward_cascade(set(mc_hawkes_seeds), MC_EVAL)
    ti_sp = diff.forward_cascade(set(t_seeds), MC_EVAL)
    s_sp = diff.forward_cascade(set(s_seeds), MC_EVAL)
    r_sp = diff.forward_cascade(set(r_seeds), MC_EVAL)

    # Jaccard
    jac_mc_ti = len(set(mc_hawkes_seeds) & set(t_seeds)) / max(
        len(set(mc_hawkes_seeds) | set(t_seeds)), 1
    )

    print(f'  Results:')
    print(f'    MC-Hawkes-Greedy spread={mc_hawkes_sp:.1f}  time={mc_hawkes_time:.1f}s  seeds={sorted(mc_hawkes_seeds)[:5]}...')
    print(f'    TIMM  spread={ti_sp:.1f}  time={ti_time:.1f}s  seeds={sorted(t_seeds)[:5]}...')
    print(f'    Static spread={s_sp:.1f}')
    print(f'    Random spread={r_sp:.1f}')
    print(f'    MC-Hawkes/TIMM Jaccard={jac_mc_ti:.3f}')
    print(f'    TIMM speedup: {mc_hawkes_time/max(ti_time,0.1):.0f}x')
    print(f'    TIMM gap vs MC-Hawkes-Greedy: {(ti_sp-mc_hawkes_sp)/max(mc_hawkes_sp,1)*100:+.1f}%')

print('\nDone.')


