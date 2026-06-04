#!/usr/bin/env python3
"""Branching factor sensitivity: vary b ∈ {0.3, 0.5, 0.7, 0.9} on DBLP."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiimm import *

SEED = 42
MC_EVAL = 200
EPS = 0.3
K = 20  # use k=20 as the primary comparison point

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
g = load_dataset('dblp', data_dir=data_dir)
print(f'DBLP: n={g.n} m={g.m} deg={g.m/max(g.n,1):.1f}')

# Diffuser for fair evaluation
from tiimm.hawkes_model import HawkesDiffusion

for target_b in [0.3, 0.5, 0.7, 0.9]:
    alpha = -np.log(1.0 - target_b / (g.m/max(g.n,1)))
    kernel = ExponentialHawkesKernel(alpha=round(alpha,4), beta=1.0)
    b_actual = (g.m/max(g.n,1)) * (1.0 - np.exp(-alpha))
    diff = HawkesDiffusion(g, kernel, 50.0, rng=np.random.default_rng(SEED+99))

    print(f'\nb={b_actual:.2f} (α={alpha:.4f}):', end=' ', flush=True)

    # TIMM
    t0 = time.time()
    timm = TIMM(g, kernel, 50.0, seed=SEED)
    t_seeds, _ = timm.run(k=K, epsilon=EPS, verbose=False)
    ti_sp = diff.forward_cascade(set(t_seeds), MC_EVAL)
    ti_time = time.time() - t0

    # StaticIMM
    si = StaticIMM(g, seed=SEED+1)
    s_seeds, _ = si.run(k=K)
    s_sp = diff.forward_cascade(set(s_seeds), MC_EVAL)

    # Random
    rd = RandomBaseline(g, seed=SEED)
    r_seeds, _ = rd.run(k=K)
    r_sp = diff.forward_cascade(set(r_seeds), MC_EVAL)

    print(f'TIMM={ti_sp:.1f} Static={s_sp:.1f} Random={r_sp:.1f} '
          f'TIMMgain={((ti_sp-s_sp)/max(s_sp,1)*100):+.0f}% '
          f'({ti_time:.1f}s)', flush=True)

print('\nDone.')
