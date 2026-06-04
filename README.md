# TIMM

Anonymous implementation package for an ICDM 2026 submission on temporal
influence maximization under Hawkes diffusion.

This repository contains the code, experiment scripts, generated figures, and
saved result summaries used by the paper. It is anonymized for peer review:
large raw datasets are not included, and personal or local-environment metadata
has been excluded from the package.

## Contents

- `src/tiimm/`: TIMM implementation, Hawkes diffusion model, TRR-set generation,
  martingale sample bounds, and baseline algorithms.
- `experiments/`: scripts for smoke tests, main comparisons, sensitivity
  studies, calibration, and figure generation.
- `figures/`: generated figures used in the manuscript.
- `results/`: saved JSON summaries used to produce tables and figures.
- `paper/`: optional anonymous manuscript files included for convenience. A
  code-only repository is also acceptable; these files can be omitted if the
  submission system already provides the paper.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Quick Check

Run the synthetic smoke test:

```bash
python experiments/smoke_test.py
```

The smoke test builds a small synthetic temporal graph, runs TIMM and several
baselines, and reports influence spread and runtime.

## Main Experiments

Representative commands:

```bash
python experiments/run_main.py
python experiments/eval_with_std.py
python experiments/generate_figures.py
python experiments/verify_submodularity.py
python experiments/calibration_trr.py
```

Dataset-specific scripts are also provided:

```bash
python experiments/run_higgs.py
python experiments/run_dblp.py
python experiments/run_reddit.py
python experiments/run_collegemsg.py
```

The raw real-world temporal datasets are public and should be placed under a
local `data/` directory if the dataset-specific scripts are rerun. The `data/`
directory is intentionally excluded from version control.

## Baseline Note

`MC-Hawkes-Greedy` is a direct Monte Carlo greedy baseline implemented under
the same Hawkes diffusion model as TIMM. It is included as a model-matched
comparison, not as a claim of an external prior baseline.

## Reproducibility Notes

- All scripts use fixed random seeds where applicable.
- Saved result summaries in `results/` correspond to the reported experiments.
- Generated figures can be reproduced with `python experiments/generate_figures.py`.
- Large raw datasets, local logs, virtual environments, and build artifacts are
  intentionally ignored by `.gitignore`.
