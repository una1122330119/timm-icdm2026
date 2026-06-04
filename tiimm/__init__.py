"""
TIMM: Temporal IMM — Influence Maximization on Continuous-Time Networks
      under Hawkes Processes with Reverse Reachable Sampling.

ICDM 2026 submission package.
"""

from .temporal_graph import TemporalGraph, TemporalEdge
from .hawkes_model import ExponentialHawkesKernel, HawkesDiffusion
from .trr_sets import TRRSetGenerator
from .martingale import compute_trr_set_bound, compute_trr_set_bound_refined
from .timms import TIMM
from .baselines import (
    StaticIMM, SnapshotIMM, MCHawkesGreedy, CELF,
    DegreeDiscount, TemporalDegree, RandomBaseline,
)
from .evaluation import ExperimentRunner, ExperimentResult
from .datasets import (
    load_dataset, generate_synthetic_graph,
    load_snap_temporal, SNAP_DATASETS,
)

__version__ = "0.1.0"
__all__ = [
    "TemporalGraph",
    "TemporalEdge",
    "ExponentialHawkesKernel",
    "HawkesDiffusion",
    "TRRSetGenerator",
    "compute_trr_set_bound",
    "compute_trr_set_bound_refined",
    "TIMM",
    "StaticIMM",
    "SnapshotIMM",
    "MCHawkesGreedy",
    "CELF",
    "DegreeDiscount",
    "TemporalDegree",
    "RandomBaseline",
    "ExperimentRunner",
    "ExperimentResult",
    "load_dataset",
    "generate_synthetic_graph",
]
