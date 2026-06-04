"""
Dataset loaders for temporal graphs.

Supports loading from:
- Edge list CSV (u, v, t)
- SNAP temporal network format
- Custom formats for common IM benchmark datasets
"""

import os
import numpy as np
import gzip
import urllib.request
from datetime import datetime
from typing import List, Tuple, Optional
from .temporal_graph import TemporalGraph

# Known dataset URLs and metadata
DATASET_INFO = {
    "higgs-twitter": {
        "url": "https://snap.stanford.edu/data/higgs-twitter.tar.gz",
        "nodes": 456631,
        "edges": 14855844,
        "description": "Twitter retweet network during Higgs boson discovery",
        "format": "snap-temporal",
    },
    "memetracker": {
        "url": "https://snap.stanford.edu/data/memetracker9.tar.gz",
        "nodes": None,  # inferred
        "edges": None,
        "description": "Meme phrases tracked across online media",
        "format": "snap-temporal",
    },
    "dblp": {
        "url": "https://snap.stanford.edu/data/com-dblp.ungraph.txt.gz",
        "nodes": 317080,
        "edges": 1049866,
        "description": "DBLP co-authorship network",
        "format": "snap-static",  # needs temporal augmentation
    },
    "reddit-hyperlinks": {
        "url": "https://snap.stanford.edu/data/soc-RedditHyperlinks-body.tsv",
        "nodes": 55863,
        "edges": 858490,
        "description": "Reddit hyperlink network with timestamps",
        "format": "tsv-temporal",
    },
}


def load_from_edge_list(
    path: str,
    num_nodes: Optional[int] = None,
    delimiter: str = ",",
    has_header: bool = False,
    src_col: int = 0,
    dst_col: int = 1,
    time_col: int = 2,
) -> TemporalGraph:
    """
    Load a temporal graph from a CSV edge list.

    Format: u,v,t (one edge per line)

    Parameters
    ----------
    path : str
        Path to the CSV file.
    num_nodes : int, optional
        Pre-specified number of nodes. Inferred if not given.
    delimiter : str
        Column delimiter.
    has_header : bool
        Whether the first line is a header.
    src_col, dst_col, time_col : int
        Column indices for source, destination, and timestamp.

    Returns
    -------
    TemporalGraph
    """
    edges = []
    with open(path, "r") as f:
        if has_header:
            next(f)
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter)
            u = int(parts[src_col])
            v = int(parts[dst_col])
            t = float(parts[time_col])
            edges.append((u, v, t))

    return TemporalGraph.from_edge_list(edges, num_nodes=num_nodes)


def load_higgs_twitter(data_dir: str = "data/higgs") -> TemporalGraph:
    """
    Load the Higgs Twitter temporal network.

    The Higgs dataset contains retweet activity during the
    discovery of the Higgs boson (July 2012). Each edge (u,v,t)
    means user u retweeted user v at Unix timestamp t.

    Parameters
    ----------
    data_dir : str
        Directory containing the extracted dataset files.

    Returns
    -------
    TemporalGraph
    """
    # Look for the retweet file
    rt_path = os.path.join(data_dir, "higgs-retweet_network.edgelist")
    if not os.path.exists(rt_path):
        # Try alternate names
        for fname in ["retweet.txt", "higgs-retweet.txt"]:
            alt = os.path.join(data_dir, fname)
            if os.path.exists(alt):
                rt_path = alt
                break

    if not os.path.exists(rt_path):
        raise FileNotFoundError(
            f"Higgs dataset not found at {data_dir}. "
            f"Download from: {DATASET_INFO['higgs-twitter']['url']}"
        )

    edges = []
    with open(rt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u, v, t = int(parts[0]), int(parts[1]), float(parts[2])
            edges.append((u, v, t))

    return TemporalGraph.from_edge_list(
        edges, num_nodes=DATASET_INFO["higgs-twitter"]["nodes"]
    )


def load_dblp_temporal(data_dir: str = "data/dblp") -> TemporalGraph:
    """
    Load DBLP citation network. Uses row order as proxy for time.

    Raw DBLP author IDs are sparse (up to ~10M). We re-index them to
    0..N-1 where N is the number of unique authors with citations.
    """
    cite_path = None
    for p in [os.path.join(data_dir, "dblp_citations.txt"),
              os.path.join(data_dir, "dblp-citations.txt"),
              os.path.join(data_dir, "cit-HepTh.txt")]:
        if os.path.isfile(p):
            cite_path = p; break
    if cite_path is None:
        raise FileNotFoundError(f"DBLP not found at {data_dir}")

    # First pass: collect unique node IDs
    raw_edges = []
    node_set = set()
    with open(cite_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                raw_edges.append((u, v))
                node_set.add(u)
                node_set.add(v)

    # Re-index to 0..N-1
    node_map = {old: new for new, old in enumerate(sorted(node_set))}
    n_unique = len(node_map)
    m = len(raw_edges)
    # Reverse direction for correct influence semantics:
    #   raw (u→v) = "u cites v" → influence flows v→u (cited → citing)
    #   TRR-set backward walk follows influence direction.
    edges = [(node_map[v], node_map[u], 100.0 * i / m)
             for i, (u, v) in enumerate(raw_edges)]
    print(f"  DBLP: {m} edges (reversed: cited→citing), "
          f"{n_unique} unique nodes (re-indexed), t in [0, 100]")
    return TemporalGraph.from_edge_list(edges, num_nodes=n_unique)


def load_reddit_hyperlinks(data_dir: str = "data/reddit") -> TemporalGraph:
    """
    Load the Reddit Hyperlink network.

    Format: TSV with columns including SOURCE_SUBREDDIT,
    TARGET_SUBREDDIT, TIMESTAMP, etc.

    Searches multiple locations: data_dir, data/, data/reddit/, etc.

    Parameters
    ----------
    data_dir : str
        Directory or file path.

    Returns
    -------
    TemporalGraph
    """
    # Search for the file in multiple locations
    candidates = [
        os.path.join(data_dir, "soc-RedditHyperlinks-body.tsv"),
        os.path.join(data_dir, "reddit_hyperlinks.tsv"),
        os.path.join("data", "soc-RedditHyperlinks-body.tsv"),
        os.path.join("data", "reddit", "soc-RedditHyperlinks-body.tsv"),
        os.path.join("data", "reddit_hyperlinks.tsv"),
        data_dir,  # if data_dir itself is the file
    ]
    path = None
    for c in candidates:
        if os.path.isfile(c):
            path = c
            break
    if path is None:
        raise FileNotFoundError(
            f"Reddit hyperlinks not found. Searched: {candidates}. "
            f"Download: curl -L -o data/reddit_hyperlinks.tsv "
            f"https://snap.stanford.edu/data/soc-RedditHyperlinks-body.tsv"
        )
    print(f"  Loading Reddit from: {path}")

    edges = []
    node_map = {}
    next_id = 0

    with open(path, "r") as f:
        header = next(f).strip().split("\t")
        # Find column indices
        try:
            src_idx = header.index("SOURCE_SUBREDDIT")
            dst_idx = header.index("TARGET_SUBREDDIT")
            ts_idx = header.index("TIMESTAMP")
        except ValueError:
            # Fallback: use positional indices
            src_idx, dst_idx, ts_idx = 0, 1, 3

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < max(src_idx, dst_idx, ts_idx) + 1:
                continue

            src_name = parts[src_idx]
            dst_name = parts[dst_idx]
            ts_str = parts[ts_idx]
            try:
                t = float(ts_str)
            except ValueError:
                # Parse datetime format: "2013-12-31 16:39:58"
                t = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S").timestamp()

            if src_name not in node_map:
                node_map[src_name] = next_id
                next_id += 1
            if dst_name not in node_map:
                node_map[dst_name] = next_id
                next_id += 1

            edges.append((node_map[src_name], node_map[dst_name], t))

    # Normalize timestamps to [0, 100] range for Hawkes horizon compatibility
    if edges:
        t_min = min(t for _, _, t in edges)
        t_max = max(t for _, _, t in edges)
        t_span = t_max - t_min
        if t_span > 1000:  # scale down if span is large (Unix timestamps)
            edges = [(u, v, 100.0 * (t - t_min) / t_span) for u, v, t in edges]
        else:
            edges = [(u, v, t - t_min) for u, v, t in edges]
        print(f"  Normalized: {len(edges)} edges, t ∈ [0, 100] (from {t_min:.0f}–{t_max:.0f})")

    return TemporalGraph.from_edge_list(edges, num_nodes=next_id)


def generate_synthetic_graph(
    n: int = 1000,
    m: int = 10000,
    time_horizon: float = 100.0,
    seed: int = 42,
) -> TemporalGraph:
    """
    Generate a synthetic temporal graph for testing.

    Uses a simple model:
    - Nodes have community structure (clusters)
    - Edges within clusters have higher density
    - Timestamps follow Hawkes-like clustering in time

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of temporal edges.
    time_horizon : float
        Maximum timestamp.
    seed : int
        Random seed.

    Returns
    -------
    TemporalGraph
    """
    rng = np.random.default_rng(seed)
    n_clusters = 5
    cluster_size = n // n_clusters
    cluster = np.repeat(np.arange(n_clusters), cluster_size)[:n]

    edges = []
    for _ in range(m):
        u = rng.integers(0, n)

        # Within-cluster edge with higher probability
        if rng.random() < 0.7:
            v_candidates = np.where(cluster == cluster[u])[0]
            v = rng.choice(v_candidates) if len(v_candidates) > 0 else rng.integers(0, n)
        else:
            v = rng.integers(0, n)

        if u == v:
            continue

        # Hawkes-like timestamp clustering
        if len(edges) > 0 and rng.random() < 0.4:
            # Cascade: time close to a recent edge
            recent_t = edges[-1][2]
            t = recent_t + rng.exponential(1.0)
        else:
            t = rng.exponential(time_horizon / 10)

        t = min(t, time_horizon)
        edges.append((u, v, t))

    return TemporalGraph.from_edge_list(edges, num_nodes=n)


def load_digg_votes(path: str) -> TemporalGraph:
    """
    Load Digg votes network and convert to temporal diffusion cascades.

    Raw format: user_id story_id weight timestamp (bipartite)
    Conversion: for each story, sort voters by time, create edges
    between consecutive voters (diffusion cascade per story).
    """
    import gzip as _gzip
    opener = _gzip.open if path.endswith(".gz") else open
    edges = []
    story_votes = {}  # story_id -> [(user_id, timestamp)]

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            user = int(parts[0])
            story = int(parts[1])
            ts = float(parts[-1])  # last column is timestamp
            if story not in story_votes:
                story_votes[story] = []
            story_votes[story].append((user, ts))

    # Build temporal edges: for each story, chain voters by time
    for story, votes in story_votes.items():
        votes.sort(key=lambda x: x[1])  # sort by timestamp
        for i in range(len(votes) - 1):
            u, t1 = votes[i]
            v, t2 = votes[i + 1]
            if t1 < t2:
                edges.append((u, v, t2))

    # Normalize timestamps
    if edges:
        t_min = min(t for _, _, t in edges)
        t_span = max(t for _, _, t in edges) - t_min
        if t_span > 0:
            edges = [(u, v, 100.0 * (t - t_min) / t_span) for u, v, t in edges]
        else:
            edges = [(u, v, 0.0) for u, v, t in edges]

    print(f"  Digg cascade edges: {len(edges)} (from {len(story_votes)} stories)")
    return TemporalGraph.from_edge_list(edges)


def load_higgs_activity(path: str) -> TemporalGraph:
    """
    Load Higgs Twitter activity file and extract retweet cascades.

    Format: userA userB timestamp interaction
    Interaction: RT (retweet), MT (mention), RE (reply)
    Filter: RT edges only, direction reversed (B→A: B posts, A retweets).

    Twitter user IDs are sparse (up to ~500K). We re-index to 0..N-1.
    """
    opener = gzip.open if path.endswith(".gz") else open
    raw_edges = []
    node_set = set()
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            u, v, t, action = int(parts[0]), int(parts[1]), float(parts[2]), parts[3]
            if action != "RT":
                continue
            # Reverse: information flows from the retweeted user (v) to the retweeter (u)
            raw_edges.append((v, u, t))
            node_set.add(u)
            node_set.add(v)

    # Re-index to 0..N-1
    node_map = {old: new for new, old in enumerate(sorted(node_set))}
    n_unique = len(node_map)
    edges = [(node_map[u], node_map[v], t) for u, v, t in raw_edges]

    # Normalize timestamps
    if edges:
        t_min = min(t for _, _, t in edges)
        t_span = max(t for _, _, t in edges) - t_min
        if t_span > 0:
            edges = [(u, v, 100.0 * (t - t_min) / t_span) for u, v, t in edges]

    print(f"  Higgs RT edges: {len(edges)}, {n_unique} unique users (re-indexed)")
    return TemporalGraph.from_edge_list(edges, num_nodes=n_unique)


def load_snap_temporal(
    path: str,
    num_nodes: Optional[int] = None,
    normalize_t: bool = True,
    t_max: float = 100.0,
) -> TemporalGraph:
    """
    Load a SNAP temporal edge list.

    Format: u v t (space/tab-separated, no header, one edge per line).
    Used by: CollegeMsg, SuperUser, Higgs, WikiTalk, MathOverflow, etc.

    Parameters
    ----------
    path : str
        Path to the .txt or .txt.gz file.
    num_nodes : int, optional
        Pre-specified number of nodes. Inferred if None.
    normalize_t : bool
        If True, normalize timestamps to [0, t_max].
    t_max : float
        Maximum timestamp after normalization.

    Returns
    -------
    TemporalGraph
    """
    # Try gzip first (some .tsv files are actually gzipped), fall back to plain text
    edges = []
    try:
        f = gzip.open(path, "rt", encoding="utf-8")
        f.readline()  # test read
        f.seek(0)
    except (gzip.BadGzipFile, UnicodeDecodeError, OSError):
        f = open(path, "r", encoding="utf-8")
    with f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                u, v, t = int(parts[0]), int(parts[1]), float(parts[2])
                edges.append((u, v, t))

    if normalize_t and edges:
        t_min = min(t for _, _, t in edges)
        t_span = max(t for _, _, t in edges) - t_min
        if t_span > 0:
            edges = [(u, v, t_max * (t - t_min) / t_span) for u, v, t in edges]
        else:
            # All timestamps identical — shift to 0 to avoid Hawkes underflow
            edges = [(u, v, 0.0) for u, v, t in edges]
        print(f"  Normalized: {len(edges)} edges, t ∈ [0, {t_max:.0f}]")

    return TemporalGraph.from_edge_list(edges, num_nodes=num_nodes)


# ── Dataset registry ──────────────────────────────────────────────────

SNAP_DATASETS = {
    "collegemsg": {
        "file": "CollegeMsg.txt",
        "nodes": 1899,
        "desc": "UC Irvine private messages, 193 days",
    },
    "superuser": {
        "file": "sx-superuser.txt",
        "nodes": 194085,
        "desc": "SuperUser Q&A, 2773 days",
    },
    "higgs": {
        "file": "higgs-retweet_network.txt",
        "nodes": 456631,
        "desc": "Higgs boson retweets, 7 days",
    },
    "wikitalk": {
        "file": "wiki-talk-temporal.txt",
        "nodes": 1140149,
        "desc": "Wikipedia user talk, 2321 days",
    },
    "mathoverflow": {
        "file": "sx-mathoverflow.txt",
        "nodes": 24818,
        "desc": "MathOverflow Q&A, 2350 days",
    },
}


def load_dataset(name: str, data_dir: str = "data") -> TemporalGraph:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        One of: "higgs-twitter", "dblp", "reddit", "synthetic-small",
        "synthetic-medium"
    data_dir : str
        Base data directory.

    Returns
    -------
    TemporalGraph
    """
    # Check specific named datasets first (before generic SNAP registry)
    if name == "mathoverflow":
        path = os.path.join(data_dir, "math_overflow.tsv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mathoverflow: {path} not found")
        print(f"  Loading MathOverflow Q&A network")
        return load_snap_temporal(path, num_nodes=24818)

    elif name == "reddit":
        return load_reddit_hyperlinks(data_dir)

    elif name == "digg":
        path = os.path.join(data_dir, "digg-votes", "out.digg-votes")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Digg votes not found. Download: "
                f"curl -L -o data/digg-votes.tar.bz2 "
                f"http://konect.cc/files/download.tsv.digg-votes.tar.bz2"
                f" && tar -xjf data/digg-votes.tar.bz2 -C data/"
            )
        print(f"  Loading Digg voting cascades")
        return load_digg_votes(path)

    elif name == "higgs-twitter" or name == "higgs":
        path = os.path.join(data_dir, "higgs-activity_time.txt.gz")
        if not os.path.isfile(path):
            path = os.path.join(data_dir, "higgs-activity_time.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Higgs activity not found. Download: "
                f"curl -L -o data/higgs-activity_time.txt.gz "
                f"https://snap.stanford.edu/data/higgs-activity_time.txt.gz"
            )
        print(f"  Loading Higgs Twitter retweet cascades")
        return load_higgs_activity(path)
    elif name == "dblp":
        return load_dblp_temporal(data_dir)

    # Generic SNAP temporal datasets (auto-detect from registry)
    elif name in SNAP_DATASETS:
        info = SNAP_DATASETS[name]
        path = os.path.join(data_dir, info["file"])
        if not os.path.isfile(path):
            path_gz = path + ".gz"
            if os.path.isfile(path_gz):
                path = path_gz
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{name} not found at {path}. Download from SNAP: "
                f"https://snap.stanford.edu/data/{info['file']}.gz"
            )
        print(f"  Loading {name}: {info['desc']}")
        return load_snap_temporal(path, num_nodes=info["nodes"])
    elif name == "synthetic-small":
        return generate_synthetic_graph(n=500, m=5000, seed=42)
    elif name == "synthetic-medium":
        return generate_synthetic_graph(n=2000, m=50000, seed=42)
    elif name == "synthetic-large":
        return generate_synthetic_graph(n=10000, m=200000, seed=42)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Known: {list(SNAP_DATASETS.keys())} + "
                         f"reddit, mathoverflow, dblp, synthetic-*")
