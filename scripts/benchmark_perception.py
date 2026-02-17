#!/usr/bin/env python3
# Copyright 2025 AMR Stack Authors
# Licensed under MIT
#
# Standalone benchmark script for the obstacle detector perception pipeline.
# Generates synthetic point clouds (random clusters on a ground plane) and
# measures clustering + classification throughput without a ROS runtime.
#
# Usage:
#   python3 scripts/benchmark_perception.py
#   python3 scripts/benchmark_perception.py --num_points 100000 --iterations 50

import argparse
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import the classification function from the obstacle detector module.
# ---------------------------------------------------------------------------

sys.path.insert(0, "src")

from amr_perception.nodes.obstacle_detector.obstacle_detector_node import classify_cluster

# Default classification parameters (same as the ROS node defaults).
DEFAULT_PARAMS: Dict[str, float] = {
    "person_width_min": 0.3,
    "person_width_max": 0.8,
    "person_height_min": 1.2,
    "person_height_max": 2.0,
    "shelf_min_height": 1.5,
    "shelf_max_width": 0.6,
    "pallet_max_height": 0.3,
    "pallet_min_width": 0.8,
}


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_cloud(
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic point cloud with a ground plane and random clusters.

    The cloud contains:
    - ~40% ground-plane points (z near 0, spread over a 20x20 m area)
    - ~60% divided among 5-12 random obstacle clusters with varying sizes

    Parameters
    ----------
    num_points : int
        Total number of points to generate.
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_points, 3)`` with dtype ``float64``.
    """
    ground_count = int(num_points * 0.4)
    cluster_count = num_points - ground_count

    # Ground plane: z ~ N(0, 0.02), xy uniform in [-10, 10].
    ground_xy = rng.uniform(-10.0, 10.0, size=(ground_count, 2))
    ground_z = rng.normal(0.0, 0.02, size=(ground_count, 1))
    ground = np.hstack([ground_xy, ground_z])

    # Random clusters.
    num_clusters = rng.integers(5, 13)
    pts_per_cluster = cluster_count // num_clusters
    clusters: List[np.ndarray] = []

    for _ in range(num_clusters):
        cx = rng.uniform(-8.0, 8.0)
        cy = rng.uniform(-8.0, 8.0)
        cz = rng.uniform(0.3, 2.0)
        sigma = rng.uniform(0.05, 0.3)
        n_pts = pts_per_cluster + rng.integers(-10, 10)
        n_pts = max(n_pts, 10)
        cluster = rng.normal(
            loc=[cx, cy, cz], scale=sigma, size=(n_pts, 3)
        )
        clusters.append(cluster)

    cluster_pts = np.vstack(clusters) if clusters else np.empty((0, 3))

    # Combine and trim / pad to exact num_points.
    all_pts = np.vstack([ground, cluster_pts])
    if len(all_pts) >= num_points:
        all_pts = all_pts[:num_points]
    else:
        # Pad with extra ground points if we're short.
        deficit = num_points - len(all_pts)
        extra_xy = rng.uniform(-10.0, 10.0, size=(deficit, 2))
        extra_z = rng.normal(0.0, 0.02, size=(deficit, 1))
        all_pts = np.vstack([all_pts, np.hstack([extra_xy, extra_z])])

    return all_pts


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def benchmark_clustering(
    points: np.ndarray,
    o3d,
    eps: float = 0.3,
    min_points: int = 10,
) -> Tuple[np.ndarray, float]:
    """Run DBSCAN clustering and return labels + elapsed time in seconds."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    t0 = time.perf_counter()
    labels = np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    elapsed = time.perf_counter() - t0
    return labels, elapsed


def benchmark_classification(
    points: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[str, int], float]:
    """Classify each cluster by bounding-box geometry; return counts + elapsed."""
    unique_labels = set(labels)
    unique_labels.discard(-1)

    counts: Dict[str, int] = {}
    t0 = time.perf_counter()

    for label in unique_labels:
        mask = labels == label
        cluster = points[mask]
        mins = cluster.min(axis=0)
        maxs = cluster.max(axis=0)
        size = maxs - mins
        width, depth, height = float(size[0]), float(size[1]), float(size[2])

        cls = classify_cluster(width, depth, height, DEFAULT_PARAMS)
        counts[cls] = counts.get(cls, 0) + 1

    elapsed = time.perf_counter() - t0
    return counts, elapsed


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(num_points: int, iterations: int) -> None:
    """Generate synthetic data, run the perception pipeline, and print stats."""
    try:
        import open3d as o3d
    except ImportError:
        print("ERROR: Open3D is required.  Install with: pip install open3d")
        sys.exit(1)

    print(f"Open3D version: {o3d.__version__}")
    print(f"Benchmark config: {num_points:,} points x {iterations} iterations")
    print("-" * 70)

    rng = np.random.default_rng(seed=2025)

    clustering_times: List[float] = []
    classification_times: List[float] = []
    total_times: List[float] = []
    cluster_counts: List[int] = []

    for i in range(iterations):
        cloud = generate_synthetic_cloud(num_points, rng)

        t_total_start = time.perf_counter()

        labels, t_cluster = benchmark_clustering(cloud, o3d)
        counts, t_classify = benchmark_classification(cloud, labels)

        t_total = time.perf_counter() - t_total_start

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        clustering_times.append(t_cluster)
        classification_times.append(t_classify)
        total_times.append(t_total)
        cluster_counts.append(n_clusters)

        # Progress indicator every 10 iterations.
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1:>{len(str(iterations))}}/{iterations}]  "
                  f"clusters={n_clusters:>3}  "
                  f"total={t_total * 1000:.1f} ms")

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    def _stats(values: List[float]) -> Tuple[float, float, float, float]:
        a = np.array(values) * 1000  # convert to ms
        return float(np.mean(a)), float(np.std(a)), float(np.min(a)), float(np.max(a))

    print()
    print("=" * 70)
    print(f"  PERCEPTION PIPELINE BENCHMARK  ({num_points:,} pts, {iterations} iters)")
    print("=" * 70)
    print()
    print(f"  {'Stage':<22} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for label, values in [
        ("DBSCAN clustering", clustering_times),
        ("Classification", classification_times),
        ("Total pipeline", total_times),
    ]:
        mean, std, vmin, vmax = _stats(values)
        print(f"  {label:<22} {mean:>10.2f} {std:>10.2f} {vmin:>10.2f} {vmax:>10.2f}")

    avg_clusters = float(np.mean(cluster_counts))
    print()
    print(f"  Average clusters per frame: {avg_clusters:.1f}")
    print(f"  Throughput: {1000.0 / np.mean(np.array(total_times) * 1000):.1f} frames/sec")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the obstacle detector perception pipeline "
        "using synthetic point clouds."
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="Number of points per synthetic cloud (default: 50000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    args = parser.parse_args()
    run_benchmark(args.num_points, args.iterations)


if __name__ == "__main__":
    main()
