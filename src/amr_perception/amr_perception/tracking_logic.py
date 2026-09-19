"""
Multi-object tracking logic, independent of ROS.

Greedy nearest-neighbour association of detections to tracks, with
exponential-moving-average smoothing of position and velocity.  Operates on
plain Python types so it can be unit tested without an rclpy runtime;
``object_tracker_node`` adapts ROS messages to and from these types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    """One detection handed to the tracker."""

    x: float
    y: float
    z: float
    class_name: str = 'unknown'
    confidence: float = 0.0


@dataclass
class Track:
    """State of one tracked object."""

    id: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    class_name: str = 'unknown'
    confidence: float = 0.0
    hits: int = 1
    lost_count: int = 0
    last_seen_time: float = 0.0  # seconds; always set by Tracker at creation

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)


class Tracker:
    """Persistent-ID tracker.

    A new track is created for every unmatched observation.  A track is
    *confirmed* once it has accumulated ``min_hits`` observations, and is
    dropped after more than ``max_lost`` consecutive frames without a match.
    Matched tracks are updated with an exponential moving average
    (``alpha`` weights the new observation) on both position and velocity.
    """

    def __init__(
        self,
        association_threshold: float,
        max_lost: int,
        min_hits: int,
        alpha: float,
    ) -> None:
        self.association_threshold = association_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.alpha = alpha
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ #

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]

    def update(self, observations: Sequence[Observation], now: float) -> List[int]:
        """Ingest one frame of observations taken at time ``now`` (seconds).

        Returns the IDs of tracks removed in this frame.
        """
        matched, unmatched_obs, unmatched_tracks = self.associate(observations)

        for obs_idx, track_id in matched:
            self._update_track(self.tracks[track_id], observations[obs_idx], now)

        for obs_idx in unmatched_obs:
            obs = observations[obs_idx]
            self.tracks[self._next_id] = Track(
                id=self._next_id,
                x=obs.x, y=obs.y, z=obs.z,
                class_name=obs.class_name,
                confidence=obs.confidence,
                last_seen_time=now,
            )
            self._next_id += 1

        removed: List[int] = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.lost_count += 1
            if track.lost_count > self.max_lost:
                removed.append(track_id)
        for track_id in removed:
            del self.tracks[track_id]
        return removed

    def _update_track(self, track: Track, obs: Observation, now: float) -> None:
        dt = now - track.last_seen_time
        if dt > 1e-6:
            inst_v = ((obs.x - track.x) / dt, (obs.y - track.y) / dt, (obs.z - track.z) / dt)
        else:
            inst_v = (0.0, 0.0, 0.0)

        a = self.alpha
        track.vx = a * inst_v[0] + (1.0 - a) * track.vx
        track.vy = a * inst_v[1] + (1.0 - a) * track.vy
        track.vz = a * inst_v[2] + (1.0 - a) * track.vz

        track.x = a * obs.x + (1.0 - a) * track.x
        track.y = a * obs.y + (1.0 - a) * track.y
        track.z = a * obs.z + (1.0 - a) * track.z

        track.class_name = obs.class_name
        track.confidence = obs.confidence
        track.hits += 1
        track.lost_count = 0
        track.last_seen_time = now

    # ------------------------------------------------------------------ #

    def associate(
        self, observations: Sequence[Observation]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy nearest-neighbour association of observations to existing tracks.

        Returns (matched (obs_idx, track_id) pairs, unmatched observation
        indices, unmatched track IDs).
        """
        track_ids = list(self.tracks.keys())
        n_obs, n_trk = len(observations), len(track_ids)
        if n_obs == 0:
            return [], [], track_ids
        if n_trk == 0:
            return [], list(range(n_obs)), []

        cost = np.empty((n_obs, n_trk), dtype=np.float64)
        for i, obs in enumerate(observations):
            for j, tid in enumerate(track_ids):
                t = self.tracks[tid]
                cost[i, j] = math.sqrt((obs.x - t.x) ** 2 + (obs.y - t.y) ** 2 + (obs.z - t.z) ** 2)

        matched: List[Tuple[int, int]] = []
        used_obs: set = set()
        used_trk: set = set()
        for flat_idx in np.argsort(cost, axis=None):
            oi = int(flat_idx // n_trk)
            tj = int(flat_idx % n_trk)
            if oi in used_obs or tj in used_trk:
                continue
            if cost[oi, tj] > self.association_threshold:
                break
            matched.append((oi, track_ids[tj]))
            used_obs.add(oi)
            used_trk.add(tj)

        unmatched_obs = [i for i in range(n_obs) if i not in used_obs]
        unmatched_trk = [track_ids[j] for j in range(n_trk) if j not in used_trk]
        return matched, unmatched_obs, unmatched_trk
