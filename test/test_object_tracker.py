# Copyright 2025 AMR Stack Authors
# Licensed under MIT
#
# Unit tests for the object tracker logic.
# These tests exercise a standalone SimpleTracker class that encapsulates the
# same association-and-update algorithm the real object_tracker_node uses,
# without any ROS dependency.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal data structures
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """Lightweight detection fed into the tracker."""

    x: float
    y: float
    z: float
    class_name: str = "obstacle"
    confidence: float = 1.0


@dataclass
class Track:
    """Internal state for a single object track."""

    track_id: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    class_name: str = "obstacle"
    hits: int = 1
    lost_frames: int = 0
    confirmed: bool = False


# ---------------------------------------------------------------------------
# Tracker logic (mirrors the real node's algorithm)
# ---------------------------------------------------------------------------


class SimpleTracker:
    """Pure-Python multi-object tracker (no ROS dependencies).

    Uses greedy nearest-neighbour association and a simple constant-velocity
    model for prediction.  Tracks must accumulate ``min_hits`` consecutive
    updates before they are promoted to *confirmed*.  Tracks that are not
    updated for ``max_lost_frames`` consecutive frames are removed.
    """

    def __init__(
        self,
        distance_threshold: float = 1.5,
        max_lost_frames: int = 5,
        min_hits: int = 3,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self._next_id: int = 0
        self.tracks: Dict[int, Track] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(self, detections: List[Detection], dt: float = 0.1) -> List[Track]:
        """Process a new frame of detections.

        Parameters
        ----------
        detections : list of Detection
            Current-frame detections.
        dt : float
            Time delta since the previous frame (seconds).  Used for
            velocity estimation.

        Returns
        -------
        list of Track
            All currently *confirmed* tracks after the update.
        """
        # 1. Predict: advance each track by its velocity.
        for t in self.tracks.values():
            t.x += t.vx * dt
            t.y += t.vy * dt
            t.z += t.vz * dt

        # 2. Associate detections to existing tracks (greedy nearest-neighbour).
        matched, unmatched_det_idxs = self._associate(detections)

        # 3. Update matched tracks.
        for track_id, det_idx in matched:
            det = detections[det_idx]
            t = self.tracks[track_id]

            # Velocity estimation from positional difference.
            if dt > 0:
                t.vx = (det.x - t.x) / dt
                t.vy = (det.y - t.y) / dt
                t.vz = (det.z - t.z) / dt

            t.x = det.x
            t.y = det.y
            t.z = det.z
            t.class_name = det.class_name
            t.hits += 1
            t.lost_frames = 0

            if t.hits >= self.min_hits:
                t.confirmed = True

        # 4. Create new tracks from unmatched detections.
        for idx in unmatched_det_idxs:
            det = detections[idx]
            self.tracks[self._next_id] = Track(
                track_id=self._next_id,
                x=det.x,
                y=det.y,
                z=det.z,
                class_name=det.class_name,
            )
            self._next_id += 1

        # 5. Increment lost_frames for tracks that were not matched.
        matched_track_ids = {tid for tid, _ in matched}
        for tid, t in list(self.tracks.items()):
            if tid not in matched_track_ids and tid < self._next_id - len(unmatched_det_idxs):
                t.lost_frames += 1

        # 6. Prune dead tracks.
        dead = [tid for tid, t in self.tracks.items() if t.lost_frames > self.max_lost_frames]
        for tid in dead:
            del self.tracks[tid]

        return [t for t in self.tracks.values() if t.confirmed]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _associate(
        self, detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Greedy nearest-neighbour matching.

        Returns (matched_pairs, unmatched_detection_indices).
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections)))

        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        dist = np.zeros((n_tracks, n_dets))
        for i, tid in enumerate(track_ids):
            t = self.tracks[tid]
            for j, d in enumerate(detections):
                dist[i, j] = np.sqrt((t.x - d.x) ** 2 + (t.y - d.y) ** 2 + (t.z - d.z) ** 2)

        matched: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()

        for flat_idx in np.argsort(dist, axis=None):
            ti = int(flat_idx // n_dets)
            di = int(flat_idx % n_dets)
            if ti in used_tracks or di in used_dets:
                continue
            if dist[ti, di] > self.distance_threshold:
                break
            matched.append((track_ids[ti], di))
            used_tracks.add(ti)
            used_dets.add(di)

        unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
        return matched, unmatched_dets


# ===================================================================
# Tests
# ===================================================================


class TestTrackCreation:
    """Tests for new track creation."""

    def test_track_creation(self):
        """A new detection with no existing tracks should create a new track."""
        tracker = SimpleTracker(min_hits=1)
        dets = [Detection(x=1.0, y=2.0, z=0.0, class_name="person")]
        confirmed = tracker.update(dets)

        assert len(tracker.tracks) == 1
        track = list(tracker.tracks.values())[0]
        assert track.x == 1.0
        assert track.y == 2.0
        assert track.class_name == "person"

    def test_multiple_tracks(self):
        """Multiple detections in the first frame should create separate tracks."""
        tracker = SimpleTracker(min_hits=1)
        dets = [
            Detection(x=0.0, y=0.0, z=0.0, class_name="person"),
            Detection(x=5.0, y=5.0, z=0.0, class_name="shelf"),
            Detection(x=10.0, y=10.0, z=0.0, class_name="pallet"),
        ]
        tracker.update(dets)
        assert len(tracker.tracks) == 3

        classes = {t.class_name for t in tracker.tracks.values()}
        assert classes == {"person", "shelf", "pallet"}


class TestTrackAssociation:
    """Tests for detection-to-track association."""

    def test_track_association(self):
        """A detection near an existing track should update that track."""
        tracker = SimpleTracker(distance_threshold=1.5, min_hits=1)

        # Frame 1: create a track.
        tracker.update([Detection(x=1.0, y=1.0, z=0.0)])
        assert len(tracker.tracks) == 1
        tid = list(tracker.tracks.keys())[0]

        # Frame 2: detection slightly moved -- should associate.
        tracker.update([Detection(x=1.1, y=1.1, z=0.0)])
        assert len(tracker.tracks) == 1
        assert list(tracker.tracks.keys())[0] == tid
        assert tracker.tracks[tid].hits == 2


class TestTrackLifecycle:
    """Tests for track loss and removal."""

    def test_track_lost(self):
        """A track not updated for max_lost_frames should be removed."""
        tracker = SimpleTracker(max_lost_frames=2, min_hits=1)

        # Frame 1: create a track.
        tracker.update([Detection(x=1.0, y=1.0, z=0.0)])
        assert len(tracker.tracks) == 1

        # Frames 2-4: no detections -- track should eventually be pruned.
        for _ in range(4):
            tracker.update([])

        assert len(tracker.tracks) == 0

    def test_track_confirmation(self):
        """A track must reach min_hits before it is considered confirmed."""
        tracker = SimpleTracker(min_hits=3, distance_threshold=2.0)

        det = Detection(x=1.0, y=1.0, z=0.0)

        # Frame 1: create track (hits=1) -- not yet confirmed.
        confirmed = tracker.update([det])
        assert len(confirmed) == 0

        # Frame 2: update (hits=2) -- still not confirmed.
        confirmed = tracker.update([Detection(x=1.05, y=1.05, z=0.0)])
        assert len(confirmed) == 0

        # Frame 3: update (hits=3) -- now confirmed.
        confirmed = tracker.update([Detection(x=1.1, y=1.1, z=0.0)])
        assert len(confirmed) == 1
        assert confirmed[0].confirmed is True


class TestVelocityEstimation:
    """Tests for velocity estimation from sequential positions."""

    def test_velocity_estimation(self):
        """Two sequential positions with known dt should produce correct velocity."""
        tracker = SimpleTracker(min_hits=1, distance_threshold=5.0)
        dt = 0.1  # 100 ms between frames

        # Frame 1.
        tracker.update([Detection(x=0.0, y=0.0, z=0.0)], dt=dt)
        tid = list(tracker.tracks.keys())[0]

        # Frame 2: moved +1.0 in x, +0.5 in y.
        tracker.update([Detection(x=1.0, y=0.5, z=0.0)], dt=dt)

        track = tracker.tracks[tid]
        # Expected vx = (1.0 - predicted_x) / dt.  After prediction, predicted
        # x = 0.0 + 0.0*0.1 = 0.0 (no velocity yet).  So vx = 1.0/0.1 = 10.0.
        assert abs(track.vx - 10.0) < 1e-3
        assert abs(track.vy - 5.0) < 1e-3
        assert abs(track.vz) < 1e-3
