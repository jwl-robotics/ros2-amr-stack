# Copyright 2025 AMR Stack Authors
# Licensed under MIT
#
# Unit tests for amr_perception.tracking_logic -- the Tracker the
# object_tracker node runs.  No ROS runtime required.

import pytest

from amr_perception.tracking_logic import Observation, Tracker


def make_tracker(association_threshold=1.5, max_lost=5, min_hits=3, alpha=0.3):
    return Tracker(association_threshold, max_lost, min_hits, alpha)


def only_track(tracker):
    assert len(tracker.tracks) == 1
    return next(iter(tracker.tracks.values()))


# ===================================================================
# Creation
# ===================================================================


class TestTrackCreation:

    def test_first_observation_creates_track(self):
        tracker = make_tracker(min_hits=1)
        tracker.update([Observation(1.0, 2.0, 0.0, "person", 0.8)], now=0.0)

        track = only_track(tracker)
        assert (track.x, track.y, track.z) == (1.0, 2.0, 0.0)
        assert track.class_name == "person"
        assert track.confidence == 0.8
        assert track.hits == 1

    def test_multiple_observations_create_separate_tracks(self):
        tracker = make_tracker(min_hits=1)
        tracker.update([
            Observation(0.0, 0.0, 0.0, "person"),
            Observation(5.0, 5.0, 0.0, "shelf"),
            Observation(10.0, 10.0, 0.0, "pallet"),
        ], now=0.0)

        assert len(tracker.tracks) == 3
        assert {t.class_name for t in tracker.tracks.values()} == {"person", "shelf", "pallet"}
        assert sorted(tracker.tracks) == [0, 1, 2]

    def test_ids_are_never_reused(self):
        tracker = make_tracker(min_hits=1, max_lost=0)
        tracker.update([Observation(0.0, 0.0, 0.0)], now=0.0)
        tracker.update([], now=0.1)  # track 0 dropped
        tracker.update([Observation(0.0, 0.0, 0.0)], now=0.2)
        assert list(tracker.tracks) == [1]


# ===================================================================
# Association and update
# ===================================================================


class TestTrackUpdate:

    def test_nearby_observation_updates_existing_track(self):
        tracker = make_tracker(min_hits=1)
        tracker.update([Observation(1.0, 1.0, 0.0)], now=0.0)
        tid = only_track(tracker).id

        tracker.update([Observation(1.1, 1.1, 0.0)], now=0.1)
        track = only_track(tracker)
        assert track.id == tid
        assert track.hits == 2

    def test_position_is_smoothed_with_alpha(self):
        tracker = make_tracker(min_hits=1, alpha=0.3)
        tracker.update([Observation(1.0, 1.0, 0.0)], now=0.0)
        tracker.update([Observation(2.0, 1.0, 0.0)], now=0.1)

        track = only_track(tracker)
        assert track.x == pytest.approx(0.3 * 2.0 + 0.7 * 1.0)
        assert track.y == pytest.approx(1.0)

    def test_alpha_one_follows_observation_exactly(self):
        tracker = make_tracker(min_hits=1, alpha=1.0, association_threshold=5.0)
        tracker.update([Observation(1.0, 1.0, 0.0)], now=0.0)
        tracker.update([Observation(2.0, 3.0, 0.5)], now=0.1)
        track = only_track(tracker)
        assert (track.x, track.y, track.z) == pytest.approx((2.0, 3.0, 0.5))

    def test_far_observation_creates_new_track(self):
        tracker = make_tracker(association_threshold=1.0, min_hits=1)
        tracker.update([Observation(0.0, 0.0, 0.0)], now=0.0)
        tracker.update([Observation(5.0, 0.0, 0.0)], now=0.1)
        assert len(tracker.tracks) == 2

    def test_label_follows_latest_observation(self):
        tracker = make_tracker(min_hits=1)
        tracker.update([Observation(0.0, 0.0, 0.0, "obstacle", 0.5)], now=0.0)
        tracker.update([Observation(0.0, 0.0, 0.0, "person", 0.9)], now=0.1)
        track = only_track(tracker)
        assert (track.class_name, track.confidence) == ("person", 0.9)


# ===================================================================
# Lifecycle
# ===================================================================


class TestTrackLifecycle:

    def test_track_dropped_after_more_than_max_lost_misses(self):
        tracker = make_tracker(max_lost=2, min_hits=1)
        tracker.update([Observation(1.0, 1.0, 0.0)], now=0.0)

        tracker.update([], now=0.1)
        tracker.update([], now=0.2)
        assert len(tracker.tracks) == 1  # lost_count == max_lost: still alive

        removed = tracker.update([], now=0.3)
        assert removed == [0]
        assert tracker.tracks == {}

    def test_match_resets_lost_count(self):
        tracker = make_tracker(max_lost=2, min_hits=1)
        tracker.update([Observation(1.0, 1.0, 0.0)], now=0.0)
        tracker.update([], now=0.1)
        tracker.update([], now=0.2)
        tracker.update([Observation(1.0, 1.0, 0.0)], now=0.3)
        assert only_track(tracker).lost_count == 0

    def test_confirmation_requires_min_hits(self):
        tracker = make_tracker(min_hits=3, association_threshold=2.0)

        tracker.update([Observation(1.0, 1.0, 0.0)], now=0.0)
        assert tracker.confirmed() == []

        tracker.update([Observation(1.05, 1.05, 0.0)], now=0.1)
        assert tracker.confirmed() == []

        tracker.update([Observation(1.1, 1.1, 0.0)], now=0.2)
        assert [t.id for t in tracker.confirmed()] == [0]


# ===================================================================
# Velocity
# ===================================================================


class TestVelocityEstimation:

    def test_velocity_is_ema_of_instantaneous_velocity(self):
        tracker = make_tracker(min_hits=1, association_threshold=5.0, alpha=0.3)
        tracker.update([Observation(0.0, 0.0, 0.0)], now=0.0)
        # +1.0 m in x and +0.5 m in y over 0.1 s: instantaneous (10, 5, 0) m/s,
        # blended 0.3 into the initial zero velocity.
        tracker.update([Observation(1.0, 0.5, 0.0)], now=0.1)

        track = only_track(tracker)
        assert track.vx == pytest.approx(3.0)
        assert track.vy == pytest.approx(1.5)
        assert track.vz == pytest.approx(0.0)
        assert track.speed == pytest.approx((3.0 ** 2 + 1.5 ** 2) ** 0.5)

    def test_alpha_one_gives_instantaneous_velocity(self):
        tracker = make_tracker(min_hits=1, association_threshold=5.0, alpha=1.0)
        tracker.update([Observation(0.0, 0.0, 0.0)], now=0.0)
        tracker.update([Observation(1.0, 0.5, 0.0)], now=0.1)
        track = only_track(tracker)
        assert (track.vx, track.vy, track.vz) == pytest.approx((10.0, 5.0, 0.0))

    def test_zero_dt_does_not_divide(self):
        tracker = make_tracker(min_hits=1, alpha=1.0)
        tracker.update([Observation(0.0, 0.0, 0.0)], now=1.0)
        tracker.update([Observation(1.0, 0.0, 0.0)], now=1.0)
        assert only_track(tracker).vx == 0.0
