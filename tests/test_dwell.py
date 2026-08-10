"""S6.2 — dwell and debounce on synthetic tracks, where the right answer is known exactly.

Every case here is a scripted sequence of frames: a worker stands in the zone without a
helmet, complies, leaves, flickers out of detection, comes back. Because the module takes
timestamps and states rather than reading a clock or a camera, these sequences pin the
behaviour completely — which is why S6.5 evidences the alerting layer by this file instead of
by a hand-labelled clip.
"""

from __future__ import annotations

from src.associate import HELMET, VEST
from src.dwell import DwellTracker, PPEMemory

MISSING_HELMET = (HELMET,)
MISSING_BOTH = (HELMET, VEST)


def test_zero_dwell_fires_on_the_first_observation():
    tracker = DwellTracker(dwell_seconds=0.0, zone="loading-bay")
    alerts = tracker.update(0.0, {7: MISSING_HELMET})
    assert len(alerts) == 1
    assert alerts[0].track_id == 7
    assert alerts[0].zone == "loading-bay"
    assert alerts[0].missing == MISSING_HELMET
    assert alerts[0].dwell == 0.0


def test_one_alert_per_incident_not_one_per_frame():
    """The defect the S2 skeleton hit: one worker, one incident, one row."""
    tracker = DwellTracker(dwell_seconds=0.0)
    fired = [len(tracker.update(t / 10, {7: MISSING_HELMET})) for t in range(50)]
    assert sum(fired) == 1
    assert fired[0] == 1


def test_dwell_threshold_delays_the_alert_until_it_is_met():
    tracker = DwellTracker(dwell_seconds=3.0)
    assert tracker.update(0.0, {7: MISSING_HELMET}) == []
    assert tracker.update(1.5, {7: MISSING_HELMET}) == []
    assert tracker.update(2.9, {7: MISSING_HELMET}) == []
    alerts = tracker.update(3.0, {7: MISSING_HELMET})
    assert len(alerts) == 1
    assert alerts[0].dwell == 3.0


def test_a_passer_by_never_reaches_the_threshold():
    """Someone crossing the corner of the zone for two seconds is not an incident."""
    tracker = DwellTracker(dwell_seconds=5.0)
    for t in range(20):
        assert tracker.update(t / 10, {7: MISSING_HELMET}) == []
    for t in range(20, 100):  # they have left; nobody is violating
        assert tracker.update(t / 10, {}) == []
    assert tracker.open_incidents == 0


def test_compliance_ends_the_incident_and_re_offending_alerts_again():
    tracker = DwellTracker(dwell_seconds=0.0, grace_seconds=2.0)
    assert len(tracker.update(0.0, {7: MISSING_HELMET})) == 1

    for t in (1.0, 2.0, 3.0, 4.0):  # helmet on: not reported as violating
        assert tracker.update(t, {}) == []
    assert tracker.open_incidents == 0

    assert len(tracker.update(5.0, {7: MISSING_HELMET})) == 1  # helmet off again


def test_resolve_ends_an_incident_immediately():
    """When the caller knows the worker complied, the incident should not linger."""
    tracker = DwellTracker(dwell_seconds=0.0, grace_seconds=10.0)
    tracker.update(0.0, {7: MISSING_HELMET})
    assert tracker.open_incidents == 1
    tracker.resolve(7)
    assert tracker.open_incidents == 0
    assert len(tracker.update(0.5, {7: MISSING_HELMET})) == 1  # a fresh incident, fresh alert


def test_a_dropped_detection_does_not_restart_the_timer():
    """F27: the person detector loses people. A one-frame gap must not reset the dwell."""
    tracker = DwellTracker(dwell_seconds=3.0, grace_seconds=2.0)
    tracker.update(0.0, {7: MISSING_HELMET})
    tracker.update(1.0, {})  # missed frame
    tracker.update(2.0, {7: MISSING_HELMET})
    tracker.update(2.5, {})  # missed again
    alerts = tracker.update(3.0, {7: MISSING_HELMET})
    assert len(alerts) == 1
    assert alerts[0].started_at == 0.0  # the timer bridged both gaps rather than restarting


def test_a_gap_beyond_the_grace_window_starts_a_new_incident():
    tracker = DwellTracker(dwell_seconds=3.0, grace_seconds=2.0)
    tracker.update(0.0, {7: MISSING_HELMET})
    assert tracker.update(10.0, {7: MISSING_HELMET}) == []  # gone and back: timer restarts
    assert tracker.update(11.0, {7: MISSING_HELMET}) == []
    assert tracker.update(12.0, {7: MISSING_HELMET}) == []
    alerts = tracker.update(13.0, {7: MISSING_HELMET})
    assert len(alerts) == 1
    assert alerts[0].started_at == 10.0  # timed from the return, not from the first sighting


def test_each_person_is_timed_independently():
    tracker = DwellTracker(dwell_seconds=2.0)
    tracker.update(0.0, {7: MISSING_HELMET})
    tracker.update(1.0, {7: MISSING_HELMET, 9: MISSING_BOTH})
    alerts = tracker.update(2.0, {7: MISSING_HELMET, 9: MISSING_BOTH})
    assert [a.track_id for a in alerts] == [7]  # 9 has only been violating for a second
    alerts = tracker.update(3.0, {7: MISSING_HELMET, 9: MISSING_BOTH})
    assert [a.track_id for a in alerts] == [9]


def test_the_alert_reports_what_was_missing_when_it_fired():
    """A worker who loses their vest mid-incident should be alerted for both items."""
    tracker = DwellTracker(dwell_seconds=2.0)
    tracker.update(0.0, {7: MISSING_HELMET})
    alerts = tracker.update(2.0, {7: MISSING_BOTH})
    assert alerts[0].missing == MISSING_BOTH
    assert alerts[0].missing_names == "helmet+vest"


def test_missing_names_reads_as_the_log_row_will():
    tracker = DwellTracker(dwell_seconds=0.0)
    alert = tracker.update(0.0, {7: (VEST,)})[0]
    assert alert.missing_names == "vest"


# ------------------------------------------------------- PPE memory (the flicker fix)

REQUIRED = (HELMET, VEST)


def test_a_helmet_missed_for_one_frame_does_not_undress_the_worker():
    """The defect a viewer sees first: the same person flipping compliant/violating."""
    memory = PPEMemory(seconds=1.0)
    memory.observe(0.0, 7, REQUIRED)  # seen with both
    memory.observe(0.04, 7, (VEST,))  # helmet missed on this frame
    assert memory.missing(0.04, 7, REQUIRED) == ()


def test_ppe_is_forgotten_once_the_memory_expires():
    """Believing it forever would hide a worker who really did take the helmet off."""
    memory = PPEMemory(seconds=1.0)
    memory.observe(0.0, 7, REQUIRED)
    memory.observe(1.5, 7, (VEST,))
    assert memory.missing(1.5, 7, REQUIRED) == (HELMET,)


def test_memory_is_per_person():
    memory = PPEMemory(seconds=1.0)
    memory.observe(0.0, 7, REQUIRED)
    memory.observe(0.0, 9, ())
    assert memory.missing(0.0, 7, REQUIRED) == ()
    assert memory.missing(0.0, 9, REQUIRED) == REQUIRED


def test_a_worker_never_seen_wearing_anything_is_missing_everything():
    memory = PPEMemory(seconds=1.0)
    assert memory.missing(0.0, 7, REQUIRED) == REQUIRED


def test_zero_seconds_disables_the_memory():
    """The evaluation path judges each frame alone, so the switch has to be real."""
    memory = PPEMemory(seconds=0.0)
    memory.observe(0.0, 7, REQUIRED)
    assert memory.missing(0.04, 7, REQUIRED) == REQUIRED


def test_forget_drops_stale_tracks():
    memory = PPEMemory(seconds=1.0)
    memory.observe(0.0, 7, REQUIRED)
    memory.observe(10.0, 9, REQUIRED)
    memory.forget(before=5.0)
    assert memory.missing(10.0, 9, REQUIRED) == ()
    assert memory.missing(10.0, 7, REQUIRED) == REQUIRED
