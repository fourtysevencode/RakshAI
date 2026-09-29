from backend.models.schemas import FrameSample, PPEObservation, TrackObservation
from backend.services.violations import ViolationTracker

DT = 0.2  # 5 samples per second


def sample(t, hardhat_state, track_id=1, near=()):
    ppe = {
        "hardhat": PPEObservation(hardhat_state, 0.8 if hardhat_state != "unseen" else 0.0, [0, 0, 1, 1]),
        "vest": PPEObservation("present", 0.9),
        "mask": PPEObservation("present", 0.9),
    }
    obs = TrackObservation(track_id=track_id, bbox=[0, 0, 10, 30], confidence=0.9, ppe=ppe, near=list(near))
    return FrameSample(frame_idx=int(t / DT), t=t, dt=DT, tracks=[obs], counts={}, unattributed=[], detections=[])


def run(states, **kw):
    vt = ViolationTracker(["hardhat"], **kw)
    for i, s in enumerate(states):
        vt.feed(sample(round(i * DT, 3), s))
    vt.finalize()
    return vt.events


def test_short_lapse_below_persistence_is_not_an_event():
    # missing for 0.8 s (samples at 0.0..0.6), then present
    assert run(["missing"] * 4 + ["present"] * 10, min_violation_s=1.0) == []


def test_sustained_missing_becomes_one_event_with_duration():
    events = run(["present"] * 2 + ["missing"] * 10 + ["present"] * 15, min_violation_s=1.0, cooldown_s=2.0)
    assert len(events) == 1
    e = events[0]
    assert e.type == "NO_HARDHAT" and e.track_id == 1
    assert abs(e.t_start - 0.4) < 1e-6
    assert abs(e.t_end - (0.4 + 9 * DT + DT)) < 1e-6  # last missing sample + its dt
    assert e.samples == 10


def test_flicker_and_occlusion_do_not_split_an_event():
    states = ["missing"] * 8 + ["present"] + ["missing"] * 3 + ["unseen"] * 4 + ["missing"] * 5
    events = run(states, min_violation_s=1.0, cooldown_s=2.0)
    assert len(events) == 1
    assert events[0].samples == 16


def test_gap_longer_than_cooldown_starts_a_new_event():
    states = ["missing"] * 8 + ["present"] * 12 + ["missing"] * 8  # 2.4 s of present in between
    events = run(states, min_violation_s=1.0, cooldown_s=2.0)
    assert len(events) == 2


def test_track_reappearing_after_long_absence_is_a_new_event():
    vt = ViolationTracker(["hardhat"], min_violation_s=0.4, cooldown_s=1.0)
    for i in range(5):
        vt.feed(sample(i * DT, "missing"))
    for i in range(5):  # same track back 10 s later
        vt.feed(sample(10 + i * DT, "missing"))
    vt.finalize()
    assert len(vt.events) == 2


def test_items_not_required_never_create_events():
    vt = ViolationTracker(["vest"])
    for i in range(20):
        vt.feed(sample(i * DT, "missing"))
    vt.finalize()
    assert vt.events == []


def test_near_hazard_flag_and_evidence_track_best_observation():
    vt = ViolationTracker(["hardhat"], min_violation_s=0.0)
    vt.feed(sample(0.0, "missing"))
    improved, _ = vt.feed(sample(0.2, "missing", near=("machinery",)))
    vt.finalize()
    assert vt.events[0].near_hazard
    assert vt.events[0].evidence["t"] == 0.0  # equal confidence: first observation kept
    assert improved == []


def test_discarded_runs_are_reported():
    vt = ViolationTracker(["hardhat"], min_violation_s=1.0)
    vt.feed(sample(0.0, "missing"))
    _, discarded = vt.feed(sample(0.2, "present"))
    assert discarded == ["ev0001"]
