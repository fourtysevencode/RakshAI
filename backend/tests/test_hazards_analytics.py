from backend.models.schemas import Detection, FrameSample, PPEObservation, TrackObservation
from backend.services.analytics import build_results
from backend.services.hazards import box_gap, nearby_hazards
from backend.services.violations import ViolationTracker


def test_box_gap():
    assert box_gap([0, 0, 10, 10], [5, 5, 20, 20]) == 0
    assert box_gap([0, 0, 10, 10], [13, 0, 20, 10]) == 3
    assert box_gap([0, 0, 10, 10], [13, 14, 20, 20]) == 5  # 3-4-5 triangle


def test_nearby_hazards_scales_with_person_height():
    person = [100, 100, 140, 300]  # 200 px tall -> near within 100 px
    close = Detection(8, "machinery", 0.8, [230, 100, 400, 300])  # 90 px gap
    far = Detection(9, "vehicle", 0.8, [260, 100, 400, 300])  # 120 px gap
    cone = Detection(6, "Safety Cone", 0.8, [140, 280, 150, 300])
    assert nearby_hazards(person, [close, far, cone]) == ["machinery"]


def _obs(tid, hardhat, conf=0.9):
    ppe = {"hardhat": PPEObservation(hardhat, 0.8 if hardhat != "unseen" else 0.0),
           "vest": PPEObservation("unseen"), "mask": PPEObservation("unseen")}
    return TrackObservation(track_id=tid, bbox=[0, 0, 64, 144], confidence=conf, ppe=ppe)


def test_build_results_per_worker_summary():
    dt = 0.2
    samples = []
    for i in range(50):  # 10 s
        t = round(i * dt, 3)
        tracks = [_obs(1, "missing" if 10 <= i < 30 else "present")]
        if i < 2:
            tracks.append(_obs(7, "present", conf=0.3))  # brief phantom track
        if i >= 25:
            tracks.append(_obs(3, "unseen"))
        samples.append(FrameSample(i, t, dt, tracks, {"persons": len(tracks)}, [], []))
    vt = ViolationTracker(["hardhat"], min_violation_s=1.0, cooldown_s=2.0)
    for s in samples:
        vt.feed(s)
    vt.finalize()
    r = build_results(samples, vt.events, 10.0, (640, 360), ["hardhat"])

    s = r["summary"]
    assert s["workers_observed"] == 2 and s["brief_tracks"] == 1
    assert s["workers_non_compliant"] == 1 and s["workers_unconfirmed"] == 1
    assert s["total_events"] == 1
    w1 = next(p for p in r["persons"] if p["track_id"] == 1)
    assert w1["worker_id"] == 1 and w1["status"] == "non_compliant"
    assert abs(w1["ppe"]["hardhat"]["compliance"] - 0.6) < 1e-6
    assert w1["ppe"]["vest"]["status"] == "unseen" and not w1["ppe"]["vest"]["required"]
    assert 0 < w1["violation_share"] <= 1
    assert r["events"][0]["worker_id"] == 1 and r["events"][0]["duration"] > 3.9
    assert max(o["persons"] for o in r["occupancy"]) == 2  # phantom not counted
    assert set(r["tracks"]) == {"1", "2"}
    row = r["tracks"]["1"][15]
    assert row[5] == "NO_HARDHAT" and all(0 <= v <= 1 for v in row[1:5])
    assert r["findings"]


def test_worker_never_seen_wearing_ppe_is_unconfirmed_not_compliant():
    dt = 0.2
    samples = []
    for i in range(10):  # 2 s: vest briefly flagged missing, then not visible
        ppe = {"hardhat": PPEObservation("unseen"), "mask": PPEObservation("unseen"),
               "vest": PPEObservation("missing" if i < 2 else "unseen", 0.6)}
        samples.append(FrameSample(i, i * dt, dt, [TrackObservation(1, [0, 0, 64, 144], 0.9, ppe)], {}, [], []))
    vt = ViolationTracker(["hardhat", "vest", "mask"], min_violation_s=1.0)
    for s in samples:
        vt.feed(s)
    vt.finalize()
    r = build_results(samples, vt.events, 2.0, (640, 360), ["hardhat", "vest", "mask"])
    p = r["persons"][0]
    assert p["status"] == "unconfirmed" and p["ppe"]["vest"]["status"] == "brief"
    assert "only confirmed for 0 of 1" in r["findings"][0]
