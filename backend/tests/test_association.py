from backend.models.schemas import Detection
from backend.services.association import associate
from backend.services.tracking import suppress_duplicate_persons


def det(name, conf, bbox):
    return Detection(0, name, conf, bbox)


def test_head_item_goes_to_person_whose_head_region_matches():
    # Worker 1 stands in front; worker 2 is taller and behind, boxes overlap heavily.
    # The hardhat sits at the top of worker 1's box but mid-torso of worker 2's box.
    tracks = [(1, [100, 200, 200, 500], 0.9), (2, [90, 50, 210, 520], 0.9)]
    hardhat = det("NO-Hardhat", 0.8, [130, 205, 170, 240])
    obs, unattributed = associate(tracks, [hardhat])
    by_id = {o.track_id: o for o in obs}
    assert by_id[1].ppe["hardhat"].state == "missing"
    assert by_id[2].ppe["hardhat"].state == "unseen"
    assert unattributed == []


def test_vest_prefers_torso_band():
    tracks = [(1, [100, 100, 200, 400], 0.9), (2, [100, 250, 200, 550], 0.9)]
    # Vest centre at y=200: torso of worker 1 (ry=0.33), above worker 2's box.
    vest = det("Safety Vest", 0.9, [110, 160, 190, 240])
    obs, _ = associate(tracks, [vest])
    by_id = {o.track_id: o for o in obs}
    assert by_id[1].ppe["vest"].state == "present"
    assert by_id[2].ppe["vest"].state == "unseen"


def test_unmatched_violation_is_unattributed_and_positive_ppe_is_ignored():
    tracks = [(1, [0, 0, 100, 300], 0.9)]
    stray_violation = det("NO-Mask", 0.7, [500, 500, 520, 520])
    stray_ppe = det("Hardhat", 0.7, [600, 600, 620, 620])
    obs, unattributed = associate(tracks, [stray_violation, stray_ppe])
    assert [d.class_name for d in unattributed] == ["NO-Mask"]
    assert all(p.state == "unseen" for p in obs[0].ppe.values())


def test_conflicting_present_and_missing_uses_higher_confidence():
    tracks = [(1, [0, 0, 100, 300], 0.9)]
    obs, _ = associate(tracks, [det("Hardhat", 0.4, [30, 0, 70, 30]), det("NO-Hardhat", 0.7, [30, 0, 70, 30])])
    assert obs[0].ppe["hardhat"].state == "missing"
    obs, _ = associate(tracks, [det("Hardhat", 0.9, [30, 0, 70, 30]), det("NO-Hardhat", 0.7, [30, 0, 70, 30])])
    assert obs[0].ppe["hardhat"].state == "present"


def test_face_mid_box_in_closeup_still_assigned():
    # Close-up: person box spans the frame, face sits mid-box (outside the head band)
    tracks = [(1, [13, 87, 1620, 1080], 0.84)]
    obs, unattributed = associate(tracks, [det("NO-Mask", 0.95, [628, 547, 921, 775])])
    assert obs[0].ppe["mask"].state == "missing"
    assert unattributed == []


def test_duplicate_person_boxes_are_suppressed():
    tight = det("Person", 0.66, [745, 139, 1163, 689])
    loose = det("Person", 0.39, [500, 133, 1172, 693])
    other = det("Person", 0.5, [100, 100, 300, 600])
    kept = suppress_duplicate_persons([loose, tight, other])
    assert {d.confidence for d in kept} == {0.66, 0.5}
