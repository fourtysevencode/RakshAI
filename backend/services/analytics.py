"""Aggregate sampled observations + violation events into the analysis result.

Everything here is pure data shaping (no OpenCV / model), so it is unit-testable.
Boxes in the output are normalized to 0-1 of the processed frame size.
"""
from collections import defaultdict
from typing import Dict, List, Optional

from backend.models.schemas import (
    PPE_ITEMS, PPE_LABEL, VIOLATION_TYPE, FrameSample, ViolationEvent,
)

SEVERITY_WEIGHT = {"hardhat": 3.0, "vest": 2.0, "mask": 1.0}
HAZARD_MULTIPLIER = 1.5
BRIEF_MIN_SAMPLES = 3
BRIEF_MIN_SECONDS = 1.0
TYPE_LABEL = {VIOLATION_TYPE[i]: f"No {PPE_LABEL[i].lower()}" for i in PPE_ITEMS}
BUCKET_SIZES = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
MAX_OCCUPANCY_POINTS = 400


def _norm(b: List[float], w: int, h: int) -> List[float]:
    return [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]


def severity_score(ev: ViolationEvent) -> float:
    return SEVERITY_WEIGHT[ev.item] * ev.duration * (HAZARD_MULTIPLIER if ev.near_hazard else 1.0)


def severity_level(score: float) -> str:
    if score >= 15:
        return "high"
    if score >= 5:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _fmt_s(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}m {sec:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


def build_results(
    samples: List[FrameSample],
    events: List[ViolationEvent],
    duration: float,
    proc_size: tuple,
    required_ppe: List[str],
) -> dict:
    pw, ph = proc_size
    if not duration and samples:
        duration = samples[-1].t + samples[-1].dt
    if duration:
        for ev in events:  # the last sample's dt can reach past the end of the clip
            ev.t_end = min(ev.t_end, duration)

    # ---- per-track aggregates -------------------------------------------------
    agg: Dict[int, dict] = {}
    for s in samples:
        for o in s.tracks:
            a = agg.get(o.track_id)
            if a is None:
                a = agg[o.track_id] = {
                    "first": s.t, "last": s.t + s.dt, "visible": 0.0, "n": 0,
                    "ppe": {i: {"present": 0.0, "missing": 0.0, "unseen": 0.0} for i in PPE_ITEMS},
                    "near": {"machinery": 0.0, "vehicle": 0.0},
                    "best": (o.confidence, s.t, o.bbox),
                    "path": [],
                }
            a["last"] = s.t + s.dt
            a["visible"] += s.dt
            a["n"] += 1
            for item, po in o.ppe.items():
                a["ppe"][item][po.state] += s.dt
            for kind in o.near:
                a["near"][kind] = a["near"].get(kind, 0.0) + s.dt
            if o.confidence > a["best"][0]:
                a["best"] = (o.confidence, s.t, o.bbox)
            a["path"].append((s.t, s.dt, o.bbox))

    ev_by_track: Dict[int, List[ViolationEvent]] = defaultdict(list)
    for ev in events:
        ev_by_track[ev.track_id].append(ev)

    brief = [tid for tid, a in agg.items()
             if not ev_by_track.get(tid) and (a["n"] < BRIEF_MIN_SAMPLES or a["visible"] < BRIEF_MIN_SECONDS)]
    kept = sorted((tid for tid in agg if tid not in brief), key=lambda t: (agg[t]["first"], t))
    worker_of = {tid: i + 1 for i, tid in enumerate(kept)}

    # ---- events ----------------------------------------------------------------
    events_out = []
    for ev in events:
        wid = worker_of.get(ev.track_id)
        events_out.append({
            "event_id": ev.event_id,
            "worker_id": wid,
            "worker": f"Worker {wid}",
            "track_id": ev.track_id,
            "type": ev.type,
            "item": ev.item,
            "label": TYPE_LABEL[ev.type],
            "t_start": round(ev.t_start, 3),
            "t_end": round(ev.t_end, 3),
            "duration": round(ev.duration, 3),
            "frame_start": ev.frame_start,
            "frame_end": ev.frame_end,
            "peak_conf": round(ev.peak_conf, 3),
            "mean_conf": round(ev.mean_conf, 3),
            "samples": ev.samples,
            "near_hazard": ev.near_hazard,
            "severity": round(severity_score(ev), 2),
            "evidence": ({"t": round(ev.evidence["t"], 3), "bbox": _norm(ev.evidence["bbox"], pw, ph)}
                         if ev.evidence else None),
        })

    # ---- persons ---------------------------------------------------------------
    persons = []
    for tid in kept:
        a = agg[tid]
        evs = ev_by_track.get(tid, [])
        ppe = {}
        for item in PPE_ITEMS:
            p = a["ppe"][item]
            observed = p["present"] + p["missing"]
            n_ev = sum(1 for e in evs if e.item == item)
            ppe[item] = {
                "label": PPE_LABEL[item],
                "required": item in required_ppe,
                "present_s": round(p["present"], 2),
                "missing_s": round(p["missing"], 2),
                "unseen_s": round(p["unseen"], 2),
                "compliance": round(p["present"] / observed, 4) if observed > 0 else None,
                "events": n_ev,
                # brief = missing was seen but never long enough to become an event
            "status": ("violation" if n_ev else "unseen" if observed == 0
                       else "brief" if p["missing"] > 0 else "ok"),
            }
        req_items = [i for i in PPE_ITEMS if i in required_ppe]
        score = sum(severity_score(e) for e in evs)
        # Measured on the worker's own samples so it can't exceed their visible time
        viol_s = sum(dt for t, dt, _ in a["path"] if any(e.t_start - 1e-6 <= t < e.t_end for e in evs))
        # "compliant" needs positive evidence: required PPE actually seen being worn
        observed = [i for i in req_items if ppe[i]["compliance"] is not None]
        if evs:
            status = "non_compliant"
        elif observed and all(ppe[i]["compliance"] >= 0.5 for i in observed):
            status = "compliant"
        else:
            status = "unconfirmed"
        conf, bt, bb = a["best"]
        persons.append({
            "worker_id": worker_of[tid],
            "label": f"Worker {worker_of[tid]}",
            "track_id": tid,
            "first_seen": round(a["first"], 3),
            "last_seen": round(a["last"], 3),
            "visible_s": round(a["visible"], 2),
            "samples": a["n"],
            "status": status,
            "severity": {"score": round(score, 2), "level": severity_level(score)},
            "violation_s": round(viol_s, 2),
            "violation_share": round(viol_s / a["visible"], 4) if a["visible"] else 0.0,
            "ppe": ppe,
            "exposure": {k: round(v, 2) for k, v in a["near"].items()},
            "event_ids": [e.event_id for e in evs],
            "best_frame": {"t": round(bt, 3), "bbox": _norm(bb, pw, ph), "confidence": round(conf, 3)},
        })

    # ---- PPE breakdown -----------------------------------------------------------
    ppe_breakdown = []
    for item in PPE_ITEMS:
        present = sum(p["ppe"][item]["present_s"] for p in persons)
        missing = sum(p["ppe"][item]["missing_s"] for p in persons)
        item_events = [e for e in events_out if e["item"] == item]
        ppe_breakdown.append({
            "item": item,
            "label": PPE_LABEL[item],
            "type": VIOLATION_TYPE[item],
            "required": item in required_ppe,
            "workers_observed": sum(1 for p in persons if p["ppe"][item]["status"] != "unseen"),
            "workers_violating": sum(1 for p in persons if p["ppe"][item]["events"]),
            "compliance": round(present / (present + missing), 4) if present + missing > 0 else None,
            "events": len(item_events),
            "violation_s": round(sum(e["duration"] for e in item_events), 2),
        })

    # ---- summary -----------------------------------------------------------------
    req_items = [i for i in PPE_ITEMS if i in required_ppe]
    req_present = sum(p["ppe"][i]["present_s"] for p in persons for i in req_items)
    req_missing = sum(p["ppe"][i]["missing_s"] for p in persons for i in req_items)
    total_visible = sum(p["visible_s"] for p in persons)
    total_viol = sum(p["violation_s"] for p in persons)
    longest = max(events_out, key=lambda e: e["duration"], default=None)
    events_by_type = {VIOLATION_TYPE[i]: 0 for i in PPE_ITEMS if i in required_ppe}
    for e in events_out:
        events_by_type[e["type"]] = events_by_type.get(e["type"], 0) + 1
    summary = {
        "duration_s": round(duration, 3),
        "frames_analyzed": len(samples),
        "workers_observed": len(persons),
        "workers_non_compliant": sum(1 for p in persons if p["status"] == "non_compliant"),
        "workers_compliant": sum(1 for p in persons if p["status"] == "compliant"),
        "workers_unconfirmed": sum(1 for p in persons if p["status"] == "unconfirmed"),
        "brief_tracks": len(brief),
        "total_events": len(events_out),
        "events_by_type": events_by_type,
        "overall_compliance": round(req_present / (req_present + req_missing), 4) if req_present + req_missing > 0 else None,
        "avg_event_s": round(sum(e["duration"] for e in events_out) / len(events_out), 2) if events_out else 0.0,
        "longest_event": ({k: longest[k] for k in ("event_id", "worker", "label", "duration", "t_start")}
                          if longest else None),
        "events_per_worker": round(len(events_out) / len(persons), 2) if persons else 0.0,
        "events_per_hour": round(len(events_out) / duration * 3600, 1) if duration else 0.0,
        "violation_time_share": round(total_viol / total_visible, 4) if total_visible else 0.0,
        "events_near_hazard": sum(1 for e in events_out if e["near_hazard"]),
    }

    # ---- timeline buckets --------------------------------------------------------
    bucket = next((b for b in BUCKET_SIZES if duration / b <= 40), BUCKET_SIZES[-1])
    n_buckets = max(1, int(-(-duration // bucket))) if duration else 1
    buckets = [{"t": i * bucket, "events": {}, "violating_workers": 0, "workers_present": 0}
               for i in range(n_buckets)]
    for e in events_out:
        i = min(n_buckets - 1, int(e["t_start"] // bucket))
        buckets[i]["events"][e["type"]] = buckets[i]["events"].get(e["type"], 0) + 1
    for i, b in enumerate(buckets):
        lo, hi = i * bucket, (i + 1) * bucket
        b["violating_workers"] = len({e["worker_id"] for e in events_out if e["t_start"] < hi and e["t_end"] > lo})
    for s in samples:
        i = min(n_buckets - 1, int(s.t // bucket))
        present = sum(1 for o in s.tracks if o.track_id in worker_of)
        buckets[i]["workers_present"] = max(buckets[i]["workers_present"], present)

    # ---- occupancy + scene -------------------------------------------------------
    # Count only real workers (not brief/phantom tracks) as persons in frame
    worker_counts = [sum(1 for o in s.tracks if o.track_id in worker_of) for s in samples]
    occupancy = [{"t": round(s.t, 3), **s.counts, "persons": n} for s, n in zip(samples, worker_counts)]
    if len(occupancy) > MAX_OCCUPANCY_POINTS:
        step = len(occupancy) / MAX_OCCUPANCY_POINTS
        binned = []
        for k in range(MAX_OCCUPANCY_POINTS):
            chunk = occupancy[int(k * step):int((k + 1) * step)] or [occupancy[int(k * step)]]
            binned.append({"t": chunk[0]["t"], **{key: max(c[key] for c in chunk) for key in chunk[0] if key != "t"}})
        occupancy = binned
    total_dt = sum(s.dt for s in samples) or 1.0
    unattributed: Dict[str, int] = defaultdict(int)
    for s in samples:
        for d in s.unattributed:
            unattributed[d.class_name] += 1
    scene = {
        "peak_workers": max(worker_counts, default=0),
        "peak_machinery": max((s.counts.get("machinery", 0) for s in samples), default=0),
        "peak_vehicles": max((s.counts.get("vehicle", 0) for s in samples), default=0),
        "peak_cones": max((s.counts.get("cones", 0) for s in samples), default=0),
        "machinery_time_share": round(sum(s.dt for s in samples if s.counts.get("machinery")) / total_dt, 4),
        "vehicle_time_share": round(sum(s.dt for s in samples if s.counts.get("vehicle")) / total_dt, 4),
        "cones_time_share": round(sum(s.dt for s in samples if s.counts.get("cones")) / total_dt, 4),
        "unattributed_observations": dict(unattributed),
    }

    # ---- overlay tracks ----------------------------------------------------------
    tracks = {}
    for tid in kept:
        evs = ev_by_track.get(tid, [])
        rows = []
        for t, _, bb in agg[tid]["path"]:
            active = [e.type for e in evs if e.t_start - 1e-6 <= t < e.t_end]
            rows.append([round(t, 3), *_norm(bb, pw, ph), ",".join(active)])
        tracks[str(worker_of[tid])] = rows

    return {
        "summary": summary,
        "ppe": ppe_breakdown,
        "persons": persons,
        "events": events_out,
        "timeline": {"bucket_s": bucket, "buckets": buckets},
        "occupancy": occupancy,
        "scene": scene,
        "tracks": tracks,
        "findings": key_findings(summary, ppe_breakdown, persons, events_out, scene, total_visible),
    }


def key_findings(summary: dict, ppe: List[dict], persons: List[dict], events: List[dict],
                 scene: dict, total_visible: float) -> List[str]:
    out: List[str] = []
    n, bad = summary["workers_observed"], summary["workers_non_compliant"]
    if n == 0:
        return ["No workers were detected in this footage."]
    if bad:
        out.append(f"{bad} of {n} worker{'s' if n != 1 else ''} observed had at least one PPE violation.")
    elif summary["workers_compliant"] == n:
        out.append(f"All {n} worker{'s' if n != 1 else ''} observed complied with the required PPE.")
    else:
        out.append(f"No sustained violations, but required PPE was only confirmed for "
                   f"{summary['workers_compliant']} of {n} workers (others were not seen clearly enough).")
    req = [b for b in ppe if b["required"] and b["events"]]
    if req:
        top = max(req, key=lambda b: (b["violation_s"], b["events"]))
        out.append(f"{TYPE_LABEL[top['type']]} was the most significant violation "
                   f"({top['events']} event{'s' if top['events'] != 1 else ''}, {_fmt_s(top['violation_s'])} in total).")
    offenders = [p for p in persons if p["violation_s"] > 0]
    total_v = sum(p["violation_s"] for p in offenders)
    if len(offenders) >= 2 and total_v > 0:
        worst = max(offenders, key=lambda p: p["violation_s"])
        out.append(f"{worst['label']} accounts for {worst['violation_s'] / total_v:.0%} of all violation time.")
    elif len(offenders) == 1:
        w = offenders[0]
        out.append(f"{w['label']} was in violation for {_fmt_s(w['violation_s'])} "
                   f"({w['violation_share']:.0%} of their time on camera).")
    if summary["events_near_hazard"]:
        k = summary["events_near_hazard"]
        out.append(f"{k} violation event{'s' if k != 1 else ''} occurred near machinery or vehicles.")
    measured = [b for b in ppe if b["required"] and b["compliance"] is not None]
    if measured:
        low = min(measured, key=lambda b: b["compliance"])
        if low["compliance"] < 0.95:
            out.append(f"{low['label']} compliance was lowest at {low['compliance']:.0%} of observed worker-time.")
    unseen = sum(p["ppe"][i]["unseen_s"] for p in persons for i in p["ppe"] if p["ppe"][i]["required"])
    req_n = sum(1 for b in ppe if b["required"])
    if req_n and total_visible and unseen / (total_visible * req_n) > 0.4:
        out.append(f"PPE could not be assessed for {unseen / (total_visible * req_n):.0%} of observed worker-time "
                   "(e.g. worker facing away or partly out of frame).")
    return out[:5]
