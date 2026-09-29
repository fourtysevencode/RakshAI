"""PDF safety report (ReportLab platypus + matplotlib charts).

All images (charts, worker thumbnails, evidence frames) are passed around as
in-memory bytes; nothing but the final PDF is written to disk.
"""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ---- palette (matches the web UI) -----------------------------------------------
INK = colors.HexColor("#1F2328")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#E5E7EB")
PANEL = colors.HexColor("#F6F7F9")
YELLOW = colors.HexColor("#F5B400")
RED = colors.HexColor("#C0392B")
AMBER = colors.HexColor("#B7791F")
GREEN = colors.HexColor("#2E7D32")
TYPE_HEX = {"NO_HARDHAT": "#E4572E", "NO_VEST": "#E0A100", "NO_MASK": "#7C5CBF"}
TYPE_LABEL = {"NO_HARDHAT": "No hardhat", "NO_VEST": "No safety vest", "NO_MASK": "No mask"}
STATUS_STYLE = {  # PPE item status -> (label, colour)
    "violation": ("MISSING", RED),
    "brief": ("BRIEF LAPSE", AMBER),
    "ok": ("OK", GREEN),
    "unseen": ("NOT SEEN", MUTED),
}
SEVERITY_COLOR = {"high": RED, "medium": AMBER, "low": colors.HexColor("#8A6D00"), "none": GREEN}
MAX_TIMELINE_ROWS = 25
PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                    textColor=INK, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                    textColor=INK, spaceBefore=10, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=_ss["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=14,
                    textColor=INK, spaceBefore=2, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=INK)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, leading=10.5, textColor=MUTED)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=11)
CELL_B = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=10, bulletIndent=0, spaceAfter=3)
KPI_NUM = ParagraphStyle("KpiNum", parent=BODY, fontName="Helvetica-Bold", fontSize=20, leading=23, alignment=TA_CENTER)
KPI_LBL = ParagraphStyle("KpiLbl", parent=SMALL, alignment=TA_CENTER, fontSize=7.5, leading=9.5)


# ---- formatting helpers -----------------------------------------------------------
def fmt_ts(s: Optional[float]) -> str:
    if s is None:
        return "-"
    s = max(0.0, float(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:04.1f}" if h >= 1 else f"{int(m):02d}:{sec:04.1f}"


def fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "-"
    s = float(s)
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m} min {sec:02d} s" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def fmt_pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hex(c) -> str:
    return c.hexval().replace("0x", "#")


def _image(data: bytes, max_w: float, max_h: float) -> Optional[Image]:
    if not data:
        return None
    iw, ih = ImageReader(BytesIO(data)).getSize()
    scale = min(max_w / iw, max_h / ih)
    return Image(BytesIO(data), width=iw * scale, height=ih * scale)


def _fig_image(fig, width: float) -> Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    iw, ih = ImageReader(BytesIO(buf.getvalue())).getSize()
    return Image(buf, width=width, height=width * ih / iw)


def _grid_table(rows: List[list], col_widths: List[float], header: bool = True, zebra: bool = True) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ]
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), PANEL))
    t.setStyle(TableStyle(style))
    return t


def _status_para(status: str, compliance: Optional[float], required: bool) -> Paragraph:
    label, color = STATUS_STYLE.get(status, ("-", MUTED))
    if not required:
        color = MUTED
    pct = f" <font color='{_hex(MUTED)}'>{fmt_pct(compliance)}</font>" if compliance is not None else ""
    req = "" if required else f" <font size=7 color='{_hex(MUTED)}'>(not req.)</font>"
    return Paragraph(f"<font color='{_hex(color)}'><b>{label}</b></font>{pct}{req}", CELL)


# ---- page decoration --------------------------------------------------------------
class _NumberedCanvas(rl_canvas.Canvas):
    """Canvas that knows the total page count so the footer can say 'Page X of Y'."""

    meta: Dict[str, str] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_pages = []

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for state in self._saved_pages:
            self.__dict__.update(state)
            self._decorate(total)
            super().showPage()
        super().save()

    def _decorate(self, total: int):
        m = self.meta
        self.saveState()
        self.setFillColor(YELLOW)
        self.rect(0, PAGE_H - 5 * mm, PAGE_W, 5 * mm, stroke=0, fill=1)
        self.setFillColor(INK)
        self.setFont("Helvetica-Bold", 8.5)
        self.drawString(MARGIN, PAGE_H - 11 * mm, "RakshAI  ·  PPE Safety Report")
        self.setFont("Helvetica", 8.5)
        self.setFillColor(MUTED)
        self.drawRightString(PAGE_W - MARGIN, PAGE_H - 11 * mm, f"{m.get('site', '')}  ·  {m.get('camera', '')}")
        self.setStrokeColor(LINE)
        self.setLineWidth(0.5)
        self.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
        self.setFont("Helvetica", 7.5)
        self.drawString(MARGIN, 8 * mm, f"Analysis {m.get('analysis_id', '')}  ·  Generated {m.get('generated', '')}")
        self.drawRightString(PAGE_W - MARGIN, 8 * mm, f"Page {self._pageNumber} of {total}")
        self.restoreState()


# ---- charts -------------------------------------------------------------------------
def _ppe_chart(ppe: List[dict]) -> Optional[Image]:
    items = [b for b in ppe if b["compliance"] is not None]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.5 + 0.45 * len(items)))
    labels = [b["label"] + ("" if b["required"] else " (not required)") for b in items]
    vals = [b["compliance"] * 100 for b in items]
    cols = [TYPE_HEX[b["type"]] if b["required"] else "#C4C8CE" for b in items]
    ax.barh(labels, [100] * len(items), color="#EEF0F3", height=0.55)
    ax.barh(labels, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(min(v + 1.5, 92), i, f"{v:.0f}%", va="center", fontsize=9, color="#1F2328")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Compliance (% of observed worker-time wearing the item)", fontsize=8, color="#6B7280")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5, length=0)
    return _fig_image(fig, CONTENT_W)


def _timeline_chart(result: dict) -> Optional[Image]:
    persons = [p for p in result["persons"]]
    if not persons:
        return None
    duration = result["summary"]["duration_s"] or max(p["last_seen"] for p in persons)
    ranked = sorted(persons, key=lambda p: (-p["severity"]["score"], p["worker_id"]))[:MAX_TIMELINE_ROWS]
    rows = sorted(ranked, key=lambda p: p["worker_id"])
    events_by_worker: Dict[int, List[dict]] = {}
    for e in result["events"]:
        events_by_worker.setdefault(e["worker_id"], []).append(e)
    types = [t for t in TYPE_HEX if any(e["type"] == t for e in result["events"])]

    # Visible segments per worker from their sampled track (gaps > 0.6 s split a segment)
    visible: Dict[int, List[tuple]] = {}
    for wid, samples in result.get("tracks", {}).items():
        segs, start, prev = [], None, None
        for row in samples:
            t = row[0]
            if start is None:
                start = t
            elif t - prev > 0.6:
                segs.append((start, prev - start + 0.2))
                start = t
            prev = t
        if start is not None:
            segs.append((start, prev - start + 0.2))
        visible[int(wid)] = segs

    n = len(rows)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 1.3 + 0.32 * n * 1.0 + 1.1), sharex=True,
        gridspec_kw={"height_ratios": [max(1.2, 0.32 * n * 1.0 + 0.4), 1.1], "hspace": 0.12},
    )
    lane_h = 0.8
    for i, p in enumerate(rows):
        y = n - 1 - i
        segs = visible.get(p["worker_id"]) or [(p["first_seen"], p["last_seen"] - p["first_seen"])]
        ax.broken_barh(segs, (y - lane_h / 2, lane_h), color="#E3E6EA")
        evs = events_by_worker.get(p["worker_id"], [])
        sub_h = lane_h / max(1, len(types))
        for e in evs:
            k = types.index(e["type"])
            ax.broken_barh([(e["t_start"], max(e["duration"], duration * 0.004))],
                           (y - lane_h / 2 + k * sub_h, sub_h), color=TYPE_HEX[e["type"]])
            if e["near_hazard"]:
                ax.plot(e["t_start"], y + lane_h / 2 + 0.08, marker="v", color="#1F2328", markersize=3.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([p["label"] for p in reversed(rows)], fontsize=8)
    ax.set_ylim(-0.7, n - 0.3 + 0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_HEX[t]) for t in types]
    labels = [TYPE_LABEL[t] for t in types]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#E3E6EA"))
    labels.append("On camera")
    if any(e["near_hazard"] for e in result["events"]):
        handles.append(plt.Line2D([], [], marker="v", color="#1F2328", linestyle="", markersize=4))
        labels.append("Near machinery/vehicle")
    ax.legend(handles, labels, fontsize=7, ncol=len(labels), loc="lower left", bbox_to_anchor=(0, 1.0),
              frameon=False, handlelength=1.2, columnspacing=1.0)

    occ = result["occupancy"]
    ts = [o["t"] for o in occ]
    ax2.step(ts, [o["persons"] for o in occ], where="post", color="#1F2328", linewidth=1.1, label="Workers in frame")
    for key, col, lbl in (("machinery", "#F5B400", "Machinery present"), ("vehicle", "#4C78A8", "Vehicle present")):
        if any(o.get(key) for o in occ):
            ax2.fill_between(ts, 0, [1 if o.get(key) else 0 for o in occ], step="post", color=col, alpha=0.25,
                             transform=ax2.get_xaxis_transform(), label=lbl, linewidth=0)
    ax2.set_ylim(0, max(1, max((o["persons"] for o in occ), default=1)) + 0.5)
    ax2.set_ylabel("Workers", fontsize=8)
    ax2.legend(fontsize=7, loc="upper right", frameon=False, ncol=3)
    ax2.set_xlim(0, duration)
    ax2.set_xlabel("Video time", fontsize=8)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_ts(v)[:-2] if v >= 60 else f"{v:.0f}s"))
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=7.5)
        a.grid(axis="x", color="#E5E7EB", linewidth=0.5)
        a.set_axisbelow(True)
    return _fig_image(fig, CONTENT_W)


# ---- sections ---------------------------------------------------------------------
def _kpi_tiles(summary: dict) -> Table:
    nc = summary["workers_non_compliant"]
    tiles = [
        (str(summary["workers_observed"]), "Workers observed", INK),
        (str(nc), "Workers with violations", RED if nc else GREEN),
        (fmt_pct(summary["overall_compliance"]), "Overall PPE compliance", INK),
        (str(summary["total_events"]), "Violation events", RED if summary["total_events"] else GREEN),
        (fmt_dur(summary["longest_event"]["duration"]) if summary["longest_event"] else "-",
         "Longest violation", INK),
    ]
    cells = [[Paragraph(f"<font color='{_hex(c)}'>{_esc(v)}</font>", KPI_NUM) for v, _, c in tiles],
             [Paragraph(_esc(lbl), KPI_LBL) for _, lbl, _ in tiles]]
    w = CONTENT_W / len(tiles)
    t = Table(cells, colWidths=[w] * len(tiles), rowHeights=[28, 16])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEABOVE", (0, 0), (-1, 0), 2, YELLOW),
    ]))
    return t


def _summary_section(meta: dict, result: dict) -> list:
    s, v, st = result["summary"], result["video"], result["settings"]
    story = [Paragraph("PPE Compliance Report", H1),
             Paragraph(f"{_esc(meta.get('site_name', ''))}  ·  Camera {_esc(meta.get('camera_id', ''))}", SMALL),
             Spacer(1, 8)]
    fps = "every frame" if not st["analysis_fps"] else f"{st['sample_fps']:g} frames/s"
    required = ", ".join(b["label"] for b in result["ppe"] if b["required"]) or "none"
    info = [
        ["Site", meta.get("site_name", "-"), "Video", meta.get("filename", "-")],
        ["Camera", meta.get("camera_id", "-"), "Duration", fmt_dur(v["duration_s"])],
        ["Analysed", meta.get("created_at", "-").replace("T", " ").replace("Z", " UTC"),
         "Resolution", f"{v['width']}×{v['height']}"],
        ["Required PPE", required, "Sampling", f"{fps} ({s['frames_analyzed']} frames analysed)"],
    ]
    info = [[Paragraph(f"<font color='{_hex(MUTED)}'>{a}</font>", CELL), Paragraph(_esc(b), CELL),
             Paragraph(f"<font color='{_hex(MUTED)}'>{c}</font>", CELL), Paragraph(_esc(d), CELL)] for a, b, c, d in info]
    story += [_grid_table(info, [24 * mm, 62 * mm, 22 * mm, CONTENT_W - 108 * mm], header=False, zebra=False),
              Spacer(1, 10), _kpi_tiles(s), Spacer(1, 12), Paragraph("Key findings", H2)]
    for f in result["findings"]:
        story.append(Paragraph(_esc(f), BULLET, bulletText="•"))
    story += [Spacer(1, 6), Paragraph(
        "This report is generated automatically from video by a computer-vision model. Observations are subject "
        "to model limitations (see Methodology) and should be reviewed by qualified safety personnel before being "
        "used for compliance or disciplinary decisions.", SMALL)]
    return story


def _ppe_section(result: dict) -> list:
    story = [Paragraph("PPE breakdown", H2)]
    rows = [["PPE item", "Required", "Workers assessed", "Workers in violation", "Compliance", "Events", "Violation time"]]
    for b in result["ppe"]:
        rows.append([b["label"], "Yes" if b["required"] else "No", b["workers_observed"],
                     b["workers_violating"] if b["required"] else "-", fmt_pct(b["compliance"]),
                     b["events"] if b["required"] else "-", fmt_dur(b["violation_s"]) if b["required"] else "-"])
    story.append(_grid_table(rows, [30 * mm, 18 * mm, 27 * mm, 30 * mm, 24 * mm, 17 * mm, CONTENT_W - 146 * mm]))
    story.append(Paragraph("Boots are not assessed: the detection model has no footwear class.", SMALL))
    chart = _ppe_chart(result["ppe"])
    if chart:
        story += [Spacer(1, 6), chart]
    return story


def _timeline_section(result: dict) -> list:
    story = [Paragraph("Timeline", H2)]
    chart = _timeline_chart(result)
    if chart is None:
        return story + [Paragraph("No workers were tracked in this footage.", BODY)]
    story.append(chart)
    if len(result["persons"]) > MAX_TIMELINE_ROWS:
        story.append(Paragraph(f"Showing the {MAX_TIMELINE_ROWS} workers with the highest severity.", SMALL))
    tl = result["timeline"]
    busy = sorted((b for b in tl["buckets"] if b["events"]), key=lambda b: -sum(b["events"].values()))[:3]
    if busy and len(tl["buckets"]) > 3 and tl["bucket_s"] >= 10:
        spans = ", ".join(f"{fmt_ts(b['t'])}–{fmt_ts(b['t'] + tl['bucket_s'])} ({sum(b['events'].values())})"
                          for b in busy)
        story.append(Paragraph(f"Busiest periods by new violation events: {spans}.", SMALL))
    return story


def _worker_block(p: dict, events: List[dict], evidence) -> list:
    sev = p["severity"]
    head = Paragraph(
        f"{_esc(p['label'])} &nbsp;<font size=8 color='{_hex(SEVERITY_COLOR[sev['level']])}'>"
        f"<b>{sev['level'].upper()} SEVERITY</b></font> <font size=8 color='{_hex(MUTED)}'>"
        f"(score {sev['score']:g})</font>", H3)
    facts = [
        ["On camera", f"{fmt_ts(p['first_seen'])} – {fmt_ts(p['last_seen'])} ({fmt_dur(p['visible_s'])} visible)"],
        ["In violation", f"{fmt_dur(p['violation_s'])} ({fmt_pct(p['violation_share'])} of visible time)"],
        ["Near machinery", fmt_dur(p["exposure"].get("machinery", 0))],
        ["Near vehicles", fmt_dur(p["exposure"].get("vehicle", 0))],
    ]
    facts_t = Table([[Paragraph(f"<font color='{_hex(MUTED)}'>{a}</font>", CELL), Paragraph(_esc(b), CELL)]
                     for a, b in facts], colWidths=[26 * mm, 80 * mm])
    facts_t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    ppe_row = Table([[Paragraph(f"<font color='{_hex(MUTED)}'>{it['label']}</font>", CELL) for it in p["ppe"].values()],
                     [_status_para(it["status"], it["compliance"], it["required"]) for it in p["ppe"].values()]],
                    colWidths=[36 * mm] * len(p["ppe"]))
    ppe_row.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LINEAFTER", (0, 0), (-2, -1), 2, colors.white),
                                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    right = [facts_t, Spacer(1, 5), ppe_row]
    thumb = _image(evidence.thumbnails.get(p["track_id"], b"") if evidence else b"", 34 * mm, 46 * mm)
    top = Table([[thumb or Paragraph("no image", SMALL), right]], colWidths=[38 * mm, CONTENT_W - 38 * mm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))

    block = [head, top, Spacer(1, 5)]
    if events:
        rows = [["Violation", "Start", "End", "Duration", "Peak conf.", "Near hazard"]]
        for e in events:
            rows.append([e["label"], fmt_ts(e["t_start"]), fmt_ts(e["t_end"]), fmt_dur(e["duration"]),
                         f"{e['peak_conf']:.2f}", "Yes" if e["near_hazard"] else "-"])
        block.append(_grid_table(rows, [36 * mm, 24 * mm, 24 * mm, 24 * mm, 22 * mm, CONTENT_W - 130 * mm]))

    shots = []
    if evidence:
        for e in sorted(events, key=lambda e: -e["severity"]):
            img = _image(evidence.event_images.get(e["event_id"], b""), (CONTENT_W - 6 * mm) / 2, 60 * mm)
            if img:
                cap = Paragraph(f"{_esc(e['label'])} at {fmt_ts(e['evidence']['t'] if e['evidence'] else e['t_start'])} "
                                f"(confidence {e['peak_conf']:.2f})", SMALL)
                shots.append([img, cap])
            if len(shots) == 2:
                break
    if shots:
        cells = [[s[0] for s in shots], [s[1] for s in shots]]
        t = Table(cells, colWidths=[CONTENT_W / 2] * len(shots))
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        block += [Spacer(1, 5), t]
    return [KeepTogether(block), Spacer(1, 10)]


def _workers_section(result: dict, evidence) -> list:
    persons = result["persons"]
    story = [Paragraph("Worker profiles", H2)]
    if not persons:
        return story + [Paragraph("No workers were tracked in this footage.", BODY)]
    ev_by_worker: Dict[int, List[dict]] = {}
    for e in result["events"]:
        ev_by_worker.setdefault(e["worker_id"], []).append(e)
    flagged = sorted((p for p in persons if p["status"] == "non_compliant"), key=lambda p: -p["severity"]["score"])
    others = [p for p in persons if p["status"] != "non_compliant"]
    if flagged:
        story.append(Paragraph("Workers with at least one violation, most severe first. Severity weighs each "
                               "violation's duration by PPE type (hardhat ×3, vest ×2, mask ×1) and ×1.5 near "
                               "machinery or vehicles.", SMALL))
        story.append(Spacer(1, 6))
        for p in flagged:
            story += _worker_block(p, ev_by_worker.get(p["worker_id"], []), evidence)
    else:
        story.append(Paragraph("No worker had a sustained PPE violation.", BODY))
    if others:
        story.append(Paragraph("Workers without violations", H3))
        rows = [["", "Worker", "On camera", "Visible", *[it["label"] for it in others[0]["ppe"].values()], "Status"]]
        for p in others:
            thumb = _image(evidence.thumbnails.get(p["track_id"], b"") if evidence else b"", 9 * mm, 12 * mm)
            rows.append([thumb or "", p["label"], f"{fmt_ts(p['first_seen'])} – {fmt_ts(p['last_seen'])}",
                         fmt_dur(p["visible_s"]),
                         *[_status_para(it["status"], it["compliance"], it["required"]) for it in p["ppe"].values()],
                         Paragraph("Compliant" if p["status"] == "compliant" else "PPE not confirmed", CELL)])
        story.append(_grid_table(rows, [11 * mm, 18 * mm, 30 * mm, 14 * mm, 25 * mm, 25 * mm, 25 * mm,
                                        CONTENT_W - 148 * mm]))
    brief = result["summary"]["brief_tracks"]
    if brief:
        story.append(Paragraph(f"{brief} brief detection{'s' if brief != 1 else ''} (under 1 s on camera, no "
                               "violations) were excluded from worker counts.", SMALL))
    return story


def _events_section(result: dict) -> list:
    story = [Paragraph("Event log", H2)]
    events = result["events"]
    if not events:
        return story + [Paragraph("No PPE violation events were detected.", BODY)]
    rows = [["#", "Worker", "Violation", "Start", "End", "Duration", "Peak conf.", "Near hazard", "Severity"]]
    for i, e in enumerate(events, 1):
        rows.append([i, e["worker"], e["label"], fmt_ts(e["t_start"]), fmt_ts(e["t_end"]), fmt_dur(e["duration"]),
                     f"{e['peak_conf']:.2f}", "Yes" if e["near_hazard"] else "-", f"{e['severity']:g}"])
    story.append(_grid_table(rows, [9 * mm, 20 * mm, 28 * mm, 18 * mm, 18 * mm, 20 * mm, 18 * mm, 20 * mm,
                                    CONTENT_W - 151 * mm]))
    return story


def _scene_section(result: dict) -> list:
    sc = result["scene"]
    rows = [["", "Peak in one frame", "Share of footage present"],
            ["Workers", sc["peak_workers"], "-"],
            ["Machinery", sc["peak_machinery"], fmt_pct(sc["machinery_time_share"])],
            ["Vehicles", sc["peak_vehicles"], fmt_pct(sc["vehicle_time_share"])],
            ["Safety cones", sc["peak_cones"], fmt_pct(sc["cones_time_share"])]]
    story = [Paragraph("Scene context", H2), _grid_table(rows, [40 * mm, 45 * mm, 50 * mm])]
    un = sc.get("unattributed_observations") or {}
    if un:
        parts = ", ".join(f"{k} ×{v}" for k, v in sorted(un.items()))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Missing-PPE detections that could not be linked to a tracked worker "
                               f"(frame-level, not counted as events): {_esc(parts)}.", SMALL))
    return story


def _methodology_section(result: dict) -> list:
    st, m, v = result["settings"], result.get("model") or {}, result["video"]
    classes = ", ".join(m.get("classes") or [])
    sampling = "Every frame" if not st["analysis_fps"] else f"{st['sample_fps']:g} frames per second"
    items = [
        f"<b>Detection.</b> YOLO11n object detector ({_esc(m.get('path', 'best.onnx'))}, "
        f"{_esc(m.get('backend') or 'n/a')} backend) with classes: {_esc(classes)}. Detections below "
        f"{st['conf_threshold']:.2f} confidence are ignored for PPE and hazards.",
        f"<b>Sampling.</b> {sampling} of video, using each frame's real timestamp; frames are processed at up to "
        f"{v['processed_width']}×{v['processed_height']}. Processing took {fmt_dur(result.get('processing_s'))}.",
        "<b>Worker tracking.</b> Person detections are linked across frames with ByteTrack. Each tracked "
        "worker keeps an ID for as long as they stay visible (short occlusions of up to about 2 s are bridged).",
        "<b>PPE association.</b> A PPE or missing-PPE detection is assigned to the worker whose box contains most "
        "of it, preferring the expected body region (head for hardhat/mask, torso for vest). Per sample, each "
        "item is OK, MISSING or NOT SEEN (e.g. head out of view); NOT SEEN time is excluded from compliance.",
        f"<b>Violation events.</b> An event starts once an item is missing for at least {st['min_violation_s']:g} s "
        f"and ends after {st['cooldown_s']:g} s without a missing observation, so model flicker neither creates "
        "nor splits events. Shorter lapses are reported as BRIEF LAPSE.",
        "<b>Proximity.</b> A worker is near machinery/vehicles when the gap between their boxes is under half the "
        "worker's height. This is an image-space approximation, not a measured distance.",
    ]
    limits = [
        "Boots/footwear cannot be assessed (no model class).",
        "Worker IDs are per video; a worker who leaves and re-enters, or is heavily occluded, may get a new ID.",
        "Masks and hardhats are hard to judge for workers facing away or far from the camera.",
        "Proximity uses 2D image distance and does not account for depth.",
    ]
    story = [Paragraph("Methodology", H2)]
    story += [Paragraph(i, BULLET, bulletText="•") for i in items]
    story += [Paragraph("Limitations", H3)]
    story += [Paragraph(_esc(i), BULLET, bulletText="•") for i in limits]
    story += [Spacer(1, 6), Paragraph(
        "AI-generated observations must be reviewed by qualified safety personnel before being used for compliance "
        "decisions.", SMALL)]
    return story


def generate_report(output_path: Path, meta: dict, result: dict, evidence=None) -> str:
    """Build the PDF report. `evidence` is an EvidenceStore (in-memory JPEGs) or None."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    class Canvas(_NumberedCanvas):
        pass

    Canvas.meta = {"analysis_id": meta.get("analysis_id", ""), "site": meta.get("site_name", ""),
                   "camera": meta.get("camera_id", ""), "generated": generated}
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=18 * mm,
        bottomMargin=18 * mm, title=f"RakshAI PPE report – {meta.get('site_name', '')}", author="RakshAI",
    )
    story = []
    story += _summary_section(meta, result)
    story += _ppe_section(result)
    story += [PageBreak()]
    story += _timeline_section(result)
    story += _workers_section(result, evidence)
    story += _events_section(result)
    story += _scene_section(result)
    story += _methodology_section(result)
    doc.build(story, canvasmaker=Canvas)
    return str(output_path)
