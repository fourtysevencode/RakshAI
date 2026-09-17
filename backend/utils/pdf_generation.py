try:
    from pathlib import Path
    from typing import Optional, Dict, Any, List
except Exception:
    Path = None  # type: ignore
    Optional = None  # type: ignore
    Dict = None  # type: ignore
    Any = None  # type: ignore
    List = None  # type: ignore

# Optional dependency: ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None  # type: ignore
    letter = None  # type: ignore


def _ensure_path(p: Path):
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)


def generate_pdf(analysis_id: str,
                 meta: Dict[str, Any],
                 results: Dict[str, Any],
                 events: List[Dict[str, Any]],
                 output_path: Path) -> Optional[str]:
    """Generate a simple PDF report for the given analysis.

    If ReportLab is unavailable, fall back to a tiny text file with a .pdf extension.
    """
    _ensure_path(output_path)
    if canvas is None or letter is None:
        try:
            with open(str(output_path), "wb") as f:
                f.write(b"RakshAI Report (PDF placeholder)\n")
        except Exception:
            pass
        return str(output_path)

    c = canvas.Canvas(str(output_path), pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    text.textLine(f"RakshAI Analysis Report: {analysis_id}")
    text.textLine("")
    if meta:
        site = meta.get("site_name", "Unknown Site")
        cam = meta.get("camera_id", "Unknown Camera")
        text.textLine(f"Site: {site}")
        text.textLine(f"Camera: {cam}")
        text.textLine("")
    if results:
        duration = results.get("duration_seconds", 0.0)
        frames = results.get("frames_processed", 0)
        text.textLine(f"Duration (s): {duration}")
        text.textLine(f"Frames processed: {frames}")
        text.textLine("")
    if results.get("violations"):
        v = results.get("violations", {})
        text.textLine("Violations:")
        text.textLine(f"  Total: {v.get('total_violations', 0)}")
        types = v.get("types", {})
        if types:
            text.textLine("  Types:")
            for k, val in types.items():
                text.textLine(f"    {k}: {val}")
        text.textLine("")
    if events:
        text.textLine("Events:")
        for e in (events[:10] or []):
            t = e.get('timestamp', 0.0)
            typ = e.get('type', 'UNKNOWN')
            log = e.get('log', '')
            if log:
                text.textLine(f"- {typ} ({log}) at {t:.2f}s")
            else:
                text.textLine(f"- {typ} at {t:.2f}s")
        if len(events) > 10:
            text.textLine("... more events ...")
    c.drawText(text)
    c.showPage()
    c.save()
    return str(output_path)
