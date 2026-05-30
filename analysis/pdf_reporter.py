"""
analysis/pdf_reporter.py
------------------------
Generates a professional PDF security report from a suite result.
"""

from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm

# ── Colour palette ────────────────────────────────────────────────────────
C_BG        = colors.HexColor("#0d1117")
C_ACCENT    = colors.HexColor("#00d4ff")
C_SUCCESS   = colors.HexColor("#3fb950")
C_FAIL      = colors.HexColor("#f85149")
C_WARN      = colors.HexColor("#e3b341")
C_GREY      = colors.HexColor("#8b949e")
C_WHITE     = colors.white
C_DARK      = colors.HexColor("#161b22")
C_BORDER    = colors.HexColor("#30363d")

STATUS_COLORS = {
    "SUCCESS":      C_SUCCESS,
    "FAILURE":      C_FAIL,
    "PARTIAL":      C_WARN,
    "INCONCLUSIVE": C_ACCENT,
    "ERROR":        colors.orange,
}

RISK_COLORS = {
    "CRITICAL": C_FAIL,
    "HIGH":     colors.orange,
    "MEDIUM":   C_WARN,
    "LOW":      C_SUCCESS,
}


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", fontSize=22, textColor=C_ACCENT,
                                 fontName="Helvetica-Bold", spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontSize=10, textColor=C_GREY,
                                    fontName="Helvetica", spaceAfter=12),
        "h2": ParagraphStyle("h2", fontSize=13, textColor=C_WHITE,
                              fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", fontSize=9, textColor=C_GREY,
                                fontName="Helvetica", spaceAfter=4),
        "code": ParagraphStyle("code", fontSize=8, textColor=C_ACCENT,
                                fontName="Courier", spaceAfter=4,
                                backColor=C_DARK, leftIndent=8, rightIndent=8),
        "finding": ParagraphStyle("finding", fontSize=9, textColor=C_WHITE,
                                   fontName="Helvetica-Bold"),
    }


def _metric_table(metrics: list[tuple]) -> Table:
    """Render a row of metric boxes: [(label, value), ...]"""
    data = [[Paragraph(v, ParagraphStyle("mv", fontSize=18, textColor=C_ACCENT,
                                          fontName="Helvetica-Bold", alignment=TA_CENTER))
             for _, v in metrics],
            [Paragraph(l, ParagraphStyle("ml", fontSize=8, textColor=C_GREY,
                                          fontName="Helvetica", alignment=TA_CENTER))
             for l, _ in metrics]]
    col_w = (PAGE_W - 2 * MARGIN) / len(metrics)
    t = Table(data, colWidths=[col_w] * len(metrics), rowHeights=[28, 14])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
        ("BOX",        (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _results_table(results: list, styles: dict) -> Table:
    headers = ["ID", "Attack Name", "Status", "Score", "Risk", "MITRE"]
    header_row = [Paragraph(h, ParagraphStyle("th", fontSize=8, textColor=C_ACCENT,
                                               fontName="Helvetica-Bold")) for h in headers]
    rows = [header_row]
    for r in results:
        status = r.get("status", "").upper()
        risk   = r.get("risk_level", "LOW").upper()
        sc     = r.get("score", 0)
        row = [
            Paragraph(r.get("attack_id", ""),   styles["body"]),
            Paragraph(r.get("attack_name", ""), styles["body"]),
            Paragraph(status, ParagraphStyle("st", fontSize=8, fontName="Helvetica-Bold",
                                              textColor=STATUS_COLORS.get(status, C_GREY))),
            Paragraph(f"{sc:.2f}", ParagraphStyle("sc", fontSize=8, fontName="Helvetica",
                                                   textColor=C_ACCENT)),
            Paragraph(risk, ParagraphStyle("rk", fontSize=8, fontName="Helvetica-Bold",
                                            textColor=RISK_COLORS.get(risk, C_GREY))),
            Paragraph(r.get("mitre_tactic_id", ""), styles["body"]),
        ]
        rows.append(row)

    col_widths = [30*mm, 55*mm, 28*mm, 20*mm, 22*mm, 25*mm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_DARK),
        ("BACKGROUND",    (0, 1), (-1, -1), C_BG),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_BG, C_DARK]),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return t


def generate_pdf_report(suite_result_dict: dict, output_path: str = None) -> str:
    """
    Generate a PDF report from a suite result dictionary.
    Returns the path to the generated PDF.
    """
    s = _styles()
    ts  = suite_result_dict.get("timestamp", "")[:16].replace("T", " ")
    suite = suite_result_dict.get("suite_name", "unknown").upper()
    model = suite_result_dict.get("model_name", "unknown")
    summary = suite_result_dict.get("summary", {})
    risk_summary = suite_result_dict.get("risk_summary", {})
    results = suite_result_dict.get("results", [])

    if output_path is None:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(reports_dir / f"report_{suite.lower()}_{ts_file}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("🛡 SentinelLLM Security Report", s["title"]))
    story.append(Paragraph(
        f"Suite: <b>{suite}</b> &nbsp;|&nbsp; Model: <b>{model}</b> &nbsp;|&nbsp; {ts} UTC",
        s["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=12))

    # ── Metrics ───────────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", s["h2"]))
    story.append(_metric_table([
        ("Total Attacks",   str(summary.get("total", 0))),
        ("Succeeded",       str(summary.get("successful", 0))),
        ("Success Rate",    f"{summary.get('success_rate', 0)*100:.1f}%"),
        ("Avg Score",       f"{summary.get('average_score', 0):.3f}"),
        ("Critical",        str(risk_summary.get("CRITICAL", 0))),
        ("High",            str(risk_summary.get("HIGH", 0))),
    ]))
    story.append(Spacer(1, 12))

    # ── Results table ─────────────────────────────────────────────────────
    story.append(Paragraph("Attack Results", s["h2"]))
    story.append(_results_table(results, s))
    story.append(Spacer(1, 12))

    # ── Findings detail ───────────────────────────────────────────────────
    successful = [r for r in results if r.get("status", "").upper() == "SUCCESS"]
    if successful:
        story.append(Paragraph("Successful Attack Details", s["h2"]))
        for r in successful:
            story.append(Paragraph(
                f"[{r['attack_id']}] {r['attack_name']} — Score: {r['score']:.2f}",
                s["finding"]
            ))
            story.append(Paragraph(f"MITRE: {r.get('mitre_tactic_id','')} | Risk: {r.get('risk_level','')}", s["body"]))
            if r.get("llm_response"):
                snippet = r["llm_response"][:300].replace("\n", " ")
                story.append(Paragraph(f"Response snippet: {snippet}...", s["code"]))
            story.append(Spacer(1, 6))

    # ── Defense recommendations ───────────────────────────────────────────
    if successful:
        story.append(Paragraph("Defense Recommendations", s["h2"]))
        from analysis.defense_advisor import get_recommendations
        recs = get_recommendations([r["attack_id"] for r in successful])
        for rec in recs:
            story.append(Paragraph(f"► [{rec.attack_id}] {rec.attack_title}", s["finding"]))
            story.append(Paragraph(f"Category: {rec.category}", s["body"]))
            story.append(Paragraph(f"Fix: {rec.remediation}", s["body"]))
            story.append(Paragraph(rec.code_snippet.replace("\n", "<br/>"), s["code"]))
            story.append(Spacer(1, 8))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceBefore=12))
    story.append(Paragraph(
        "SentinelLLM | OWASP LLM Top 10 | MITRE ATLAS Aligned | Generated by AI Security Testing Framework",
        ParagraphStyle("footer", fontSize=7, textColor=C_GREY, alignment=TA_CENTER)
    ))

    doc.build(story)
    return output_path
