"""
Convert FINAL_REPORT.md into a PDF (FINAL_REPORT.pdf) using ReportLab Platypus.

Handles: headings, paragraphs with **bold**/`code`, bullet & numbered lists,
fenced code blocks, blockquotes, and pipe tables.

Usage:  python3 tools_make_report_pdf.py
"""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted,
                                Table, TableStyle, ListFlowable, ListItem)

SRC = "FINAL_REPORT.md"
OUT = "FINAL_REPORT.pdf"

ss = getSampleStyleSheet()
styles = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontSize=17, leading=21, alignment=TA_CENTER),
    "h1":    ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=6,
                            textColor=colors.HexColor("#1a2a4a")),
    "h2":    ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4,
                            textColor=colors.HexColor("#33415c")),
    "body":  ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.7, leading=14, alignment=TA_JUSTIFY),
    "li":    ParagraphStyle("li", parent=ss["BodyText"], fontSize=9.7, leading=13.5),
    "code":  ParagraphStyle("code", parent=ss["Code"], fontSize=8, leading=10,
                            backColor=colors.HexColor("#f4f5f7"), borderPadding=5,
                            textColor=colors.HexColor("#242930")),
    "quote": ParagraphStyle("quote", parent=ss["BodyText"], fontSize=9.3, leading=13,
                            leftIndent=12, textColor=colors.HexColor("#444444"),
                            backColor=colors.HexColor("#fafafa"), borderPadding=4),
    "cell":  ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8.2, leading=11),
    "cellh": ParagraphStyle("cellh", parent=ss["BodyText"], fontSize=8.2, leading=11, fontName="Helvetica-Bold"),
}


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.5" color="#242930">\1</font>', text)
    return text


def build():
    lines = open(SRC, encoding="utf-8").read().split("\n")
    story = []
    i, n = 0, len(lines)
    pending_list = []

    def flush_list():
        nonlocal pending_list
        if pending_list:
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(t), styles["li"]), leftIndent=12) for t in pending_list],
                bulletType="bullet", start="•", leftIndent=14))
            story.append(Spacer(1, 4))
            pending_list = []

    while i < n:
        line = lines[i]

        if line.strip().startswith("```"):
            flush_list()
            i += 1; buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            story.append(Preformatted("\n".join(buf), styles["code"]))
            story.append(Spacer(1, 6))
            continue

        if line.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[ :\-|]+\|\s*$", lines[i + 1]):
            flush_list()
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2; rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            data = [[Paragraph(inline(h), styles["cellh"]) for h in header]]
            for r in rows:
                data.append([Paragraph(inline(c), styles["cell"]) for c in r])
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c2d0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t); story.append(Spacer(1, 8))
            continue

        if line.startswith("# "):
            flush_list(); story.append(Paragraph(inline(line[2:].strip()), styles["title"])); story.append(Spacer(1, 8))
        elif line.startswith("## "):
            flush_list(); story.append(Paragraph(inline(line[3:].strip()), styles["h1"]))
        elif line.startswith("### "):
            flush_list(); story.append(Paragraph(inline(line[4:].strip()), styles["h2"]))
        elif line.strip() == "---":
            flush_list()
        elif line.strip().startswith(">"):
            flush_list(); story.append(Paragraph(inline(line.strip().lstrip(">").strip()), styles["quote"])); story.append(Spacer(1, 4))
        elif re.match(r"^\s*[-*] ", line):
            pending_list.append(re.sub(r"^\s*[-*] ", "", line))
        elif re.match(r"^\s*\d+\. ", line):
            pending_list.append(re.sub(r"^\s*\d+\. ", "", line))
        elif line.strip() == "":
            flush_list()
        else:
            flush_list()
            buf = [line]
            while (i + 1 < n and lines[i + 1].strip() != ""
                   and not re.match(r"^(#|\||```|>|\s*[-*] |\s*\d+\. |---)", lines[i + 1])):
                i += 1; buf.append(lines[i])
            story.append(Paragraph(inline(" ".join(s.strip() for s in buf)), styles["body"]))
            story.append(Spacer(1, 4))
        i += 1
    flush_list()

    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="CS460-560 Project #6 — Final Report")
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
