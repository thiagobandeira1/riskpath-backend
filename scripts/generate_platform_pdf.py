"""Generate a NotebookLM-friendly PDF from docs/RISKPATH_PLATFORM_GUIDE.md.

Why this exists:
    The user wants to feed the platform documentation into NotebookLM
    (which then auto-generates an explainer video). NotebookLM ingests
    PDFs cleanly. Rather than wrestle with pandoc or browser-print
    pipelines, we generate the PDF directly from a single source-of-
    truth markdown file using reportlab's Platypus flowables.

    The PDF is intentionally simple: serif body type, conservative
    heading sizes, generous line spacing. NotebookLM cares about
    structure and prose, not visual flourishes.

Usage:
    python scripts/generate_platform_pdf.py
    # writes docs/RISKPATH_PLATFORM_GUIDE.pdf
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ───────────────────────────────────────────────────────────── paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MD_SRC = _REPO_ROOT / "docs" / "RISKPATH_PLATFORM_GUIDE.md"
_PDF_OUT = _REPO_ROOT / "docs" / "RISKPATH_PLATFORM_GUIDE.pdf"

# ───────────────────────────────────────────────────────────── styles
_BODY_FONT = "Helvetica"
_BODY_FONT_BOLD = "Helvetica-Bold"
_BODY_FONT_ITALIC = "Helvetica-Oblique"
_MONO_FONT = "Courier"

_INK = HexColor("#1A1A1A")
_MUTED = HexColor("#555555")
_ACCENT = HexColor("#5B21B6")  # violet — matches the platform's purple

_STYLES = {
    "h1": ParagraphStyle(
        "H1",
        fontName=_BODY_FONT_BOLD,
        fontSize=22,
        leading=26,
        spaceBefore=18,
        spaceAfter=10,
        textColor=_ACCENT,
        alignment=TA_LEFT,
    ),
    "h2": ParagraphStyle(
        "H2",
        fontName=_BODY_FONT_BOLD,
        fontSize=16,
        leading=20,
        spaceBefore=18,
        spaceAfter=8,
        textColor=_ACCENT,
    ),
    "h3": ParagraphStyle(
        "H3",
        fontName=_BODY_FONT_BOLD,
        fontSize=13,
        leading=17,
        spaceBefore=14,
        spaceAfter=6,
        textColor=_INK,
    ),
    "h4": ParagraphStyle(
        "H4",
        fontName=_BODY_FONT_BOLD,
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
        textColor=_INK,
    ),
    "body": ParagraphStyle(
        "Body",
        fontName=_BODY_FONT,
        fontSize=10.5,
        leading=15,
        spaceAfter=8,
        textColor=_INK,
    ),
    "subtitle": ParagraphStyle(
        "Subtitle",
        fontName=_BODY_FONT_ITALIC,
        fontSize=12,
        leading=16,
        spaceAfter=6,
        textColor=_MUTED,
    ),
    "meta": ParagraphStyle(
        "Meta",
        fontName=_BODY_FONT_ITALIC,
        fontSize=9,
        leading=12,
        spaceAfter=14,
        textColor=_MUTED,
    ),
    "bullet": ParagraphStyle(
        "Bullet",
        fontName=_BODY_FONT,
        fontSize=10.5,
        leading=14,
        leftIndent=12,
        spaceAfter=2,
        textColor=_INK,
    ),
    # Blockquote — used for the verbatim "SAY" talking points the presenter
    # reads aloud. Tinted background so they pop on the page and are easy to
    # find at a glance during the demo.
    "quote": ParagraphStyle(
        "Quote",
        fontName=_BODY_FONT_ITALIC,
        fontSize=11,
        leading=16,
        leftIndent=14,
        rightIndent=10,
        spaceBefore=4,
        spaceAfter=8,
        textColor=HexColor("#2A1A4A"),
        backColor=HexColor("#F4F0FA"),
        borderPadding=(6, 8, 6, 10),
    ),
}


# ───────────────────────────────────────────────────────────── inline formatting
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(text: str) -> str:
    """Convert a single markdown line's inline syntax to reportlab markup."""
    out = html.escape(text)
    # Re-unescape ampersand inside URLs handled below; reportlab uses HTML entities.
    out = _CODE_RE.sub(lambda m: f'<font name="{_MONO_FONT}">{m.group(1)}</font>', out)
    out = _BOLD_RE.sub(r"<b>\1</b>", out)
    out = _ITALIC_RE.sub(r"<i>\1</i>", out)

    def _link_repl(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        return f'<link href="{url}" color="#5B21B6"><u>{label}</u></link>'

    out = _LINK_RE.sub(_link_repl, out)
    return out


# ───────────────────────────────────────────────────────────── markdown parser
def _parse_table(lines: list[str], start: int) -> tuple[Table | None, int]:
    """Parse a GitHub-flavoured markdown table starting at lines[start].

    Returns the rendered Table and the index AFTER the table, or (None, start)
    if no table is found.
    """
    if start >= len(lines) or "|" not in lines[start]:
        return None, start
    if start + 1 >= len(lines) or not re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", lines[start + 1]):
        return None, start

    def _split(row: str) -> list[str]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        return cells

    header = _split(lines[start])
    rows: list[list[str]] = [header]
    i = start + 2
    while i < len(lines) and "|" in lines[i] and lines[i].strip():
        rows.append(_split(lines[i]))
        i += 1

    # Build cell paragraphs with inline formatting.
    cell_style = ParagraphStyle(
        "TableCell",
        fontName=_BODY_FONT,
        fontSize=9.5,
        leading=12,
        textColor=_INK,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        fontName=_BODY_FONT_BOLD,
        fontSize=9.5,
        leading=12,
        textColor=_INK,
    )
    data = [
        [Paragraph(_inline(c), header_style if r_idx == 0 else cell_style) for c in row]
        for r_idx, row in enumerate(rows)
    ]
    tbl = Table(data, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F4F0FA")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, _ACCENT),
                ("LINEBELOW", (0, -1), (-1, -1), 0.25, HexColor("#E5E5E5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#FAFAFA")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl, i


def _is_block_marker(line: str) -> bool:
    """True if a line starts a non-paragraph block (heading, list, table, HR, quote).

    Critically distinguishes `* ` (bullet) from `**bold**` (emphasis at line start)
    and `- ` (bullet) from in-text dashes — checking only the bare leading char
    triggers an infinite loop on bold-prefixed paragraphs like
    `**Hospital teams** use the platform...`.
    """
    s = line.lstrip()
    if not s:
        return True
    if s.startswith(("# ", "## ", "### ", "#### ", "##### ")):
        return True
    if s.startswith(("- ", "* ", "+ ")):
        return True
    if s.startswith("|"):
        return True
    if s.startswith("---"):
        return True
    if s.startswith("> "):
        return True
    if re.match(r"^\d+\.\s", s):
        return True
    return False


def _collect_blockquote(lines: list[str], start: int) -> tuple[str, int]:
    """Collect a `> `-prefixed blockquote (the verbatim 'SAY' talking points)."""
    buf: list[str] = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("> "):
        buf.append(lines[i].lstrip()[2:].rstrip())
        i += 1
    # blank line(s) inside a quote -> stop; advance at least one to be safe
    if i == start:
        i = start + 1
    return " ".join(buf), i


def _collect_paragraph(lines: list[str], start: int) -> tuple[str, int]:
    """Collect a paragraph (consecutive non-empty, non-special lines)."""
    buf: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip() and not _is_block_marker(lines[i]):
        buf.append(lines[i].strip())
        i += 1
    # SAFETY: never return start == i on a non-blank line, or the outer loop
    # would spin forever. If we hit a block marker on the first line, advance
    # past it so the outer loop's next iteration picks the right handler.
    if i == start:
        i = start + 1
    return " ".join(buf), i


def _collect_bullets(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect consecutive bullet items."""
    items: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith(("- ", "* ")):
            items.append(stripped[2:].rstrip())
            i += 1
        elif lines[i].startswith(("  ", "\t")) and items:
            # continuation line for the previous item
            items[-1] += " " + lines[i].strip()
            i += 1
        else:
            break
    return items, i


def _collect_numbered(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect consecutive numbered-list items."""
    items: list[str] = []
    i = start
    pat = re.compile(r"^(\d+)\.\s+(.*)$")
    while i < len(lines):
        m = pat.match(lines[i].lstrip())
        if m:
            items.append(m.group(2).rstrip())
            i += 1
        elif lines[i].startswith(("  ", "\t")) and items:
            items[-1] += " " + lines[i].strip()
            i += 1
        else:
            break
    return items, i


def _parse_markdown(md_text: str) -> list:
    """Convert markdown text into a list of reportlab flowables."""
    lines = md_text.replace("\r\n", "\n").split("\n")
    flow: list = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # blank
        if not line.strip():
            i += 1
            continue

        # horizontal rule
        if line.strip() == "---":
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#E5E5E5"), spaceBefore=2, spaceAfter=6))
            i += 1
            continue

        # heading 1
        if line.startswith("# "):
            flow.append(Paragraph(_inline(line[2:].strip()), _STYLES["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            flow.append(Paragraph(_inline(line[3:].strip()), _STYLES["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            flow.append(Paragraph(_inline(line[4:].strip()), _STYLES["h3"]))
            i += 1
            continue
        if line.startswith("#### "):
            flow.append(Paragraph(_inline(line[5:].strip()), _STYLES["h4"]))
            i += 1
            continue

        # subtitle-as-bold paragraph immediately after H1
        # (we treat **...** on its own line as a subtitle)
        if (
            line.strip().startswith("**")
            and line.strip().endswith("**")
            and len(line.strip()) > 4
            and not line.strip().startswith("**bold**")
        ):
            inner = line.strip()[2:-2]
            flow.append(Paragraph(inner, _STYLES["subtitle"]))
            i += 1
            continue

        # italic-only line (meta)
        if line.strip().startswith("*") and line.strip().endswith("*") and "*" not in line.strip()[1:-1]:
            inner = line.strip()[1:-1]
            flow.append(Paragraph(inner, _STYLES["meta"]))
            i += 1
            continue

        # blockquote (the verbatim "SAY" talking points)
        if line.lstrip().startswith("> "):
            quote, new_i = _collect_blockquote(lines, i)
            flow.append(Paragraph(_inline(quote), _STYLES["quote"]))
            i = new_i
            continue

        # table
        tbl, new_i = _parse_table(lines, i)
        if tbl is not None:
            flow.append(KeepTogether([Spacer(1, 4), tbl, Spacer(1, 8)]))
            i = new_i
            continue

        # bullet list
        if line.lstrip().startswith(("- ", "* ")):
            items, new_i = _collect_bullets(lines, i)
            lf = ListFlowable(
                [ListItem(Paragraph(_inline(it), _STYLES["bullet"]), leftIndent=12) for it in items],
                bulletType="bullet",
                start="•",
                leftIndent=14,
                bulletFontName=_BODY_FONT,
                bulletFontSize=10,
            )
            flow.append(lf)
            flow.append(Spacer(1, 4))
            i = new_i
            continue

        # numbered list
        if re.match(r"^\d+\.\s+", line.lstrip()):
            items, new_i = _collect_numbered(lines, i)
            lf = ListFlowable(
                [ListItem(Paragraph(_inline(it), _STYLES["bullet"]), leftIndent=14) for it in items],
                bulletType="1",
                leftIndent=18,
                bulletFontName=_BODY_FONT,
                bulletFontSize=10,
            )
            flow.append(lf)
            flow.append(Spacer(1, 4))
            i = new_i
            continue

        # ordinary paragraph (collect continuation lines)
        para, new_i = _collect_paragraph(lines, i)
        if para:
            flow.append(Paragraph(_inline(para), _STYLES["body"]))
        i = new_i

    return flow


# ───────────────────────────────────────────────────────────── page chrome
_FOOTER_TEXT = "RiskPath Platform Guide · v0.1.0"


def _make_on_page(footer: str):
    def _on_page(canvas, doc):  # noqa: ANN001
        canvas.saveState()
        canvas.setFont(_BODY_FONT, 8)
        canvas.setFillColor(_MUTED)
        canvas.drawString(0.75 * inch, 0.5 * inch, footer)
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    return _on_page


# ───────────────────────────────────────────────────────────── main
def render_pdf(md_path: Path, pdf_path: Path, *, title: str, footer: str) -> None:
    """Render a markdown file to a PDF."""
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown source missing at {md_path}")

    md = md_path.read_text(encoding="utf-8")
    flowables = _parse_markdown(md)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.9 * inch,
        title=title,
        author="RiskPath",
        subject="RiskPath — Clinical Decision-Support Platform",
    )
    on_page = _make_on_page(footer)
    doc.build(flowables, onFirstPage=on_page, onLaterPages=on_page)
    size_kb = pdf_path.stat().st_size / 1024
    try:
        rel = pdf_path.relative_to(_REPO_ROOT)
    except ValueError:
        rel = pdf_path
    print(f"Wrote {rel} ({size_kb:.1f} KB)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render a markdown doc to PDF.")
    parser.add_argument(
        "--input", "-i", type=Path, default=_MD_SRC,
        help="Input markdown file (default: docs/RISKPATH_PLATFORM_GUIDE.md)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output PDF path (default: input path with .pdf extension)",
    )
    parser.add_argument(
        "--title", "-t", type=str, default="RiskPath Platform Guide",
        help="PDF document title (metadata + footer)",
    )
    args = parser.parse_args()

    out = args.output or args.input.with_suffix(".pdf")
    footer = f"{args.title} · v0.1.0" if args.input == _MD_SRC else args.title
    render_pdf(args.input, out, title=args.title, footer=footer)


if __name__ == "__main__":
    main()
