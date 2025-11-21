#!/usr/bin/env python3
"""
Simple PDF → Markdown converter using PyMuPDF (fitz).

Usage:
  python tools/pdf_to_md.py /absolute/path/to/file.pdf [-o output.md]

Notes:
  - Requires the `pymupdf` package (import name: `fitz`).
  - This script extracts text per page and wraps each page with a heading.
  - Images, tables, and layout are not preserved beyond plain text + headings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path.as_posix())
    parts: list[str] = []
    for i, page in enumerate(doc, start=1):
        parts.append(f"\n\n## Page {i}\n\n")
        parts.append(page.get_text("text"))
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using PyMuPDF"
    )
    parser.add_argument("pdf", type=str, help="Path to input PDF")
    parser.add_argument("-o", "--output", type=str, help="Output .md path (optional)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = pdf_path.with_suffix(".md")

    md = convert_pdf_to_markdown(pdf_path)
    out_path.write_text(md, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
