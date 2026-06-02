"""Contract text parser that preserves legal section boundaries.

The parser deliberately keeps the structure simple and deterministic: section
headings in the corpus follow a predictable numbered pattern, and preserving
those sections gives the RAG/citation layers stable references such as
`[TC-1001, Section 8.1]`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.documents.metadata_extractor import extract_metadata


SECTION_HEADING = re.compile(r"^(\d+)\.\s+([A-Z][A-Z0-9 /&-]+)$")


@dataclass(slots=True)
class ParsedSection:
    section_number: str
    section_title: str
    text: str
    line_start: int
    line_end: int


@dataclass(slots=True)
class ParsedContract:
    filename: str
    raw_text: str
    metadata: dict[str, object]
    sections: list[ParsedSection]


def parse_contract_file(path: Path) -> ParsedContract:
    """Parse a contract file from disk using UTF-8 text."""

    raw_text = path.read_text(encoding="utf-8")
    return parse_contract_text(raw_text, path.name)


def parse_contract_text(raw_text: str, filename: str = "uploaded.txt") -> ParsedContract:
    """Split raw contract text into metadata plus numbered sections.

    Section 0 is reserved for the contract header. That header often contains
    parties, contract ID, title, and contact information, so we keep it
    searchable and citable instead of dropping it.
    """

    lines = raw_text.splitlines()
    section_starts: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines, start=1):
        match = SECTION_HEADING.match(line.strip())
        if match:
            section_starts.append((index, match))

    sections: list[ParsedSection] = []
    if section_starts and section_starts[0][0] > 1:
        header_end = section_starts[0][0] - 1
        header_text = "\n".join(lines[:header_end]).strip()
        if header_text:
            sections.append(
                ParsedSection(
                    section_number="0",
                    section_title="Contract Header",
                    text=header_text,
                    line_start=1,
                    line_end=header_end,
                )
            )

    for position, (line_number, match) in enumerate(section_starts):
        next_line_number = (
            section_starts[position + 1][0] if position + 1 < len(section_starts) else len(lines) + 1
        )
        section_lines = lines[line_number - 1 : next_line_number - 1]
        sections.append(
            ParsedSection(
                section_number=match.group(1),
                section_title=match.group(2).strip().title(),
                text="\n".join(section_lines).strip(),
                line_start=line_number,
                line_end=next_line_number - 1,
            )
        )

    return ParsedContract(
        filename=filename,
        raw_text=raw_text,
        metadata=extract_metadata(raw_text, filename),
        sections=sections,
    )
