"""P - Personal information blocker.

Detects personal information and stops it before it reaches the vector store
(ingestion gate), before it reaches the LLM (query gate), and before it reaches
the user (output gate).

Implementation:
  * Deterministic regex recognizers (always available, no model download):
    EMAIL, PHONE (incl. Saudi mobile 05xxxxxxxx), SAUDI_ID/IQAMA (10 digits
    starting with 1 or 2), CREDIT_CARD (13-19 digits), IP_ADDRESS.
  * Optional Microsoft Presidio enhancement (if presidio-analyzer is
    installed) for NLP/NER-based detection of names and other entities.

The user-facing policy (Sovereign-approved): REDACT chunks with a few
identifiers; BLOCK chunks that are fundamentally personal (dense identifiers
or multiple high-severity types in one chunk). 'report' mode only logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Recognizers (deterministic, local, no model download)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:05\d{8}|0\d{9}|5\d{8})\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("SAUDI_ID", re.compile(r"\b[12]\d{9}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

# High-severity types: a chunk containing these is "fundamentally personal".
_HIGH_SEVERITY = {"SAUDI_ID", "PHONE", "CREDIT_CARD", "EMAIL"}


@dataclass
class PIIHit:
    """A detected piece of personal information."""

    entity_type: str
    start: int
    end: int
    text: str
    score: float = 1.0
    source: str = "regex"


class PIIMode(str, Enum):
    """How the ingestion gate treats personal information."""

    BLOCK = "block"   # fundamentally personal chunks are skipped entirely
    REDACT = "redact"  # identifiers are masked, chunk still indexed
    REPORT = "report"  # only log, do not change anything


@dataclass
class PIIClassification:
    """Result of classifying one chunk."""

    action: str          # "block" | "redact" | "clean"
    hits: list[PIIHit] = field(default_factory=list)
    redacted_text: str = ""
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"


def detect_pii(text: str) -> list[PIIHit]:
    """Detect personal information using local deterministic recognizers."""
    hits: list[PIIHit] = []
    for entity_type, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            hits.append(
                PIIHit(
                    entity_type=entity_type,
                    start=m.start(),
                    end=m.end(),
                    text=m.group(0),
                    source="regex",
                )
            )
    # Optional Presidio enhancement (if installed) - adds NER-based hits.
    hits = _enhance_with_presidio(text, hits)
    # Sort by position for clean redaction.
    hits.sort(key=lambda h: (h.start, h.end))
    return hits


def _enhance_with_presidio(text: str, hits: list[PIIHit]) -> list[PIIHit]:
    """Merge Presidio hits if available; otherwise return regex hits only."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
    except Exception:
        return hits

    try:
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        existing = {(h.entity_type, h.start, h.end) for h in hits}
        for r in results:
            if (r.entity_type, r.start, r.end) not in existing:
                hits.append(
                    PIIHit(
                        entity_type=r.entity_type,
                        start=r.start,
                        end=r.end,
                        text=text[r.start : r.end],
                        score=r.score,
                        source="presidio",
                    )
                )
    except Exception:
        # Presidio may need a spaCy model it cannot download offline; never
        # let the optional enhancer break the deterministic gate.
        pass
    hits.sort(key=lambda h: (h.start, h.end))
    return hits


def redact_text(text: str, placeholder: str = "[REDACTED]") -> tuple[str, list[PIIHit]]:
    """Replace all detected identifiers with a placeholder."""
    hits = detect_pii(text)
    if not hits:
        return text, hits
    # Redact from the end so offsets stay valid.
    pieces = list(text)
    # Simplest robust approach: build output by walking hits in reverse.
    out = text
    for h in reversed(hits):
        out = out[: h.start] + placeholder + out[h.end :]
    return out, hits


def _is_fundamentally_personal(hits: list[PIIHit], text: str) -> tuple[bool, str]:
    """Heuristic: is this chunk fundamentally personal (block) or incidental (redact)?

    Block if:
      - >= 2 distinct HIGH_SEVERITY entity types appear together
      - a single high-severity entity is very dense (>= 2 occurrences)
      - identifiers cover a large share of the chunk
    Otherwise the chunk only needs redaction.
    """
    if not hits:
        return False, ""

    high_severity_types = {h.entity_type for h in hits if h.entity_type in _HIGH_SEVERITY}
    if len(high_severity_types) >= 2:
        return True, f"multiple high-severity identifiers: {sorted(high_severity_types)}"

    high_hits = [h for h in hits if h.entity_type in _HIGH_SEVERITY]
    if len(high_hits) >= 2:
        return True, f"dense identifiers ({len(high_hits)} hits)"

    covered = sum(h.end - h.start for h in hits)
    if len(text) > 0 and covered / len(text) > 0.3:
        return True, "identifiers cover >30% of chunk"

    return False, ""


def classify_chunk(text: str, mode: str | PIIMode = "block") -> PIIClassification:
    """Classify one chunk: block | redact | clean."""
    mode = PIIMode(mode)
    hits = detect_pii(text)
    if not hits:
        return PIIClassification(action="clean", hits=[], redacted_text=text)

    if mode == PIIMode.REPORT:
        return PIIClassification(
            action="report",
            hits=hits,
            redacted_text=text,
            reason="report mode: PII logged only",
        )

    fundamental, reason = _is_fundamentally_personal(hits, text)

    if mode == PIIMode.BLOCK and fundamental:
        return PIIClassification(
            action="block", hits=hits, reason=f"blocked: {reason}"
        )

    # REDACT (default when not blocking): mask identifiers, keep the chunk.
    redacted, _ = redact_text(text)
    return PIIClassification(
        action="redact",
        hits=hits,
        redacted_text=redacted,
        reason="redacted: identifiers masked",
    )


def pii_filter_output(
    answer: str, mode: str | PIIMode = "block", placeholder: str = "[REDACTED]"
) -> tuple[str, list[PIIHit]]:
    """Output gate: redact any personal information before showing to the user.

    The output gate always redacts (never blocks) - an answer reaching the user
    must not contain identifiers, even if the source chunk passed ingestion.
    """
    return redact_text(answer, placeholder)