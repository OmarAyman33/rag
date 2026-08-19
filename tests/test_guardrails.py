"""Tests for the guardrails package.

These tests run WITHOUT the LLM and WITHOUT a real vector store:
- G1 similarity threshold
- G2 max snippets
- G3 scope gate
- P PII detection / classification / redaction
- G5 verification helpers (with a fake judge client)
- pipeline composition (with a fake pipeline that skips heavy deps)

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.similarity import (
    RetrievedChunk,
    filter_by_threshold,
    similarity_from_distance,
)
from guardrails.max_snippets import cap_snippets
from guardrails.scope import decide_scope, ScopeDecision
from guardrails.pii import (
    detect_pii,
    classify_chunk,
    redact_text,
    pii_filter_output,
    PIIMode,
)
from guardrails.verification import (
    verify_answer,
    rebuild_from_supported,
    Verdict,
)
from guardrails.config import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def chunk(rid, text, distance, **meta):
    return RetrievedChunk(id=rid, text=text, distance=distance, metadata=meta)


def sample_chunks():
    return [
        chunk("c1", "alpha", 0.10),   # sim 0.90
        chunk("c2", "beta", 0.70),    # sim 0.30
        chunk("c3", "gamma", 0.85),   # sim 0.15 (below 0.20 threshold)
        chunk("c4", "delta", 0.95),   # sim 0.05
    ]


class FakeJudge:
    """A fake OpenAI client that returns scripted judge JSON."""

    def __init__(self, claims=None, verdicts=None):
        self._claims = claims or ["claim one", "claim two", "claim three"]
        self._verdicts = verdicts or [
            {"claim": "claim one", "supported": True},
            {"claim": "claim two", "supported": False},
            {"claim": "claim three", "supported": True},
        ]
        self.calls = []

    def __getattr__(self, name):
        if name == "responses":
            return self._responses()
        raise AttributeError(name)

    def _responses(self):
        class _Resp:
            def __init__(self, text):
                self.output_text = text

        class _Responses:
            def __init__(self, outer):
                self.outer = outer

            def create(self, model, instructions, input, **kwargs):
                prompt = input if isinstance(input, str) else str(input)
                self.outer.calls.append(prompt)
                if "Break the assistant answer" in prompt or "extract relevant" in prompt:
                    payload = {"claims": self.outer._claims}
                else:
                    payload = {"verdicts": self.outer._verdicts}
                return _Resp(json.dumps(payload))

        return _Responses(self)


def fake_settings():
    return Settings()


# ---------------------------------------------------------------------------
# G1 - similarity threshold
# ---------------------------------------------------------------------------


def test_similarity_from_distance():
    assert similarity_from_distance(0.0) == pytest.approx(1.0)
    assert similarity_from_distance(0.5) == pytest.approx(0.5)
    assert similarity_from_distance(1.0) == pytest.approx(0.0)
    # clamp
    assert similarity_from_distance(2.0) == pytest.approx(-1.0)


def test_g1_threshold_filters_low_similarity():
    kept = filter_by_threshold(sample_chunks(), 0.20)
    ids = [c.id for c in kept]
    assert ids == ["c1", "c2"]  # c3 (0.15), c4 (0.05) dropped


def test_g1_threshold_zero_keeps_everything():
    kept = filter_by_threshold(sample_chunks(), 0.0)
    assert len(kept) == 4


# ---------------------------------------------------------------------------
# G2 - max snippets
# ---------------------------------------------------------------------------


def test_g2_caps_snippets():
    chunks = sample_chunks()
    capped = cap_snippets(chunks, 2)
    assert len(capped) == 2
    assert capped[0].id == "c1"  # most relevant first


def test_g2_zero_returns_empty():
    assert cap_snippets(sample_chunks(), 0) == []


# ---------------------------------------------------------------------------
# G3 - scope gate
# ---------------------------------------------------------------------------


def test_g3_refuses_when_no_chunks():
    decision = decide_scope([], 0.20)
    assert decision.refused
    assert not decision.allowed


def test_g3_allows_when_chunks_exist():
    decision = decide_scope(sample_chunks()[:1], 0.20)
    assert decision.allowed
    assert not decision.refused


# ---------------------------------------------------------------------------
# P - PII detection / classification
# ---------------------------------------------------------------------------


def test_pii_detects_email():
    hits = detect_pii("Contact me at test.user@example.com please")
    assert any(h.entity_type == "EMAIL" for h in hits)


def test_pii_detects_saudi_id():
    hits = detect_pii("My national ID is 1098765432")
    assert any(h.entity_type == "SAUDI_ID" for h in hits)


def test_pii_detects_phone():
    hits = detect_pii("Call me on 0551234567")
    assert any(h.entity_type == "PHONE" for h in hits)


def test_pii_clean_chunk():
    cls = classify_chunk("The sky is blue and the grass is green.", mode="block")
    assert cls.action == "clean"


def test_pii_blocks_fundamentally_personal():
    cls = classify_chunk(
        "Name: Ahmed Ali. ID: 1098765432. Phone: 0551234567. Email: a@b.com",
        mode="block",
    )
    assert cls.action == "block"


def test_pii_redacts_incidental_identifier():
    cls = classify_chunk(
        "The contract was signed by a@b.com on Monday.",
        mode="block",
    )
    assert cls.action == "redact"
    assert "a@b.com" not in cls.redacted_text
    assert "[REDACTED]" in cls.redacted_text


def test_pii_report_mode_never_blocks():
    cls = classify_chunk(
        "Name: Ahmed Ali. ID: 1098765432. Phone: 0551234567.",
        mode="report",
    )
    assert cls.action == "report"


def test_pii_output_gate_redacts():
    cleaned, hits = pii_filter_output("Reach out to me at omar@test.com", mode="redact")
    assert hits
    assert "omar@test.com" not in cleaned


def test_redact_text_masks_all():
    text = "email: a@b.com phone: 0551234567"
    out, hits = redact_text(text)
    assert len(hits) == 2
    assert "a@b.com" not in out
    assert "0551234567" not in out


# ---------------------------------------------------------------------------
# G5 - verification (fake judge)
# ---------------------------------------------------------------------------


def test_verify_answer_returns_verdict():
    settings = fake_settings()
    verdict = verify_answer(FakeJudge(), settings, "Answer text here", "context")
    assert isinstance(verdict, Verdict)
    assert verdict.claims == ["claim one", "claim two", "claim three"]
    assert verdict.supported == ["claim one", "claim three"]
    assert verdict.unsupported == ["claim two"]
    assert verdict.faithfulness == pytest.approx(2 / 3)


def test_verify_answer_faithfulness_one_when_all_supported():
    judge = FakeJudge(
        verdicts=[
            {"claim": "claim one", "supported": True},
            {"claim": "claim two", "supported": True},
            {"claim": "claim three", "supported": True},
        ]
    )
    verdict = verify_answer(judge, fake_settings(), "Answer", "context")
    assert verdict.faithfulness == pytest.approx(1.0)
    assert verdict.fully_grounded


def test_rebuild_from_supported():
    v = Verdict(claims=["a", "b"], supported=["a"], unsupported=["b"], faithfulness=0.5)
    out = rebuild_from_supported(v)
    assert out is not None
    assert "a" in out
    assert "b" not in out


def test_rebuild_none_when_nothing_supported():
    v = Verdict(claims=["a"], supported=[], unsupported=["a"], faithfulness=0.0)
    assert rebuild_from_supported(v) is None


# ---------------------------------------------------------------------------
# Pipeline composition (no heavy deps / no LLM)
# ---------------------------------------------------------------------------


class FakePipeline:
    """Minimal stand-in mirroring GuardedPipeline.run()'s control flow."""

    def __init__(self, settings):
        self.settings = settings

    def run(self, question):
        # Emulate: retrieve -> G1 -> G3 -> G2 -> generate -> G5 -> P
        all_chunks = sample_chunks()
        kept = filter_by_threshold(all_chunks, self.settings.similarity_threshold)
        scope = decide_scope(kept, self.settings.similarity_threshold)
        if scope.refused:
            return {
                "refused": True,
                "reason": scope.reason,
                "dropped": len(all_chunks) - len(kept),
            }
        final = cap_snippets(kept, self.settings.max_snippets)
        return {
            "refused": False,
            "chunks": [c.id for c in final],
            "dropped": len(all_chunks) - len(kept),
        }


def test_pipeline_flow_threshold_and_cap():
    settings = Settings(similarity_threshold=0.20, max_snippets=2)
    out = FakePipeline(settings).run("q")
    assert not out["refused"]
    assert out["chunks"] == ["c1", "c2"]
    assert out["dropped"] == 2  # c3, c4


def test_pipeline_flow_refuses_when_threshold_high():
    settings = Settings(similarity_threshold=0.99, max_snippets=5)
    out = FakePipeline(settings).run("q")
    assert out["refused"]