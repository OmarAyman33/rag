"""G5 - Post-generation verification (Luna as judge).

The model generates an answer from the retrieved context. Before the user sees
it, we verify it:

  1. DECOMPOSE the answer into atomic claims (judge prompt).
  2. VERIFY each claim against the retrieved context (NLI-style judgment).
  3. SCORE faithfulness = supported claims / total claims.
  4. ACT: keep only supported claims; if nothing survives -> refuse.

This is the RAGAS faithfulness procedure, run as a runtime gate with the
same model (Luna, as requested) acting as judge. Verification is easier than
generation, so the same model is an acceptable dev-time judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import Settings

# Judge prompts ---------------------------------------------------------------

_EXTRACT_PROMPT = """You are a strict fact-checking judge.
Break the assistant answer below into a list of ATOMIC, standalone factual claims.
Each claim must express exactly ONE assertion, with pronouns resolved.
Return ONLY a JSON object of the form {{"claims": ["claim 1", "claim 2", ...]}}.
Do not add anything outside the JSON.

ANSWER:
{answer}
"""

_VERIFY_PROMPT = """You are a strict fact-checking judge.
For each claim, decide whether it can be DIRECTLY inferred from the provided
context passages. A claim is supported ONLY if the context explicitly states
it. If the context is silent or contradicts the claim, it is NOT supported.
Return ONLY a JSON object of the form:
{{"verdicts": [{{"claim": "...", "supported": true|false}}, ...]}}

CONTEXT:
{context}

CLAIMS:
{claims}
"""


@dataclass
class Verdict:
    """Verification result for one answer."""

    claims: list[str] = field(default_factory=list)
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    faithfulness: float = 1.0
    raw: dict = field(default_factory=dict)

    @property
    def fully_grounded(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict:
        return {
            "claims": self.claims,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "faithfulness": round(self.faithfulness, 4),
        }


class VerificationError(Exception):
    """Raised when the judge fails to produce parseable output."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict:
    text = _strip_code_fences(text)
    # Try strict first, then find the first balanced {...} block.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise VerificationError("no JSON object found in judge output")
        return json.loads(match.group(0))


def _call_judge(client, model: str, system: str, user: str, max_tokens: int) -> str:
    """Call the OpenAI Responses API exactly like the original test_query.py."""
    response = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
    )
    return response.output_text


def _extract_claims(client, settings: Settings, answer: str) -> list[str]:
    system = "You are a strict fact-checking judge. You only return JSON."
    user = _EXTRACT_PROMPT.format(answer=answer)
    raw = _call_judge(client, settings.llm_model, system, user, 1024)
    data = _parse_json(raw)
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise VerificationError("judge returned malformed claims")
    return [str(c).strip() for c in claims if str(c).strip()]


def _verify_claims(client, settings: Settings, claims: list[str], context: str) -> list[bool]:
    if not claims:
        return []
    system = "You are a strict fact-checking judge. You only return JSON."
    user = _VERIFY_PROMPT.format(
        context=context,
        claims=json.dumps(claims, ensure_ascii=False),
    )
    raw = _call_judge(client, settings.llm_model, system, user, 2048)
    data = _parse_json(raw)
    verdicts = data.get("verdicts", [])
    result: list[bool] = []
    for v in verdicts:
        result.append(bool(v.get("supported", False)))
    # If the judge returned fewer verdicts than claims, treat missing as unsupported.
    while len(result) < len(claims):
        result.append(False)
    return result[: len(claims)]


def verify_answer(
    client,
    settings: Settings,
    answer: str,
    context: str,
) -> Verdict:
    """Run the full G5 verification pipeline on a generated answer.

    client: an openai.OpenAI() client (same as the original repo).
    Returns a Verdict with supported/unsupported claims and faithfulness.
    Raises VerificationError if the judge output cannot be parsed (caller may
    fall back to refusing the answer - fail closed).
    """
    claims = _extract_claims(client, settings, answer)
    if not claims:
        return Verdict(claims=[], supported=[], unsupported=[])

    flags = _verify_claims(client, settings, claims, context)

    supported = [c for c, f in zip(claims, flags) if f]
    unsupported = [c for c, f in zip(claims, flags) if not f]
    faithfulness = len(supported) / len(claims) if claims else 1.0

    return Verdict(
        claims=claims,
        supported=supported,
        unsupported=unsupported,
        faithfulness=faithfulness,
    )


def rebuild_from_supported(verdict: Verdict) -> str | None:
    """Reconstruct an answer from only the supported claims.

    Returns None when nothing is supported -> caller must refuse.
    """
    if not verdict.supported:
        return None
    return "\n".join(f"- {c}" for c in verdict.supported)