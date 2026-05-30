"""Safety guardrails for urgent / high-risk queries.

The question is screened for emergency red flags before any retrieval or
generation; a match returns a fixed escalation message instead of advice.
Detection uses curated regexes (deterministic and auditable). Tuned to favour
recall: a false positive is far safer than missing a real emergency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Each pattern targets an emergency red flag described in the safety guidance.
# Patterns are intentionally specific (e.g. "severe" / "sudden" qualifiers,
# negations like "can't breathe") to reduce false positives on educational
# questions that merely mention a symptom word.
_HIGH_RISK_PATTERNS: List[tuple[str, str]] = [
    # "chest" within a short window of a pain/pressure word, in EITHER order, so
    # we catch "chest pain", "chest feels tight", and "crushing pressure ... chest".
    ("chest pain/pressure", r"\bchest\b.{0,25}\b(pain|pressure|tight|tightness|tightening|hurt\w*|discomfort|heavy|heaviness|squeez\w*|crush\w*)"),
    ("pain/pressure in chest", r"\b(pain|pressure|tight|tightness|hurt\w*|discomfort|heavy|heaviness|squeez\w*|crush\w*)\b.{0,25}\bchest\b"),
    ("heart attack", r"\bheart attack\b"),
    ("severe shortness of breath", r"\b(severe|sudden|extreme|suddenly)\b.{0,25}\b(short(ness)? of breath|breathing|breathe|breath|breathless)\b"),
    ("cannot breathe", r"\b(can'?t|cannot|can not|unable to|struggling to|hard to|difficult to|catch my breath)\b.{0,12}\bbreathe?\b"),
    ("painful breathing", r"\b(hurt\w*|painful|hard|difficult|struggling)\b.{0,12}\bbreath"),
    ("gasping for air", r"\bgasping (for )?(air|breath)\b"),
    ("fainting", r"\b(faint(ed|ing|s)?|pass(ed|ing) out|passed out|blacked out|loss of consciousness|unconscious|collaps\w*)\b"),
    ("severe dizziness", r"\b(severe|sudden|extreme|suddenly)\b.{0,15}\b(dizz|lightheaded|light-headed)\b"),
    ("stroke signs", r"\b(stroke|sudden weakness|arm weakness|numb(ness)? on one side)\b|\bface\b.{0,12}\bdrooping\b|\b(slurred|slurring)\b.{0,12}\bspeech\b|\bspeech\b.{0,12}\b(slurred|slurring)\b"),
    ("coughing blood", r"\bcough(ing)? up (blood|pink|frothy)\b"),
    ("emergency", r"\b(emergency|9-?1-?1|call an ambulance|life[- ]threatening)\b"),
    ("dangerous palpitations", r"\b(rapid|irregular|racing|pounding)\b.{0,20}\b(heart ?beat|heart|pulse|palpitations)\b.{0,40}\b(faint|pass(ed|ing)? out|collaps\w*|chest|breath)\b|\b(heart (is )?racing)\b.{0,40}\b(faint|pass(ed|ing)? out|collaps\w*)\b"),
    ("self-harm / suicidal", r"\b(suicid\w*|kill(ing)? myself|end(ing)? my life|harm(ing)? (myself|my self)|hurt(ing)? myself|self[- ]harm)\b"),
]

ESCALATION_MESSAGE = (
    "Your message mentions symptoms that can be signs of a medical emergency. "
    "I can't provide medical advice for this. If you are experiencing chest pain "
    "or pressure, severe or sudden shortness of breath, fainting, severe "
    "dizziness, signs of a stroke (face drooping, arm weakness, slurred speech), "
    "or any symptom that feels life-threatening, call your local emergency number "
    "(such as 911) or go to the nearest emergency department now. If you are not "
    "in immediate danger but are worried, contact your healthcare team right away. "
    "If you are in crisis or thinking about harming yourself, please contact your "
    "local emergency services or a crisis line immediately."
)


@dataclass
class GuardrailResult:
    triggered: bool
    matched_categories: List[str]
    message: Optional[str] = None


def check_high_risk(question: str) -> GuardrailResult:
    """Return whether the question contains emergency red flags."""
    text = question.lower()
    matched: List[str] = []
    for label, pattern in _HIGH_RISK_PATTERNS:
        if re.search(pattern, text):
            matched.append(label)

    if matched:
        return GuardrailResult(
            triggered=True,
            matched_categories=matched,
            message=ESCALATION_MESSAGE,
        )
    return GuardrailResult(triggered=False, matched_categories=[])
