"""Evaluation harness.

Quantifies the behaviours that matter for a safety-critical RAG system:

1. **Retrieval quality** - hit@k and MRR against gold relevant documents.
2. **Evidence-sufficiency gate** - does it answer when it should and refuse when
   it should? (accuracy + precision/recall on the "answer" decision).
3. **Safety guardrail** - precision/recall/accuracy on emergency detection, with
   a separate breakdown for adversarial (red-team) cases and an explicit list of
   the most dangerous error: missed emergencies (false negatives).
4. **Faithfulness** - mean groundedness of generated answers vs. cited evidence.

Run:
    python -m eval.run_eval            # prints report
    python -m eval.run_eval --write    # also writes eval/RESULTS.md

The harness builds the pipeline once and reuses it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from app.config import settings
from app.guardrails import check_high_risk
from app.logging_store import ResearchLogger
from app.rag import RagPipeline
from app.sufficiency import evaluate_sufficiency

DATASETS = Path(__file__).resolve().parent / "datasets"
RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


def _load(name: str) -> dict:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and not actual:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


# --------------------------------------------------------------------------
# Individual evaluations
# --------------------------------------------------------------------------
def eval_retrieval(pipeline: RagPipeline) -> dict:
    cases = _load("retrieval_cases.json")["cases"]
    hits = 0
    reciprocal_ranks: List[float] = []
    for case in cases:
        results = pipeline._retrieve(case["question"])
        doc_ids = [r.chunk.document_id for r in results]
        relevant = set(case["relevant_docs"])
        rank = next((i + 1 for i, d in enumerate(doc_ids) if d in relevant), 0)
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return {
        "n": len(cases),
        "hit_at_k": hits / len(cases),
        "mrr": statistics.fmean(reciprocal_ranks),
        "k": settings.top_k,
    }


def eval_threshold_calibration(pipeline: RagPipeline) -> dict:
    """Top retrieval scores by gold label; justifies the sufficiency thresholds."""
    cases = _load("gate_cases.json")["cases"]
    on_topic: List[float] = []
    off_topic: List[float] = []
    for case in cases:
        results = pipeline._retrieve(case["question"])
        top = results[0].similarity_score if results else 0.0
        (on_topic if case["should_answer"] else off_topic).append(top)

    def _stats(scores: List[float]) -> dict:
        return {
            "n": len(scores),
            "min": min(scores),
            "max": max(scores),
            "mean": statistics.fmean(scores),
        }

    return {"on_topic": _stats(on_topic), "off_topic": _stats(off_topic)}


def eval_gate(pipeline: RagPipeline) -> dict:
    """Gate decision via retrieve + sufficiency (no LLM; tests the gate itself)."""
    cases = _load("gate_cases.json")["cases"]
    prf = PRF()
    mistakes: List[str] = []
    for case in cases:
        results = pipeline._retrieve(case["question"])
        decision = evaluate_sufficiency(results)
        prf.add(predicted=decision.sufficient, actual=case["should_answer"])
        if decision.sufficient != case["should_answer"]:
            mistakes.append(
                f"{'pass' if decision.sufficient else 'refuse'} (gold="
                f"{'answer' if case['should_answer'] else 'refuse'}): "
                f"{case['question']!r}"
            )
    return {"n": len(cases), "prf": prf, "mistakes": mistakes}


def eval_faithfulness(pipeline: RagPipeline) -> dict | None:
    """End-to-end groundedness on on-topic cases (requires LLM API key)."""
    if not settings.use_llm:
        return None
    scores: List[float] = []
    for case in _load("gate_cases.json")["cases"]:
        if not case["should_answer"]:
            continue
        resp = pipeline.answer(case["question"])
        if resp.groundedness:
            scores.append(resp.groundedness["score"])
    if not scores:
        return None
    return {
        "n": len(scores),
        "mean_groundedness": statistics.fmean(scores),
    }


def eval_guardrail() -> dict:
    cases = _load("guardrail_cases.json")["cases"]
    overall = PRF()
    adversarial = PRF()
    false_negatives: List[str] = []
    false_positives: List[str] = []
    for case in cases:
        predicted = check_high_risk(case["question"]).triggered
        actual = case["is_emergency"]
        overall.add(predicted, actual)
        if case.get("adversarial"):
            adversarial.add(predicted, actual)
        if actual and not predicted:
            false_negatives.append(case["question"])
        if predicted and not actual:
            false_positives.append(case["question"])
    return {
        "n": len(cases),
        "overall": overall,
        "adversarial": adversarial,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report(pipeline: RagPipeline, *, include_faithfulness: bool = True) -> str:
    retr = eval_retrieval(pipeline)
    cal = eval_threshold_calibration(pipeline)
    gate = eval_gate(pipeline)
    guard = eval_guardrail()
    faith = eval_faithfulness(pipeline) if include_faithfulness else None

    lines: List[str] = []
    lines.append("# Evaluation Results")
    lines.append("")
    lines.append(
        f"- Embedding backend: `{pipeline.embedder.backend_name}`  \n"
        f"- Generation: `{settings.llm_provider}:{settings.llm_model_name}`"
        f"{' (LLM)' if settings.use_llm else ' (NO KEY - generation disabled)'}  \n"
        f"- Indexed chunks: {pipeline.store.size}"
    )
    lines.append("")

    # Retrieval
    lines.append("## 1. Retrieval quality")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Cases | {retr['n']} |")
    lines.append(f"| hit@{retr['k']} | {_pct(retr['hit_at_k'])} |")
    lines.append(f"| MRR | {retr['mrr']:.3f} |")
    lines.append("")

    # Threshold calibration
    ot, off = cal["on_topic"], cal["off_topic"]
    lines.append("## 2. Threshold calibration (why 0.42 / 0.32)")
    lines.append("")
    lines.append("Top cosine similarity from retrieval, grouped by gold label:")
    lines.append("")
    lines.append("| Query type | Cases | Top similarity (min – max) | Mean |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| On-topic (should answer) | {ot['n']} | "
        f"{ot['min']:.3f} – {ot['max']:.3f} | {ot['mean']:.3f} |"
    )
    lines.append(
        f"| Off-topic (should refuse) | {off['n']} | "
        f"{off['min']:.3f} – {off['max']:.3f} | {off['mean']:.3f} |"
    )
    lines.append("")
    lines.append(
        f"Configured thresholds: `MIN_TOP_SIMILARITY={settings.min_top_similarity}`, "
        f"`MIN_SUPPORT_SIMILARITY={settings.min_support_similarity}`, "
        f"`MIN_SUPPORTING_CHUNKS={settings.min_supporting_chunks}`. "
        f"The 0.42 floor sits in the gap between groups "
        f"(off-topic max {off['max']:.3f} vs on-topic min {ot['min']:.3f})."
    )
    lines.append("")

    # Gate
    g: PRF = gate["prf"]
    lines.append("## 3. Evidence-sufficiency gate (answer vs. refuse)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Cases | {gate['n']} |")
    lines.append(f"| Accuracy | {_pct(g.accuracy)} |")
    lines.append(f"| Precision (answer) | {_pct(g.precision)} |")
    lines.append(f"| Recall (answer) | {_pct(g.recall)} |")
    lines.append(f"| F1 | {g.f1:.3f} |")
    lines.append(
        f"| Confusion | TP={g.tp} FP={g.fp} TN={g.tn} FN={g.fn} |"
    )
    lines.append("")
    if gate["mistakes"]:
        lines.append("Misclassifications:")
        for m in gate["mistakes"]:
            lines.append(f"- {m}")
    else:
        lines.append("No misclassifications. ✅")
    lines.append("")

    # Guardrail
    o: PRF = guard["overall"]
    a: PRF = guard["adversarial"]
    lines.append("## 4. Safety guardrail (emergency detection)")
    lines.append("")
    lines.append("| Metric | Overall | Adversarial subset |")
    lines.append("|---|---|---|")
    lines.append(f"| Cases | {o.total} | {a.total} |")
    lines.append(f"| Accuracy | {_pct(o.accuracy)} | {_pct(a.accuracy)} |")
    lines.append(f"| Precision | {_pct(o.precision)} | {_pct(a.precision)} |")
    lines.append(f"| Recall (catch emergencies) | {_pct(o.recall)} | {_pct(a.recall)} |")
    lines.append(f"| F1 | {o.f1:.3f} | {a.f1:.3f} |")
    lines.append("")
    lines.append(
        f"**Missed emergencies (false negatives - critical): "
        f"{len(guard['false_negatives'])}**"
    )
    for q in guard["false_negatives"]:
        lines.append(f"- ❌ {q!r}")
    lines.append("")
    lines.append(
        f"False positives (over-triggered, safe but reduces precision): "
        f"{len(guard['false_positives'])}"
    )
    for q in guard["false_positives"]:
        lines.append(f"- ⚠️ {q!r}")
    lines.append("")

    # Faithfulness
    lines.append("## 5. Faithfulness / groundedness")
    lines.append("")
    if faith:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| On-topic cases scored | {faith['n']} |")
        lines.append(f"| Mean groundedness | {faith['mean_groundedness']:.3f} |")
        lines.append("")
        lines.append(
            "_Groundedness = mean per-sentence max cosine similarity of the answer "
            "to its cited evidence (offline entailment proxy)._"
        )
    else:
        lines.append("_Skipped (no LLM API key configured)._")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write eval/RESULTS.md")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: skip faithfulness (no LLM key required)",
    )
    args = parser.parse_args()

    tmp = Path(__file__).resolve().parent / "_eval_run_log"
    logger = ResearchLogger(
        jsonl_path=tmp.with_suffix(".jsonl"), sqlite_path=tmp.with_suffix(".db")
    )
    pipeline = RagPipeline(logger=logger)
    report = build_report(pipeline, include_faithfulness=not args.ci)
    print(report)
    if args.write:
        RESULTS_PATH.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
