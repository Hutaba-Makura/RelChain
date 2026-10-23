from __future__ import annotations

from .models import Answer, RetrievalResult, StateStatus


class EvidenceExtractiveAnswerGenerator:
    """A safe offline reader that never invents claims outside retrieved evidence."""

    def generate(self, retrieval: RetrievalResult) -> Answer:
        active = [
            item for item in retrieval.states if item.status == StateStatus.ACTIVE
        ]
        uncertain = [
            item for item in retrieval.states if item.status == StateStatus.UNCERTAIN
        ]
        if active:
            facts = [
                f"{item.attribute}（{item.scope}）は「{item.value}」" for item in active
            ]
            text = "、".join(facts) + "です。"
            uncertainty = None
            if uncertain:
                uncertainty = "矛盾または根拠不足の状態候補もあります。"
        elif uncertain:
            facts = [
                f"{item.attribute}（{item.scope}）={item.value}" for item in uncertain
            ]
            text = "推定候補は " + "、".join(facts) + " です。"
            uncertainty = (
                "確定に必要な独立した根拠が不足しているか、矛盾が残っています。"
            )
        elif retrieval.evidence:
            quoted = [item.raw_text for item in retrieval.evidence[:3]]
            text = "関連する原文: " + " / ".join(quoted)
            uncertainty = "構造化されたUser Stateは見つかりませんでした。"
        else:
            text = "根拠となる記憶が見つかりませんでした。"
            uncertainty = "information_not_found"
        return Answer(
            text=text,
            evidence_turn_ids=retrieval.trace.evidence_turn_ids,
            evidence_event_ids=retrieval.trace.evidence_event_ids,
            uncertainty=uncertainty,
            retrieval=retrieval,
        )
