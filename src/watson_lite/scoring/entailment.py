from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence

try:
    from sentence_transformers import CrossEncoder
except ImportError as exc:  # pragma: no cover - exercised via lazy init tests
    CrossEncoder = None
    _CROSS_ENCODER_IMPORT_ERROR: ImportError | None = exc
else:
    _CROSS_ENCODER_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

ENTAILMENT_MODEL = "cross-encoder/nli-deberta-v3-small"
_DEFAULT_ENTAILMENT_INDEX = 1
_ENTAILMENT_SCORER: TextualEntailmentScorer | None = None
_ENTAILMENT_UNAVAILABLE = False


def _clamp_probability(value: float) -> float:
    """Clamp a value to the closed probability interval [0, 1]."""
    return max(0.0, min(value, 1.0))


def _stable_softmax(values: Sequence[float]) -> list[float]:
    """Convert logits to probabilities."""
    if not values:
        return []
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in exp_values]


def _normalize_label(label: str) -> str:
    """Normalize model labels for robust matching."""
    return label.strip().lower().replace("_", " ")


def _resolve_entailment_index(label2id: Mapping[str, int] | None) -> int:
    """Resolve entailment class index from label metadata."""
    if not label2id:
        return _DEFAULT_ENTAILMENT_INDEX
    for label, index in label2id.items():
        if "entail" in _normalize_label(label):
            return index
    return _DEFAULT_ENTAILMENT_INDEX


def _coerce_to_float_vector(raw_prediction: object) -> list[float]:
    """Coerce a prediction object into a numeric vector."""
    if isinstance(raw_prediction, (int, float)):
        return [float(raw_prediction)]
    if isinstance(raw_prediction, Iterable):
        values: list[float] = []
        for value in raw_prediction:
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values
    return []


class TextualEntailmentScorer:
    """Score passage-level entailment for a candidate answer hypothesis."""

    def __init__(self, model_name: str = ENTAILMENT_MODEL) -> None:
        """Initialize the NLI cross-encoder."""
        if CrossEncoder is None:
            raise ImportError(
                "Textual entailment requires sentence-transformers. "
                "Install watson-lite with the 'rerank' or 'full' extra."
            ) from _CROSS_ENCODER_IMPORT_ERROR
        logger.debug("Loading textual entailment model: %s", model_name)
        self.model = CrossEncoder(model_name, max_length=512)
        config = getattr(getattr(self.model, "model", None), "config", None)
        label2id = (
            getattr(config, "label2id", None)
            if isinstance(getattr(config, "label2id", None), Mapping)
            else None
        )
        self.entailment_index = _resolve_entailment_index(label2id)

    def _entailment_probability(self, raw_prediction: object) -> float:
        """Convert model output into an entailment probability."""
        values = _coerce_to_float_vector(raw_prediction)
        if not values:
            return 0.0
        if len(values) == 1:
            return _clamp_probability(values[0])
        probs = _stable_softmax(values)
        if not probs:
            return 0.0
        if self.entailment_index < 0 or self.entailment_index >= len(probs):
            return 0.0
        return probs[self.entailment_index]

    def score(self, question: str, candidate_span: str, passages: list[str]) -> float:
        """Return the best entailment probability across candidate passages."""
        if not question.strip() or not candidate_span.strip():
            return 0.0
        clean_passages = [passage for passage in passages if passage.strip()]
        if not clean_passages:
            return 0.0

        hypothesis = f"The answer to '{question}' is '{candidate_span}'."
        pairs = [(passage, hypothesis) for passage in clean_passages]
        predictions = self.model.predict(pairs, show_progress_bar=False)

        best = 0.0
        for prediction in predictions:
            best = max(best, self._entailment_probability(prediction))
        return _clamp_probability(best)


def score_entailment(question: str, candidate_span: str, passages: list[str]) -> float:
    """Score textual entailment with lazy model loading and safe fallback to zero."""
    global _ENTAILMENT_SCORER, _ENTAILMENT_UNAVAILABLE

    if _ENTAILMENT_UNAVAILABLE:
        return 0.0
    if _ENTAILMENT_SCORER is None:
        try:
            _ENTAILMENT_SCORER = TextualEntailmentScorer()
        except ImportError:
            logger.debug(
                "Textual entailment disabled: sentence-transformers unavailable"
            )
            _ENTAILMENT_UNAVAILABLE = True
            return 0.0
    return _ENTAILMENT_SCORER.score(question, candidate_span, passages)
