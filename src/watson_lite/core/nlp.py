import logging
from typing import TYPE_CHECKING, Any

from watson_lite.core.models import ParsedQuestion

try:
    import spacy
except ImportError as exc:  # pragma: no cover - exercised via lazy init tests
    spacy = None
    _SPACY_IMPORT_ERROR: ImportError | None = exc
else:
    _SPACY_IMPORT_ERROR = None

if TYPE_CHECKING:
    from spacy.tokens import Doc
else:  # pragma: no cover - runtime fallback used when type checking is inactive
    Doc = Any

logger = logging.getLogger(__name__)

QUESTION_TYPES = {
    "who": ["who", "whose", "whom"],
    "what": ["what", "which"],
    "when": ["when"],
    "where": ["where"],
    "how": ["how"],
    "why": ["why"],
}

# Maps common Lexical Answer Types (LATs) to Wikidata QIDs so the type
# coercion scorer can check candidate spans against the expected type.
LAT_QID_MAP: dict[str, list[str]] = {
    "person": ["Q5"],
    "people": ["Q5"],
    "city": ["Q515"],
    "country": ["Q6256"],
    "river": ["Q4022"],
    "mountain": ["Q8502"],
    "island": ["Q23442"],
    "building": ["Q41176"],
    "bridge": ["Q12280"],
    "language": ["Q34770"],
    "organization": ["Q43229"],
    "company": ["Q891723", "Q4830453"],
    "book": ["Q571"],
    "film": ["Q11424"],
    "song": ["Q7366"],
    "album": ["Q482994"],
    "sport": ["Q349"],
    "event": ["Q1656682"],
    "war": ["Q198"],
    "treaty": ["Q131569"],
    "university": ["Q3918"],
    "school": ["Q3914"],
    "museum": ["Q33506"],
    "planet": ["Q634"],
    "star": ["Q523"],
    "chemical_element": ["Q11344"],
    "year": ["Q577"],
    "number": ["Q11563"],
    "currency": ["Q8142"],
    "color": ["Q1075"],
    "animal": ["Q729"],
    "plant": ["Q756"],
    "god": ["Q407"],
}


def _extract_lat(text: str, question_type: str) -> tuple[str | None, list[str]]:
    """Extract Lexical Answer Type from the question using simple heuristics.

    Returns (lat_headword, list_of_expected_qids).  Returns (None, []) when no
    LAT can be inferred.
    """
    lower = text.lower().strip()

    if question_type == "who":
        return "person", LAT_QID_MAP["person"]

    if question_type == "where":
        return "location", LAT_QID_MAP.get("city", [])

    if question_type == "when":
        return None, []

    if question_type == "why":
        return None, []

    if question_type in ("what", "unknown"):
        # Try "what/who/which <noun phrase>" pattern.
        first_word = lower.split()[0] if lower.split() else ""
        rest = " ".join(lower.split()[1:]) if len(lower.split()) > 1 else ""

        if first_word in ("what", "which") and rest:
            # Use a simple heuristic: the first noun chunk-like word is the LAT.
            # Filter out common copula/auxiliary verbs.
            skip_words = {
                "is",
                "are",
                "was",
                "were",
                "do",
                "does",
                "did",
                "has",
                "have",
                "had",
                "can",
                "could",
                "will",
                "would",
                "shall",
                "should",
                "may",
                "might",
                "the",
                "a",
                "an",
            }
            head = rest.split()[0] if rest.split() else ""
            if head and head not in skip_words:
                qids = LAT_QID_MAP.get(head)
                if qids:
                    return head, qids

    return None, []


class NLPProcessor:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        if spacy is None:
            raise ImportError(
                "spaCy is required for NLP processing. "
                "Install watson-lite with the 'nlp' or 'full' extra."
            ) from _SPACY_IMPORT_ERROR
        logger.debug("Loading spaCy model: %s", model)
        self.nlp = spacy.load(model)

    def classify_question(self, text: str) -> str:
        first = text.strip().lower().split()[0] if text.strip() else ""
        for qtype, triggers in QUESTION_TYPES.items():
            if first in triggers:
                return qtype
        return "unknown"

    def extract_entities(self, doc: Doc) -> list[dict[str, str | int]]:
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    def extract_keywords(self, doc: Doc) -> list[str]:
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN", "VERB")
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        ]

    def get_root_verb(self, doc: Doc) -> str | None:
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return str(token.lemma_)
        return None

    def decompose_question(self, text: str) -> list[str]:
        doc = self.nlp(text)
        sub_questions: list[str] = []
        current: list[str] = []

        for token in doc:
            if token.text in ("and", "but", "or", "?") and current:
                chunk = " ".join(current).strip()
                if len(chunk.split()) > 2:
                    sub_questions.append(chunk)
                current = []
            else:
                current.append(token.text)

        if current:
            chunk = " ".join(current).strip()
            if len(chunk.split()) > 2:
                sub_questions.append(chunk)

        return sub_questions if len(sub_questions) > 1 else [text]

    def process(self, question: str) -> ParsedQuestion:
        doc = self.nlp(question)
        question_type = self.classify_question(question)
        lat, lat_qids = _extract_lat(question, question_type)
        return ParsedQuestion(
            raw=question,
            question_type=question_type,
            entities=self.extract_entities(doc),
            noun_chunks=[chunk.text for chunk in doc.noun_chunks],
            root_verb=self.get_root_verb(doc),
            sub_questions=self.decompose_question(question),
            keywords=self.extract_keywords(doc),
            lat=lat,
            lat_qids=lat_qids,
        )
