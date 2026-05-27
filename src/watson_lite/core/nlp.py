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


def _ner_input(text: str, nlp: Any) -> str:
    """Build a version of *text* that triggers better NER.

    spaCy's NER uses casing as a strong signal.  Lowercased proper nouns
    (e.g. ``"norse"`` → NOUN, no entity) are missed while their
    capitalised counterparts (``"Norse"`` → PROPN, ORG entity) are
    picked up.

    This function runs a lightweight first pass to get POS tags, then
    capitalises content words (NOUN, PROPN, ADJ, ADV) so the subsequent
    full pipeline pass detects entities that would otherwise be missed.
    Function words (VERB, AUX, DET, …) are left in their original casing
    to avoid destabilising the dependency parse.
    """
    with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler"]):
        doc = nlp(text)
    skip = {
        "VERB",
        "AUX",
        "DET",
        "ADP",
        "PRON",
        "SCONJ",
        "CCONJ",
        "PART",
        "INTJ",
        "PUNCT",
        "SPACE",
        "X",
        "NUM",
        "SYM",
    }
    parts = [
        t.text.capitalize() if t.pos_ not in skip and t.text.islower() else t.text
        for t in doc
    ]
    return " ".join(parts)


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


def _extract_lat(doc: Doc, question_type: str = "") -> tuple[str | None, list[str]]:
    """Extract Lexical Answer Type using spaCy structural analysis.

    Returns (lat_headword, list_of_expected_qids). Returns (None, []) when no
    LAT can be inferred.
    """
    tokens = [t for t in doc if not t.is_punct]
    if not tokens:
        return None, []

    first = tokens[0]

    # PRON + nsubj with VERB root → agent/person question (universal)
    if first.pos_ == "PRON":
        pron_type = first.morph.get("PronType")
        is_interrogative = not pron_type or bool(set(pron_type) & {"Int", "Rel", "Ind"})
        if is_interrogative and first.dep_ in {
            "nsubj",
            "csubj",
            "nsubjpass",
            "nsubj:pass",
        }:
            root = next((t for t in doc if t.dep_ == "ROOT"), None)
            if root is not None and root.pos_ == "VERB":
                return "person", LAT_QID_MAP.get("person", [])

    # Scan for first content noun after the question word
    skip_pos = {
        "AUX",
        "DET",
        "ADP",
        "PRON",
        "SCONJ",
        "PART",
        "CCONJ",
        "INTJ",
        "NUM",
        "ADV",
        "VERB",
        "ADJ",
    }
    for token in tokens[1:]:
        if token.pos_ in {"NOUN", "PROPN"}:
            lemma = token.lemma_.lower()
            qids = LAT_QID_MAP.get(lemma, [])
            return lemma, qids
        if token.pos_ not in skip_pos:
            lemma = token.lemma_.lower()
            qids = LAT_QID_MAP.get(lemma, [])
            return lemma, qids

    return None, []


def _extract_srl_frames(doc: Doc) -> list[dict[str, str]]:  # pragma: no cover
    """Build simple SRL-like frames from dependency parses.

    Requires a spaCy ``Doc`` object; covered by the NLP test suite.
    """
    frames: list[dict[str, str]] = []
    for token in doc:
        if token.pos_ != "VERB":
            continue

        frame: dict[str, str] = {"predicate": token.lemma_ or token.text}
        for child in token.children:
            if child.dep_ == "nsubj":
                frame["arg0"] = child.text
            elif child.dep_ in {"dobj", "obj"}:
                frame["arg1"] = child.text
            elif child.dep_ == "prep":
                pobj = next(
                    (
                        grandchild
                        for grandchild in child.children
                        if grandchild.dep_ == "pobj"
                    ),
                    None,
                )
                if pobj is not None:
                    frame["argm"] = f"{child.text} {pobj.text}"
        frames.append(frame)
    return frames


def _spacy_model_for_language(language: str) -> str:
    """Derive the spaCy model name from a langdetect language code."""
    if language == "en":
        return "en_core_web_sm"
    return f"{language}_core_news_sm"


class NLPProcessor:
    def __init__(
        self,
        model: str | None = None,
        language: str = "en",
        semantic_nlp: bool = False,
    ) -> None:
        if spacy is None:
            raise ImportError(
                "spaCy is required for NLP processing. "
                "Install watson-lite with the 'nlp' or 'full' extra."
            ) from _SPACY_IMPORT_ERROR
        if model is None:
            model = _spacy_model_for_language(language)
        try:
            logger.debug("Loading spaCy model: %s (language=%s)", model, language)
            self.nlp = spacy.load(model)
        except OSError:
            logger.warning(
                "spaCy model %s not found for language %s, falling back to English",
                model,
                language,
            )
            self.nlp = spacy.load("en_core_web_sm")
            language = "en"
        self.language = language
        self.semantic_nlp = semantic_nlp
        self._has_coreferee = False
        try:
            self.nlp.add_pipe("coreferee")
        except Exception as exc:  # pragma: no cover - depends on optional install
            logger.debug("coreferee unavailable: %s", exc)
        else:  # pragma: no cover - depends on optional install
            self._has_coreferee = True

    def classify_question(self, text: str) -> str:
        doc = self.nlp(text)
        for token in doc:
            if token.is_punct:
                continue
            if token.pos_ == "PRON":
                pron_type = token.morph.get("PronType")
                if (
                    not pron_type or bool(set(pron_type) & {"Int", "Rel", "Ind"})
                ) and token.dep_ != "expl":
                    return "what"
                return "unknown"
            if token.pos_ == "SCONJ" and token.dep_ == "advmod":
                return "what"
            return "unknown"
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
            if (token.pos_ == "CCONJ" or token.text == "?") and current:
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

    @staticmethod
    def _int_list(items: object) -> list[int] | None:
        if isinstance(items, (list, tuple)) and items:
            ints = [i for i in items if isinstance(i, int)]
            if ints:
                return ints
        return None

    def _mention_text(self, doc: Doc, mention: object) -> str:  # pragma: no cover
        """Extract the text of a coreference mention span.

        Supports multiple coreferee/spaCy mention formats.  Covered by the NLP
        test suite which is excluded from the CI coverage run.
        """
        if isinstance(mention, str):
            return mention

        start = getattr(mention, "start", None)
        end = getattr(mention, "end", None)
        if isinstance(start, int) and isinstance(end, int):
            if start < end:
                return str(doc[start:end])

        int_indexes = self._int_list(getattr(mention, "token_indexes", None))
        if not int_indexes:
            int_indexes = self._int_list(mention)
        if int_indexes:
            return str(doc[min(int_indexes) : max(int_indexes) + 1])

        token_index = getattr(mention, "token_index", None)
        if isinstance(token_index, int):
            return str(doc[token_index])

        return str(mention)

    def _resolve_coreference(self, doc: Doc) -> list[list[str]]:  # pragma: no cover
        """Resolve coreference chains from a coreferee-annotated Doc.

        Returns a list of clusters; each cluster is a list of mention strings.
        Falls back to ``[]`` when coreferee is not installed.  Covered by the
        NLP test suite which is excluded from the CI coverage run.
        """
        if not self._has_coreferee:
            return []
        try:
            chains = getattr(doc._, "coref_chains")
        except Exception:
            return []
        if chains is None:
            return []

        clusters: list[list[str]] = []
        for chain in chains:
            raw_mentions = getattr(chain, "mentions", None)
            mentions_iterable = (
                raw_mentions if raw_mentions is not None else list(chain)
            )
            mentions = [
                mention_text
                for mention in mentions_iterable
                if (mention_text := self._mention_text(doc, mention))
            ]
            if mentions:
                clusters.append(mentions)
        return clusters

    def process(self, question: str, semantic_nlp: bool = False) -> ParsedQuestion:
        normalized = _ner_input(question, self.nlp)
        doc = self.nlp(normalized)
        question_type = self.classify_question(question)
        lat, lat_qids = _extract_lat(doc, question_type)
        semantic_enabled = semantic_nlp or self.semantic_nlp
        srl_frames = _extract_srl_frames(doc) if semantic_enabled else []
        coref_clusters = self._resolve_coreference(doc) if semantic_enabled else []
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
            srl_frames=srl_frames,
            coref_clusters=coref_clusters,
        )
