import spacy
from spacy.tokens import Doc

from watson_lite.core.models import ParsedQuestion

QUESTION_TYPES = {
    "who": ["who", "whose", "whom"],
    "what": ["what", "which"],
    "when": ["when"],
    "where": ["where"],
    "how": ["how"],
    "why": ["why"],
}


class NLPProcessor:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        print(f"[NLP] Loading spaCy model: {model}")
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
        return ParsedQuestion(
            raw=question,
            question_type=self.classify_question(question),
            entities=self.extract_entities(doc),
            noun_chunks=[chunk.text for chunk in doc.noun_chunks],
            root_verb=self.get_root_verb(doc),
            sub_questions=self.decompose_question(question),
            keywords=self.extract_keywords(doc),
        )
