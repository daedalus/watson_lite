"""
core/nlp.py
NLP Preprocessing — spaCy-based NER, coreference, POS, question classification.
No LLM involved. All rule-based.
"""

import spacy
from dataclasses import dataclass, field
from typing import List, Optional

# Question types Watson-style
QUESTION_TYPES = {
    "who":   ["who", "whose", "whom"],
    "what":  ["what", "which"],
    "when":  ["when"],
    "where": ["where"],
    "how":   ["how"],
    "why":   ["why"],
}


@dataclass
class ParsedQuestion:
    raw: str
    question_type: str                  # who / what / when / where / how / why / unknown
    entities: List[dict]                # [{text, label, start, end}]
    noun_chunks: List[str]
    root_verb: Optional[str]
    sub_questions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class NLPProcessor:
    def __init__(self, model: str = "en_core_web_sm"):
        print(f"[NLP] Loading spaCy model: {model}")
        self.nlp = spacy.load(model)

    def classify_question(self, text: str) -> str:
        first = text.strip().lower().split()[0] if text.strip() else ""
        for qtype, triggers in QUESTION_TYPES.items():
            if first in triggers:
                return qtype
        return "unknown"

    def extract_entities(self, doc) -> List[dict]:
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    def extract_keywords(self, doc) -> List[str]:
        """Extract meaningful tokens: nouns, proper nouns, verbs (non-stop)."""
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN", "VERB")
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        ]

    def get_root_verb(self, doc) -> Optional[str]:
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.lemma_
        return None

    def decompose_question(self, text: str) -> List[str]:
        """
        Rule-based decomposition using conjunctions and punctuation.
        e.g. "Who built the Eiffel Tower and when was it built?"
          -> ["Who built the Eiffel Tower", "when was it built"]
        """
        doc = self.nlp(text)
        sub_questions = []
        current = []

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


if __name__ == "__main__":
    processor = NLPProcessor()
    q = "Who built the Eiffel Tower and when was it completed?"
    result = processor.process(q)
    print(f"Type:      {result.question_type}")
    print(f"Entities:  {result.entities}")
    print(f"Keywords:  {result.keywords}")
    print(f"Sub-Qs:    {result.sub_questions}")
    print(f"Root verb: {result.root_verb}")
