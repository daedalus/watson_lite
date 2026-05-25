"""Tests for search query generation from parsed questions."""
from watson_lite.core.models import ParsedQuestion
from watson_lite.retrieval.query_formulation import (
    _content_words,
    _entity_to_noun_chunk,
    generate_search_queries,
)


def _make_parsed(
    raw: str = "What is the capital of France?",
    question_type: str = "what",
    entities: list | None = None,
    noun_chunks: list[str] | None = None,
    root_verb: str | None = None,
    sub_questions: list[str] | None = None,
    keywords: list[str] | None = None,
    lat: str | None = None,
    lat_qids: list[str] | None = None,
) -> ParsedQuestion:
    return ParsedQuestion(
        raw=raw,
        question_type=question_type,
        entities=entities or [],
        noun_chunks=noun_chunks or [],
        root_verb=root_verb,
        sub_questions=sub_questions or [raw],
        keywords=keywords or [],
        lat=lat,
        lat_qids=lat_qids or [],
    )


class TestContentWords:
    def test_removes_stopwords(self) -> None:
        result = _content_words("What is the capital of France")
        assert "france" in result
        assert "capital" in result
        assert "what" not in result
        assert "the" not in result
        assert "is" not in result
        assert "of" not in result

    def test_removes_question_words(self) -> None:
        result = _content_words("Who designed the Eiffel Tower")
        assert "eiffel" in result
        assert "tower" in result
        assert "who" not in result

    def test_removes_short_words(self) -> None:
        result = _content_words("a be go")
        assert result == set() or all(len(w) > 1 for w in result)


class TestEntityToNounChunk:
    def test_matching_noun_chunk_returned(self) -> None:
        result = _entity_to_noun_chunk("Eiffel Tower", ["the Eiffel Tower", "Paris"])
        assert result == "the Eiffel Tower"

    def test_no_match_returns_none(self) -> None:
        result = _entity_to_noun_chunk("London", ["Paris", "Berlin"])
        assert result is None

    def test_case_insensitive_match(self) -> None:
        result = _entity_to_noun_chunk("france", ["the country of France"])
        assert result == "the country of France"


class TestGenerateSearchQueriesAugmented:
    def test_first_query_is_always_raw(self) -> None:
        parsed = _make_parsed(raw="Who built the Eiffel Tower?")
        queries = generate_search_queries(parsed)
        assert queries[0] == "Who built the Eiffel Tower?"

    def test_with_root_verb_and_entities(self) -> None:
        parsed = _make_parsed(
            raw="Who designed the Eiffel Tower?",
            question_type="who",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
            root_verb="design",
        )
        queries = generate_search_queries(parsed)
        assert any("design" in q for q in queries)

    def test_without_root_verb_uses_keywords(self) -> None:
        parsed = _make_parsed(
            raw="Eiffel Tower height?",
            question_type="what",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            keywords=["eiffel", "tower", "height"],
        )
        queries = generate_search_queries(parsed)
        assert len(queries) >= 1
        assert any("eiffel" in q for q in queries)

    def test_entity_with_noun_chunk_enriched(self) -> None:
        parsed = _make_parsed(
            raw="Who designed the Eiffel Tower?",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
            noun_chunks=["the Eiffel Tower"],
        )
        queries = generate_search_queries(parsed)
        assert any("the Eiffel Tower" in q for q in queries)

    def test_no_entities_uses_longest_noun_chunk(self) -> None:
        parsed = _make_parsed(
            raw="What is the capital city of France?",
            entities=[],
            noun_chunks=["the capital city of France", "France"],
        )
        queries = generate_search_queries(parsed)
        # Longest noun chunk should appear
        assert any("the capital city of France" in q for q in queries)

    def test_extra_entity_not_in_raw_appended(self) -> None:
        parsed = _make_parsed(
            raw="Who built it?",
            entities=[{"text": "Paris", "label": "LOC", "start": 0, "end": 5}],
        )
        queries = generate_search_queries(parsed)
        # "paris" is not in "who built it?" so an extra query should be created
        assert any("Paris" in q for q in queries)

    def test_lat_and_entity_query(self) -> None:
        parsed = _make_parsed(
            raw="What country is France?",
            entities=[{"text": "France", "label": "GPE", "start": 0, "end": 6}],
            lat="country",
        )
        queries = generate_search_queries(parsed)
        assert any("country" in q and "France" in q for q in queries)

    def test_sub_question_distinct_from_raw(self) -> None:
        parsed = _make_parsed(
            raw="Who is Napoleon?",
            sub_questions=["Who is Napoleon?", "when was Napoleon born"],
        )
        queries = generate_search_queries(parsed)
        assert any("born" in q for q in queries)

    def test_when_question_adds_date_and_year(self) -> None:
        parsed = _make_parsed(
            raw="When was the Eiffel Tower built?",
            question_type="when",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
        )
        queries = generate_search_queries(parsed)
        assert any("date" in q for q in queries) or any("year" in q for q in queries)

    def test_where_question_adds_location(self) -> None:
        parsed = _make_parsed(
            raw="Where is the Eiffel Tower?",
            question_type="where",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
        )
        queries = generate_search_queries(parsed)
        assert any("location" in q for q in queries)

    def test_how_question_adds_how_suffix(self) -> None:
        parsed = _make_parsed(
            raw="How tall is the Eiffel Tower?",
            question_type="how",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
        )
        queries = generate_search_queries(parsed)
        assert any("how" in q for q in queries)

    def test_why_question_adds_reason(self) -> None:
        parsed = _make_parsed(
            raw="Why did the Roman Empire fall?",
            question_type="why",
            entities=[
                {"text": "Roman Empire", "label": "ORG", "start": 0, "end": 12}
            ],
        )
        queries = generate_search_queries(parsed)
        assert any("reason" in q for q in queries)

    def test_at_most_five_queries(self) -> None:
        parsed = _make_parsed(
            raw="Who designed the Eiffel Tower?",
            question_type="who",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
            noun_chunks=["the Eiffel Tower"],
            root_verb="design",
            sub_questions=["Who designed the Eiffel Tower?", "Eiffel Tower designer"],
            lat="person",
        )
        queries = generate_search_queries(parsed)
        assert len(queries) <= 5


class TestGenerateSearchQueriesOriginal:
    def test_augment_false_uses_original_path(self) -> None:
        parsed = _make_parsed(
            raw="Who designed the Eiffel Tower?",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
            root_verb="design",
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert queries[0] == "Who designed the Eiffel Tower?"
        assert any("design" in q for q in queries)

    def test_original_when_type_adds_date_year(self) -> None:
        parsed = _make_parsed(
            raw="When was the Eiffel Tower built?",
            question_type="when",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert any("date" in q or "year" in q for q in queries)

    def test_original_where_type_adds_location(self) -> None:
        parsed = _make_parsed(
            raw="Where is Paris?",
            question_type="where",
            entities=[{"text": "Paris", "label": "GPE", "start": 0, "end": 5}],
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert any("location" in q for q in queries)

    def test_original_how_type_adds_how(self) -> None:
        parsed = _make_parsed(
            raw="How tall is it?",
            question_type="how",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert any("how" in q for q in queries)

    def test_original_why_type_adds_reason(self) -> None:
        parsed = _make_parsed(
            raw="Why did Rome fall?",
            question_type="why",
            entities=[{"text": "Rome", "label": "GPE", "start": 0, "end": 4}],
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert any("reason" in q for q in queries)

    def test_original_no_root_verb_uses_keywords(self) -> None:
        parsed = _make_parsed(
            raw="Eiffel Tower?",
            entities=[],
            root_verb=None,
            keywords=["eiffel", "tower"],
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert any("eiffel" in q for q in queries)

    def test_original_lat_entity_query(self) -> None:
        parsed = _make_parsed(
            raw="What country borders France?",
            entities=[{"text": "France", "label": "GPE", "start": 0, "end": 6}],
            lat="country",
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert any("country" in q and "France" in q for q in queries)

    def test_original_sub_question_distinct_added(self) -> None:
        parsed = _make_parsed(
            raw="Who is Napoleon?",
            sub_questions=["Who is Napoleon?", "when was Napoleon born"],
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert any("born" in q for q in queries)

    def test_original_at_most_five_queries(self) -> None:
        parsed = _make_parsed(
            raw="Who designed the Eiffel Tower?",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 12}],
            root_verb="design",
            lat="person",
            sub_questions=["Who designed the Eiffel Tower?", "Eiffel Tower designer"],
        )
        queries = generate_search_queries(parsed, augment_context=False)
        assert len(queries) <= 5
