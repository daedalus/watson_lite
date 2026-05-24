FALLBACK_ANSWERS = frozenset(
    {
        "no answer found",
        "could not retrieve relevant passages.",
    }
)


def is_fallback_answer_text(text: str) -> bool:
    return text.lower().strip() in FALLBACK_ANSWERS
