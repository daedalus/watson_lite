import logging
import sys

from watson_lite.pipeline import WatsonLite


def main() -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    watson = WatsonLite()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        watson.answer(question, verbose=True)
        return 0

    print("""
╔══════════════════════════════════════╗
║         WatsonLite  v0.1.0           ║
║  Extractive QA · No LLM · No Training║
╚══════════════════════════════════════╝
Type a question and press Enter.
Type 'quit' or Ctrl+C to exit.
""")

    while True:
        try:
            question = input("\n❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return 0

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        watson.answer(question, verbose=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
