"""
cli.py
Interactive command-line interface for WatsonLite.
"""

import sys
from pipeline import WatsonLite


def main():
    watson = WatsonLite()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        watson.answer(question, verbose=True)
        return

    print("""
╔══════════════════════════════════════╗
║         WatsonLite  v1.0             ║
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
            sys.exit(0)

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        watson.answer(question, verbose=True)


if __name__ == "__main__":
    main()
