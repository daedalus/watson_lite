import argparse
import logging
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from watson_lite.core.config import FeatureConfig
from watson_lite.evaluation.benchmark_runner import (
    RegressionThresholds,
    run_benchmark_profiles,
)
from watson_lite.pipeline import WatsonLite

try:
    _VERSION = pkg_version("watson-lite")
except PackageNotFoundError:
    _VERSION = "unknown"


def _parse_datasets(value: str) -> tuple[str, ...]:
    """Parse comma-separated dataset names into a normalized tuple."""
    datasets = tuple(
        cleaned.lower() for item in value.split(",") if (cleaned := item.strip())
    )
    if not datasets:
        raise argparse.ArgumentTypeError("At least one dataset must be provided")
    return datasets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watson-lite")
    parser.add_argument("question", nargs="*", help="Single question to answer")

    parser.add_argument(
        "--vector-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vector retrieval",
    )
    parser.add_argument(
        "--query-expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable query expansion variants",
    )
    parser.add_argument(
        "--graph-enrichment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Wikidata graph enrichment",
    )
    parser.add_argument(
        "--cross-encoder-reranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable cross-encoder reranking",
    )
    parser.add_argument(
        "--question-type-bonus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable question-type confidence bonus",
    )
    parser.add_argument(
        "--type-coercion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable type coercion signal",
    )
    parser.add_argument(
        "--term-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable IDF-weighted term match signal",
    )
    parser.add_argument(
        "--consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable temporal/geospatial consistency checks",
    )

    parser.add_argument("--wiki-top-k", type=int, default=5)
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=("wikipedia",),
        help="Comma-separated datasets to query (e.g. wikipedia,wikibooks)",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--extract-top-k", type=int, default=5)

    parser.add_argument("--benchmark-dataset")
    parser.add_argument("--benchmark-output-json", default="benchmark_results.json")
    parser.add_argument("--benchmark-output-csv")
    parser.add_argument("--ablation-sweep", action="store_true")
    parser.add_argument("--regression-check", action="store_true")
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-exact-match-drop", type=float, default=0.02)
    parser.add_argument("--max-f1-drop", type=float, default=0.02)
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    parser.add_argument("--metric-tolerance", type=float, default=0.001)
    parser.add_argument("--max-latency-p95-s", type=float)
    return parser


def _build_config(args: argparse.Namespace) -> FeatureConfig:
    return FeatureConfig(
        vector_retrieval=args.vector_retrieval,
        query_expansion=args.query_expansion,
        graph_enrichment=args.graph_enrichment,
        cross_encoder_reranking=args.cross_encoder_reranking,
        question_type_bonus=args.question_type_bonus,
        type_coercion=args.type_coercion,
        term_match=args.term_match,
        consistency=args.consistency,
        dataset_sources=args.datasets,
        wikipedia_top_k_per_query=args.wiki_top_k,
        retrieval_top_k=args.retrieval_top_k,
        rerank_top_k=args.rerank_top_k,
        extraction_top_k=args.extract_top_k,
    )


def main() -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])

    config = _build_config(args)

    if args.benchmark_dataset:
        thresholds = RegressionThresholds(
            max_accuracy_drop=args.max_accuracy_drop,
            max_exact_match_drop=args.max_exact_match_drop,
            max_f1_drop=args.max_f1_drop,
            max_recall_drop=args.max_recall_drop,
            metric_tolerance=args.metric_tolerance,
            max_latency_p95_s=args.max_latency_p95_s,
        )
        results, regressions = run_benchmark_profiles(
            dataset_path=args.benchmark_dataset,
            config=config,
            output_json_path=args.benchmark_output_json,
            output_csv_path=args.benchmark_output_csv,
            recall_k=args.recall_k,
            calibration_bins=args.calibration_bins,
            ablation_sweep=args.ablation_sweep,
            regression_check=args.regression_check,
            thresholds=thresholds,
        )
        logging.info(
            "Benchmark completed: %d profiles, %d regressions",
            len(results),
            len(regressions),
        )
        return 1 if regressions else 0

    watson = WatsonLite(config=config)

    if args.question:
        question = " ".join(args.question)
        watson.answer(question, verbose=True)
        return 0

    print(f"""
╔══════════════════════════════════════╗
║         WatsonLite  v{_VERSION:<15} ║
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
