import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from watson_lite.core.cache import get_cache
from watson_lite.core.config import FeatureConfig
from watson_lite.core.models import FinalAnswer
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


def _add_input_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", nargs="*", help="Single question to answer")
    parser.add_argument(
        "--questions-from-file",
        type=str,
        default=None,
        help="Path to a text file with one question per line (batch mode)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write results as a JSON array (batch mode or single question)",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Render answers as human-readable text or structured JSON",
    )
    parser.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="Include diagnostics in text output",
    )


def _add_feature_toggle_args(parser: argparse.ArgumentParser) -> None:
    toggles = [
        ("--vector-retrieval", "Enable/disable vector retrieval"),
        ("--query-expansion", "Enable/disable query expansion variants"),
        ("--graph-enrichment", "Enable/disable Wikidata graph enrichment"),
        ("--cross-encoder-reranking", "Enable/disable cross-encoder reranking"),
        ("--question-type-bonus", "Enable/disable question-type confidence bonus"),
        ("--type-coercion", "Enable/disable type coercion signal"),
        ("--term-match", "Enable/disable IDF-weighted term match signal"),
        ("--consistency", "Enable/disable temporal/geospatial consistency checks"),
        ("--entailment", "Enable/disable textual entailment confidence signal"),
        (
            "--answer-merging",
            "Enable/disable merging equivalent answers via Wikidata QID",
        ),
        ("--multi-hypothesis", "Enable/disable multiple hypothesis generators"),
        (
            "--per-candidate-retrieval",
            "Enable/disable per-candidate evidence re-retrieval",
        ),
        (
            "--bidirectional-validation",
            "Enable/disable bidirectional answer validation",
        ),
        ("--iterative-retrieval", "Enable/disable iterative multi-pass retrieval"),
        ("--semantic-nlp", "Enable/disable semantic NLP helpers"),
    ]
    for flag, help_text in toggles:
        parser.add_argument(
            flag, action=argparse.BooleanOptionalAction, default=None, help=help_text
        )


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=("wikipedia",),
        help="Comma-separated datasets to query (e.g. wikipedia,wikibooks,wikiquote,pubmed,arxiv)",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=_parse_datasets,
        default=None,
        help="Comma-separated dataset names to exclude (e.g. pubmed,arxiv)",
    )
    parser.add_argument(
        "--elasticsearch-url",
        type=str,
        default=None,
        help="Elasticsearch base URL (used when datasets include 'elasticsearch')",
    )
    parser.add_argument(
        "--elasticsearch-index",
        type=str,
        default=None,
        help="Elasticsearch index name (used when datasets include 'elasticsearch')",
    )
    parser.add_argument(
        "--huggingface-dataset",
        type=str,
        default=None,
        help="Hugging Face dataset id (used when datasets include 'huggingface')",
    )
    parser.add_argument(
        "--huggingface-config",
        type=str,
        default=None,
        help="Hugging Face dataset config name (optional, used when datasets include 'huggingface')",
    )
    parser.add_argument(
        "--huggingface-split",
        type=str,
        default=None,
        help="Hugging Face dataset split (used when datasets include 'huggingface')",
    )
    parser.add_argument(
        "--huggingface-token",
        type=str,
        default=None,
        help="Hugging Face auth token (optional, used for private/gated datasets)",
    )


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watson-lite")
    _add_input_output_args(parser)
    parser.add_argument(
        "--profile",
        choices=("baseline", "minimal"),
        default="baseline",
        help="Starting runtime profile",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the local cache before answering or benchmarking",
    )
    _add_feature_toggle_args(parser)
    parser.add_argument("--wiki-top-k", type=int, default=5)
    _add_dataset_args(parser)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--extract-top-k", type=int, default=5)
    parser.add_argument("--max-retrieval-passes", type=int, default=2)
    parser.add_argument("--iterative-retrieval-threshold", type=float, default=0.3)
    _add_benchmark_args(parser)
    parser.add_argument(
        "--device", type=int, default=-1, help="Torch device index (-1 = CPU, 0+ = GPU)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step-by-step pipeline logs",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG-level logging"
    )
    parser.add_argument("--logfile", type=str, help="Write logs to this file")
    return parser


def _build_config(args: argparse.Namespace) -> FeatureConfig:
    base = (
        FeatureConfig.baseline()
        if args.profile == "baseline"
        else FeatureConfig.minimal()
    )
    enabled = tuple(d for d in args.datasets if d not in (args.exclude_datasets or ()))
    overrides = {
        "dataset_sources": enabled,
        "elasticsearch_url": args.elasticsearch_url,
        "elasticsearch_index": args.elasticsearch_index,
        "huggingface_dataset": args.huggingface_dataset,
        "huggingface_config": args.huggingface_config,
        "huggingface_split": args.huggingface_split,
        "huggingface_token": args.huggingface_token,
        "wikipedia_top_k_per_query": args.wiki_top_k,
        "retrieval_top_k": args.retrieval_top_k,
        "rerank_top_k": args.rerank_top_k,
        "extraction_top_k": args.extract_top_k,
        "max_retrieval_passes": args.max_retrieval_passes,
        "iterative_retrieval_threshold": args.iterative_retrieval_threshold,
    }
    for name in (
        "vector_retrieval",
        "query_expansion",
        "graph_enrichment",
        "cross_encoder_reranking",
        "question_type_bonus",
        "type_coercion",
        "term_match",
        "consistency",
        "entailment",
        "answer_merging",
        "multi_hypothesis",
        "per_candidate_retrieval",
        "bidirectional_validation",
        "iterative_retrieval",
        "semantic_nlp",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return replace(base, **overrides)


def _print_text_answer(answer: FinalAnswer, *, show_diagnostics: bool) -> None:
    print("=" * 50)
    print(f"  ANSWER:     {answer.answer}")
    print(f"  CONFIDENCE: {answer.confidence * 100:.1f}%")
    print(f"  SOURCE:     {answer.source}")
    print(f"  URL:        {answer.url}")
    if answer.graph_facts:
        print("  GRAPH CORROBORATION:")
        for fact in answer.graph_facts[:3]:
            print(f"    · {fact}")
    print("  Confidence breakdown:")
    for key, value in answer.confidence_breakdown.items():
        print(f"    {key}: {value}")
    if show_diagnostics and answer.diagnostics is not None:
        diagnostics = answer.diagnostics
        print("  Diagnostics:")
        print(
            "    passages:"
            f" fetched={diagnostics.passages_fetched}"
            f" reranked={diagnostics.passages_reranked}"
            f" extracted={diagnostics.passages_extracted}"
        )
        print(
            "    cache:"
            f" hits={diagnostics.cache_hits}"
            f" misses={diagnostics.cache_misses}"
        )
        if diagnostics.stage_latencies_s:
            timings = ", ".join(
                f"{stage}={latency:.3f}s"
                for stage, latency in diagnostics.stage_latencies_s.items()
            )
            print(f"    timings: {timings}")
    print("=" * 50)


def _emit_answer(
    answer: FinalAnswer,
    *,
    output_format: str,
    show_diagnostics: bool,
) -> None:
    if output_format == "json":
        print(json.dumps(asdict(answer), indent=2, sort_keys=True))
        return
    _print_text_answer(answer, show_diagnostics=show_diagnostics)


def _run_benchmark_mode(args: argparse.Namespace, config: FeatureConfig) -> int:
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


def _run_batch_mode(args: argparse.Namespace, watson: WatsonLite) -> int:
    with open(args.questions_from_file, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    results = []
    for i, question in enumerate(questions):
        logging.info("[%d/%d] %s", i + 1, len(questions), question)
        answer = watson.answer(question, verbose=args.verbose)
        results.append(asdict(answer))
        _emit_answer(
            answer, output_format=args.output, show_diagnostics=args.show_diagnostics
        )
        sys.stdout.flush()
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        logging.info("Wrote %d results to %s", len(results), args.output_json)
    return 0


def _run_single_question(args: argparse.Namespace, watson: WatsonLite) -> int:
    question = " ".join(args.question)
    answer = watson.answer(question, verbose=args.verbose)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(asdict(answer), f, indent=2, sort_keys=True)
    _emit_answer(
        answer, output_format=args.output, show_diagnostics=args.show_diagnostics
    )
    return 0


def _run_interactive(args: argparse.Namespace, watson: WatsonLite) -> int:
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
        answer = watson.answer(question, verbose=args.verbose)
        _emit_answer(
            answer, output_format=args.output, show_diagnostics=args.show_diagnostics
        )
    return 0


def _setup_logging(args: argparse.Namespace) -> None:
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.logfile:
        logging.basicConfig(
            filename=args.logfile, filemode="a", format="%(message)s", level=log_level
        )
    else:
        logging.basicConfig(format="%(message)s", level=log_level, stream=sys.stdout)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    _setup_logging(args)
    config = _build_config(args)

    if args.clear_cache:
        get_cache().clear()
        logging.info("Cleared local cache")

    if args.benchmark_dataset:
        return _run_benchmark_mode(args, config)

    watson = WatsonLite(config=config, device=args.device)

    if args.questions_from_file:
        return _run_batch_mode(args, watson)
    if args.question:
        return _run_single_question(args, watson)
    return _run_interactive(args, watson)


if __name__ == "__main__":
    raise SystemExit(main())
