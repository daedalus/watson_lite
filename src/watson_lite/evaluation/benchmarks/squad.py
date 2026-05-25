from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from watson_lite.evaluation.benchmarks._download_utils import (
    download_with_resume,
)

SQUAD_V2_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def _convert_squad(
    raw_path: Path,
    output_path: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    with open(raw_path) as f:
        data = json.load(f)

    samples: list[dict[str, Any]] = []
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                question = qa.get("question", "").strip()
                answers = list(
                    {
                        a["text"].strip()
                        for a in qa.get("answers", [])
                        if a.get("text", "").strip()
                    }
                )
                if not question or not answers:
                    continue
                samples.append(
                    {
                        "question": question,
                        "answers": answers,
                        "evidence_passages": [context],
                    }
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break

    Path(output_path).write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(samples)} samples to {output_path}")
    return samples


def download_squad(
    output_path: str,
    max_samples: int | None = None,
    download_dir: str | None = None,
) -> list[dict[str, Any]]:
    if download_dir:
        tmp = Path(download_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        raw_path = tmp / "squad_v2_dev.json"
    else:
        tmp = Path(tempfile.mkdtemp(prefix="watson_squad_"))
        raw_path = tmp / "squad_v2_dev.json"

    download_with_resume(SQUAD_V2_DEV_URL, raw_path, label="SQuAD v2.0 dev")
    return _convert_squad(raw_path, output_path, max_samples=max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SQuAD v2.0 dev set")
    parser.add_argument("--output", default="benchmarks/squad_v2_dev.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--download-dir",
        help="Directory to cache raw downloads (supports resume)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run benchmark immediately after download (in-process, no subprocess)",
    )
    args = parser.parse_args()
    download_squad(
        args.output,
        max_samples=args.max_samples,
        download_dir=args.download_dir,
    )

    if args.run:
        from watson_lite.core.config import FeatureConfig
        from watson_lite.evaluation.benchmark_runner import (
            run_benchmark_profiles,
        )

        print("Running benchmark in-process ...")
        results, regressions = run_benchmark_profiles(
            dataset_path=args.output,
            config=FeatureConfig.baseline(),
            output_json_path=args.output.replace(".json", "_results.json"),
            output_csv_path=args.output.replace(".json", "_results.csv"),
        )
        print(f"Done. {len(regressions)} regressions detected.")


if __name__ == "__main__":
    main()
