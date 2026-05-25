from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from watson_lite.evaluation.benchmarks._download_utils import (
    download_with_resume,
)

NQ_OPEN_DEV_URL = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl"
NQ_OPEN_TRAIN_URL = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl"


def _convert_nq(
    raw_path: Path,
    output_path: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question", "").strip()
            raw_answers = record.get("answer", []) or []
            if isinstance(raw_answers, str):
                raw_answers = [raw_answers]
            answers = [a.strip() for a in raw_answers if a and a.strip()]
            if not question or not answers:
                continue
            samples.append(
                {
                    "question": question,
                    "answers": list(dict.fromkeys(answers)),
                    "evidence_passages": [],
                }
            )
            if max_samples is not None and len(samples) >= max_samples:
                break

    Path(output_path).write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(samples)} samples to {output_path}")
    return samples


def download_natural_questions(
    output_path: str,
    split: str = "dev",
    max_samples: int | None = None,
    download_dir: str | None = None,
) -> list[dict[str, Any]]:
    url_map = {"dev": NQ_OPEN_DEV_URL, "train": NQ_OPEN_TRAIN_URL}
    url = url_map.get(split)
    if url is None:
        msg = f"Unknown split '{split}'; choose from {list(url_map)}"
        raise ValueError(msg)

    if download_dir:
        tmp = Path(download_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        raw_path = tmp / f"nq_open_{split}.jsonl"
    else:
        tmp = Path(tempfile.mkdtemp(prefix="watson_nq_"))
        raw_path = tmp / f"nq_open_{split}.jsonl"

    download_with_resume(url, raw_path, label=f"NQ Open {split}")
    return _convert_nq(raw_path, output_path, max_samples=max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Natural Questions Open dev set"
    )
    parser.add_argument("--output", default="benchmarks/natural_questions_dev.json")
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
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
    download_natural_questions(
        args.output,
        split=args.split,
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
