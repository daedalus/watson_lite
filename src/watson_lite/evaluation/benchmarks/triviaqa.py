from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from watson_lite.evaluation.benchmarks._download_utils import (
    download_with_resume,
    stream_extract_tar_member,
)

TRIVIAQA_RC_URL = "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"
TRIVIAQA_DEV_PATH = "triviaqa-rc/qa/wikipedia-dev.json"
TRIVIAQA_TEST_PATH = "triviaqa-rc/qa/wikipedia-test.json"


def _convert_triviaqa(
    raw_bytes: bytes,
    output_path: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    raw = json.loads(raw_bytes.decode("utf-8"))
    samples: list[dict[str, Any]] = []
    for entry in raw.get("Data", []):
        question = entry.get("Question", "").strip()
        answer_data = entry.get("Answer", {}) or {}
        aliases = answer_data.get("Aliases", []) or []
        aliases = [a.strip() for a in aliases if a and a.strip()]
        if not question or not aliases:
            continue
        samples.append(
            {
                "question": question,
                "answers": list(dict.fromkeys(aliases)),
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


def download_triviaqa(
    output_path: str,
    split: str = "dev",
    max_samples: int | None = None,
    download_dir: str | None = None,
) -> list[dict[str, Any]]:
    path_map = {"dev": TRIVIAQA_DEV_PATH, "test": TRIVIAQA_TEST_PATH}
    internal_path = path_map.get(split)
    if internal_path is None:
        msg = f"Unknown split '{split}'; choose from {list(path_map)}"
        raise ValueError(msg)

    download_dir_path = Path(download_dir or "benchmarks/raw")
    download_dir_path.mkdir(parents=True, exist_ok=True)

    archive_path = download_dir_path / "triviaqa-rc.tar.gz"
    download_with_resume(TRIVIAQA_RC_URL, archive_path, label="TriviaQA RC")

    print(f"Extracting {internal_path} from archive...")
    raw_bytes = stream_extract_tar_member(archive_path, internal_path)
    return _convert_triviaqa(raw_bytes, output_path, max_samples=max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TriviaQA Wikipedia dataset")
    parser.add_argument("--output", default="benchmarks/triviaqa_dev.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--download-dir",
        default="benchmarks/raw",
        help="Directory to cache the tar.gz archive (supports resume)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run benchmark immediately after download (in-process, no subprocess)",
    )
    args = parser.parse_args()
    download_triviaqa(
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
