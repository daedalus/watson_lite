from watson_lite.core.config import FeatureConfig
from watson_lite.evaluation.benchmark_runner import run_benchmark_profiles

datasets = [
    ("benchmarks/squad_v2_dev.json",
     "benchmarks/squad_v2_dev_results.json",
     "benchmarks/squad_v2_dev_results.csv"),
    ("benchmarks/natural_questions_dev.json",
     "benchmarks/nq_open_dev_results.json",
     "benchmarks/nq_open_dev_results.csv"),
    ("benchmarks/triviaqa_dev.json",
     "benchmarks/triviaqa_dev_results.json",
     "benchmarks/triviaqa_dev_results.csv"),
]

for ds_path, json_path, csv_path in datasets:
    print(f"===== {ds_path} =====")
    results, regressions = run_benchmark_profiles(
        dataset_path=ds_path,
        config=FeatureConfig.baseline(),
        output_json_path=json_path,
        output_csv_path=csv_path,
    )
    for r in results:
        m = r.report
        print(
            f"  total_questions:        {m.total_questions}\n"
            f"  accuracy_at_1:          {m.accuracy_at_1:.3f}\n"
            f"  exact_match:            {m.exact_match:.3f}\n"
            f"  f1:                     {m.f1:.3f}\n"
            f"  answer_success_rate:    {m.answer_success_rate:.3f}\n"
            f"  failure_empty_result:   {m.failure_empty_result_rate:.3f}\n"
            f"  retrieval_recall_at_k:  {m.retrieval_recall_at_k:.3f}\n"
            f"  confidence_calibration: {m.confidence_calibration_ece:.3f}\n"
            f"  latency_p50_s:          {m.latency_p50_s:.2f}\n"
            f"  latency_p95_s:          {m.latency_p95_s:.2f}\n"
            f"  cache_hit_rate:         {m.cache_hit_rate:.3f}\n"
        )
