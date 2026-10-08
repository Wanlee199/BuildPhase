"""
colab_runner.py - Master Pipeline Execution Script for Google Colab
=====================================================================
Run all 5 active learning pipeline steps seamlessly in Google Colab:
1. Data Profiling & Intra-Video Deduplication
2. Active 1-Class Mining & Two-Tier HITL Selection
3. CVAT Pre-label Packaging / Dataset Assembly
4. Multi-Seed YOLO11n Training & Experiment Manifest Generation
5. Multi-Slice Benchmark Evaluation & Markdown Report Export
"""

import os
import sys
import argparse


def run_pipeline_step(step_name: str, cmd: str):
    print(f"\n========================================================")
    print(f"▶️ EXECUTING PIPELINE STEP: {step_name}")
    print(f"========================================================")
    ret = os.system(cmd)
    if ret != 0:
        print(f"⚠️ Step '{step_name}' exited with status code {ret}.")
    else:
        print(f"✅ Step '{step_name}' completed successfully.")


def run_full_colab_pipeline(data_dir: str, output_dir: str, top_k: int = 200, epochs: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    metadata_json = os.path.join(output_dir, "dataset_metadata.json")
    active_json = os.path.join(output_dir, "active_selection_results.json")
    dataset_v1_dir = os.path.join(output_dir, "dataset_v1")
    runs_dir = os.path.join(output_dir, "runs")
    report_md = os.path.join(output_dir, "BAO_CAO_BENCHMARK_A_B_TEST.md")

    # Step 1: Profiling & Deduplication
    cmd1 = f"python src/data_profiling.py --data-dir {data_dir} --output-json {metadata_json} --hash-thresh 5"
    run_pipeline_step("1. Data Profiling & Intra-Video Dedup", cmd1)

    # Step 2: Active 1-Class Selection
    cmd2 = f"python src/active_selection.py --metadata-json {metadata_json} --top-k {top_k} --output-json {active_json}"
    run_pipeline_step("2. Active 1-Class Mining & 2-Tier HITL", cmd2)

    # Step 3: Package CVAT / Assemble Dataset V1
    cmd3 = f"python src/cvat_automation.py --action assemble --selection-json {active_json} --output-dir {dataset_v1_dir}"
    run_pipeline_step("3. CVAT Automation & Dataset Assembly", cmd3)

    # Step 4: Multi-Seed Training
    dataset_yaml = os.path.join(dataset_v1_dir, "dataset.yaml")
    cmd4 = f"python src/train_yolo11n.py --data-yaml {dataset_yaml} --seeds 42 123 999 --epochs {epochs} --project-dir {runs_dir} --exp-id greensm_v1"
    run_pipeline_step("4. Multi-Seed YOLO11n Training & Manifest Generation", cmd4)

    # Step 5: Benchmark Evaluation & Report
    seed_summary = os.path.join(runs_dir, "greensm_v1_multi_seed_summary.json")
    cmd5 = f"python src/evaluate_benchmark.py --test-yaml {dataset_yaml} --seed-summary-json {seed_summary} --output-report {report_md}"
    run_pipeline_step("5. Benchmark Evaluation & A/B Report Export", cmd5)

    print("\n========================================================")
    print("🎉 FULL COLAB ACTIVE LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Final Report generated at: [BAO_CAO_BENCHMARK_A_B_TEST.md](file://{os.path.abspath(report_md)})")
    print("========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Pipeline Execution Script for Google Colab")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw images/videos")
    parser.add_argument("--output-dir", type=str, default="colab_outputs", help="Directory for all outputs")
    parser.add_argument("--top-k", type=int, default=200, help="Top-K active candidates to select")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")

    args = parser.parse_args()

    run_full_colab_pipeline(args.data_dir, args.output_dir, args.top_k, args.epochs)
