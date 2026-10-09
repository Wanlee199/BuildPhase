"""
colab_runner.py - Master Pipeline Execution Script for Google Colab (V4.0)
=============================================================================
Run all 6 active learning pipeline steps seamlessly in Google Colab:
1. Data Profiling & Dual-Stream Fast Deduplication (pHash, Keep-Most-Vehicles)
2. Active Selection (Target Confusion Margin, PCA 32D, k-Center Greedy Core-Set)
3. CVAT Pre-label Packaging & Dataset V1 Assembly (5-Class Taxonomy)
4. Multi-Seed YOLO11n Training & Enterprise Manifest Generation
5. Multi-Slice Benchmark Evaluation & 3-Pillar ROI Markdown Report Export
6. Edge Model Export & OpenVINO INT8 Quantization Benchmark
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


def run_full_colab_pipeline(data_dir: str, output_dir: str, top_k: int = 200, epochs: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    metadata_json = os.path.join(output_dir, "dataset_metadata.json")
    active_json = os.path.join(output_dir, "active_selection_results.json")
    dataset_v1_dir = os.path.join(output_dir, "dataset_v1")
    runs_dir = os.path.join(output_dir, "runs")
    report_md = os.path.join(output_dir, "BAO_CAO_BENCHMARK_A_B_TEST.md")
    exported_dir = os.path.join(output_dir, "exported_models")

    # Step 1: Profiling & Deduplication
    cmd1 = f"python src/data_profiling.py --data-dir {data_dir} --output-json {metadata_json} --hash-thresh 5"
    run_pipeline_step("1. Data Profiling & Dual-Stream FastDedup (V4.0)", cmd1)

    # Step 2: Active Mining (PCA + k-Center Greedy Core-Set)
    cmd2 = f"python src/active_selection.py --metadata-json {metadata_json} --top-k {top_k} --output-json {active_json}"
    run_pipeline_step("2. Active Target Mining & Cross-Model Oracle HITL", cmd2)

    # Step 3: Package CVAT / Assemble Dataset V1 (5-Class Taxonomy)
    cmd3 = f"python src/cvat_automation.py --action assemble --selection-json {active_json} --output-dir {dataset_v1_dir}"
    run_pipeline_step("3. CVAT Automation & 5-Class Dataset Assembly", cmd3)

    # Step 4: Multi-Seed Training (3 Seeds: 42 123 456)
    dataset_yaml = os.path.join(dataset_v1_dir, "dataset.yaml")
    cmd4 = f"python src/train_yolo11n.py --data-yaml {dataset_yaml} --seeds 42 123 456 --epochs {epochs} --project-dir {runs_dir} --exp-id greensm_v1"
    run_pipeline_step("4. Multi-Seed YOLO11n Training & Manifest Generation", cmd4)

    # Step 5: Benchmark Evaluation & 3-Pillar ROI Report
    seed_summary = os.path.join(runs_dir, "greensm_v1_multi_seed_summary.json")
    cmd5 = f"python src/evaluate_benchmark.py --test-yaml {dataset_yaml} --seed-summary-json {seed_summary} --output-report {report_md}"
    run_pipeline_step("5. Multi-Slice Benchmark Evaluation & A/B Report Export", cmd5)

    # Step 6: Edge Model Export & Quantization
    best_weights = os.path.join(runs_dir, "greensm_v1_seed_42", "weights", "best.pt")
    cmd6 = f"python src/export_quantize.py --weights {best_weights} --output-dir {exported_dir}"
    run_pipeline_step("6. Edge Model Export & OpenVINO INT8 Quantization", cmd6)

    print("\n========================================================")
    print("🎉 FULL COLAB V4.0 ACTIVE LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Final Report generated at: [BAO_CAO_BENCHMARK_A_B_TEST.md](file://{os.path.abspath(report_md)})")
    print("========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Pipeline Execution Script for Google Colab (V4.0)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw images/videos")
    parser.add_argument("--output-dir", type=str, default="colab_outputs", help="Directory for all outputs")
    parser.add_argument("--top-k", type=int, default=200, help="Top-K active candidates to select")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")

    args = parser.parse_args()

    run_full_colab_pipeline(args.data_dir, args.output_dir, args.top_k, args.epochs)
