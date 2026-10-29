"""
Big Data Systems Scalability & Storage Design Benchmark
Measures distributed ETL throughput, scaling latency, Parquet compression efficiency,
and memory consumption across expanding patient cohort scales.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import sys
import time
import shutil
import platform
import psutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.big_data.spark_session import get_spark_session
from src.big_data.etl_pipeline import run_etl_pipeline
from data.synthea_generator import generate_synthea_cohort


def measure_directory_size_mb(path):
    """Calculates recursive directory size in Megabytes."""
    total_bytes = 0
    if not os.path.exists(path):
        return 0.0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(fp)
            except OSError:
                pass
    return total_bytes / (1024.0 * 1024.0)


def run_scaling_benchmark(scales=[2500, 5000, 10000], output_dir="reports/figures"):
    """
    Benchmarks distributed Spark ETL execution time, record throughput, and storage compression
    across expanding patient cohort scales. Cleans up temporary benchmark cohorts upon completion.
    """
    os.makedirs(output_dir, exist_ok=True)
    benchmark_results = []

    cpu_count = os.cpu_count() or 4
    total_ram_gb = round(psutil.virtual_memory().total / (1024.0 ** 3), 1)
    os_name = f"{platform.system()} {platform.release()}"

    print("=" * 70)
    print("  BIG DATA SCALABILITY BENCHMARK SUITE")
    print(f"  System: {os_name} | {cpu_count} CPU Cores | {total_ram_gb} GB RAM")
    print(f"  Spark Execution: Local multi-threaded (master='local[*]')")
    print("=" * 70)

    spark = get_spark_session(app_name="BDSRAE-Scalability-Benchmark")

    for n in scales:
        temp_raw = f"data/raw_bench_{n}"
        temp_proc = f"data/proc_bench_{n}"
        
        print(f"\n--- Benchmarking Scale: {n:,} Patients ---")
        try:
            # 1. Generate synthetic cohort in dedicated benchmark directory
            generate_synthea_cohort(n_patients=n, output_dir=temp_raw)
            raw_size_mb = measure_directory_size_mb(temp_raw)

            # 2. Measure ETL Pipeline
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss / (1024.0 * 1024.0)

            parquet_path = run_etl_pipeline(raw_dir=temp_raw, processed_dir=temp_proc, spark=spark)

            elapsed_sec = time.time() - start_time
            end_mem = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
            parquet_size_mb = measure_directory_size_mb(parquet_path)
            compression_ratio = raw_size_mb / parquet_size_mb if parquet_size_mb > 0 else 0.0
            throughput = n / elapsed_sec if elapsed_sec > 0 else 0.0

            result = {
                "scale_n_patients": n,
                "raw_size_mb": round(raw_size_mb, 2),
                "parquet_size_mb": round(parquet_size_mb, 2),
                "compression_ratio": round(compression_ratio, 2),
                "elapsed_seconds": round(elapsed_sec, 2),
                "throughput_rec_per_sec": round(throughput, 1),
                "memory_usage_mb": round(abs(end_mem - start_mem), 2),
                "cpu_cores": cpu_count,
                "system_ram_gb": total_ram_gb
            }
            benchmark_results.append(result)
            print(f"[+] Result: {elapsed_sec:.2f}s | {throughput:.1f} rec/s | Storage: {raw_size_mb:.1f}MB CSV -> {parquet_size_mb:.2f}MB Parquet ({compression_ratio:.1f}x compression)")

        finally:
            # Clean up temporary benchmark directories so primary dataset is unaffected
            shutil.rmtree(temp_raw, ignore_errors=True)
            shutil.rmtree(temp_proc, ignore_errors=True)

    spark.stop()

    if not benchmark_results:
        print("[!] No benchmark results recorded.")
        return pd.DataFrame()

    # Save benchmark table
    df_results = pd.DataFrame(benchmark_results)
    csv_out = os.path.join(output_dir, "scalability_benchmark.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[OK] Benchmark metrics saved to: {csv_out}")

    # Plot scalability figures
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Runtime Scaling
    axes[0].plot(df_results["scale_n_patients"], df_results["elapsed_seconds"], marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    axes[0].set_title("PySpark ETL Runtime vs. Cohort Scale", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Synthetic Patients", fontsize=11)
    axes[0].set_ylabel("Execution Time (seconds)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Storage Compression Design (CSV vs Partitioned Parquet)
    x = np.arange(len(df_results))
    width = 0.35
    axes[1].bar(x - width/2, df_results["raw_size_mb"], width, label="Raw CSV Tables", color="#d62728", alpha=0.85)
    axes[1].bar(x + width/2, df_results["parquet_size_mb"], width, label="Partitioned Parquet", color="#2ca02c", alpha=0.85)
    axes[1].set_title("Storage Footprint & Compression Efficiency", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Cohort Scale", fontsize=11)
    axes[1].set_ylabel("Storage Size (MB)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{n//1000}k" if n >= 1000 else str(n) for n in df_results["scale_n_patients"]])
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Record Processing Throughput
    axes[2].plot(df_results["scale_n_patients"], df_results["throughput_rec_per_sec"], marker="s", color="#9467bd", linewidth=2.5, markersize=8)
    axes[2].set_title("PySpark Processing Throughput", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Number of Synthetic Patients", fontsize=11)
    axes[2].set_ylabel("Throughput (Patients / second)", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "big_data_scaling.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Scalability visualization saved to: {plot_path}")

    return df_results


if __name__ == "__main__":
    run_scaling_benchmark()
