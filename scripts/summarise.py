#!/usr/bin/env python3
"""
Benchmark summarization script for garbled-gpt2.

Reads CSV files produced by garbler and evaluator binaries and generates
Markdown and LaTeX summary tables.

Usage:
    python scripts/summarise.py --csv-pattern "benchmark_*.csv" --output results
"""

import argparse
import csv
import glob
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_csv_files(pattern: str) -> Tuple[List[Dict], List[Dict]]:
    """Read CSV files matching the pattern and separate garbler/evaluator results."""
    garbler_results = []
    evaluator_results = []

    for csv_file in glob.glob(pattern):
        print(f"Reading {csv_file}")
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for field in [
                    "wall_time_ms",
                    "eval_time_ms",
                    "reveal_time_ms",
                    "bytes_sent",
                    "bytes_recv",
                    "total_bytes",
                    "initial_mem_mb",
                    "peak_mem_mb",
                    "final_mem_mb",
                    "mem_delta_mb",
                    "input_size",
                    "run_id",
                    "seed",
                ]:
                    if row[field]:  # Skip empty fields
                        row[field] = float(row[field])

                if row["garble_time_ms"]:  # Only garbler has this field
                    row["garble_time_ms"] = float(row["garble_time_ms"])

                if row["role"] == "garbler":
                    garbler_results.append(row)
                elif row["role"] == "evaluator":
                    evaluator_results.append(row)

    return garbler_results, evaluator_results


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "stddev": 0}

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0,
    }


def generate_markdown_summary(
    garbler_results: List[Dict], evaluator_results: List[Dict]
) -> str:
    """Generate Markdown summary table."""
    md = ["# Garbled-GPT2 Benchmark Results\n"]

    if not garbler_results and not evaluator_results:
        md.append("No benchmark results found.\n")
        return "\n".join(md)

    # Overall summary
    total_runs = len(garbler_results) if garbler_results else len(evaluator_results)
    md.append(f"**Total Runs:** {total_runs}\n")

    if garbler_results:
        input_size = garbler_results[0]["input_size"]
        md.append(f"**Input Size:** {int(input_size)} elements\n")

    # Garbler results
    if garbler_results:
        md.append("## Garbler Performance\n")
        md.append("| Metric | Mean | Median | Min | Max | Std Dev |")
        md.append("|--------|------|--------|-----|-----|---------|")

        metrics = [
            ("Wall Time (ms)", [r["wall_time_ms"] for r in garbler_results]),
            ("Garble Time (ms)", [r["garble_time_ms"] for r in garbler_results]),
            ("Eval Time (ms)", [r["eval_time_ms"] for r in garbler_results]),
            ("Reveal Time (ms)", [r["reveal_time_ms"] for r in garbler_results]),
            ("Bytes Sent", [r["bytes_sent"] for r in garbler_results]),
            ("Bytes Received", [r["bytes_recv"] for r in garbler_results]),
            ("Total Bytes", [r["total_bytes"] for r in garbler_results]),
            ("Peak Memory (MB)", [r["peak_mem_mb"] for r in garbler_results]),
            ("Memory Delta (MB)", [r["mem_delta_mb"] for r in garbler_results]),
        ]

        for metric_name, values in metrics:
            stats = calculate_stats(values)
            md.append(
                f"| {metric_name} | {stats['mean']:.2f} | {stats['median']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['stddev']:.2f} |"
            )

        md.append("")

    # Evaluator results
    if evaluator_results:
        md.append("## Evaluator Performance\n")
        md.append("| Metric | Mean | Median | Min | Max | Std Dev |")
        md.append("|--------|------|--------|-----|-----|---------|")

        metrics = [
            ("Wall Time (ms)", [r["wall_time_ms"] for r in evaluator_results]),
            ("Eval Time (ms)", [r["eval_time_ms"] for r in evaluator_results]),
            ("Reveal Time (ms)", [r["reveal_time_ms"] for r in evaluator_results]),
            ("Bytes Sent", [r["bytes_sent"] for r in evaluator_results]),
            ("Bytes Received", [r["bytes_recv"] for r in evaluator_results]),
            ("Total Bytes", [r["total_bytes"] for r in evaluator_results]),
            ("Peak Memory (MB)", [r["peak_mem_mb"] for r in evaluator_results]),
            ("Memory Delta (MB)", [r["mem_delta_mb"] for r in evaluator_results]),
        ]

        for metric_name, values in metrics:
            stats = calculate_stats(values)
            md.append(
                f"| {metric_name} | {stats['mean']:.2f} | {stats['median']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['stddev']:.2f} |"
            )

        md.append("")

    # Combined performance overview
    if garbler_results and evaluator_results:
        md.append("## Combined Overview\n")
        garbler_wall_times = [r["wall_time_ms"] for r in garbler_results]
        evaluator_wall_times = [r["wall_time_ms"] for r in evaluator_results]
        total_comm_bytes = [r["total_bytes"] for r in garbler_results]

        garbler_stats = calculate_stats(garbler_wall_times)
        evaluator_stats = calculate_stats(evaluator_wall_times)
        comm_stats = calculate_stats(total_comm_bytes)

        md.append(f"**Average Garbler Time:** {garbler_stats['mean']:.2f} ms")
        md.append(f"**Average Evaluator Time:** {evaluator_stats['mean']:.2f} ms")
        md.append(
            f"**Average Communication:** {comm_stats['mean']:.0f} bytes ({comm_stats['mean']/1024:.1f} KB)"
        )
        md.append("")

    return "\n".join(md)


def generate_latex_table(
    garbler_results: List[Dict], evaluator_results: List[Dict]
) -> str:
    """Generate LaTeX summary table."""
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Garbled-GPT2 Benchmark Results}",
        "\\label{tab:benchmark-results}",
        "\\begin{tabular}{|l|r|r|r|r|r|}",
        "\\hline",
        "\\textbf{Metric} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Min} & \\textbf{Max} & \\textbf{Std Dev} \\\\",
        "\\hline",
    ]

    if garbler_results:
        latex.append("\\multicolumn{6}{|c|}{\\textbf{Garbler Performance}} \\\\")
        latex.append("\\hline")

        metrics = [
            ("Wall Time (ms)", [r["wall_time_ms"] for r in garbler_results]),
            ("Garble Time (ms)", [r["garble_time_ms"] for r in garbler_results]),
            ("Eval Time (ms)", [r["eval_time_ms"] for r in garbler_results]),
            ("Reveal Time (ms)", [r["reveal_time_ms"] for r in garbler_results]),
            ("Peak Memory (MB)", [r["peak_mem_mb"] for r in garbler_results]),
            ("Memory Delta (MB)", [r["mem_delta_mb"] for r in garbler_results]),
            ("Total Bytes", [r["total_bytes"] for r in garbler_results]),
        ]

        for metric_name, values in metrics:
            stats = calculate_stats(values)
            latex.append(
                f"{metric_name} & {stats['mean']:.2f} & {stats['median']:.2f} & "
                f"{stats['min']:.2f} & {stats['max']:.2f} & {stats['stddev']:.2f} \\\\"
            )

        latex.append("\\hline")

    if evaluator_results:
        latex.append("\\multicolumn{6}{|c|}{\\textbf{Evaluator Performance}} \\\\")
        latex.append("\\hline")

        metrics = [
            ("Wall Time (ms)", [r["wall_time_ms"] for r in evaluator_results]),
            ("Eval Time (ms)", [r["eval_time_ms"] for r in evaluator_results]),
            ("Reveal Time (ms)", [r["reveal_time_ms"] for r in evaluator_results]),
            ("Peak Memory (MB)", [r["peak_mem_mb"] for r in evaluator_results]),
            ("Memory Delta (MB)", [r["mem_delta_mb"] for r in evaluator_results]),
            ("Total Bytes", [r["total_bytes"] for r in evaluator_results]),
        ]

        for metric_name, values in metrics:
            stats = calculate_stats(values)
            latex.append(
                f"{metric_name} & {stats['mean']:.2f} & {stats['median']:.2f} & "
                f"{stats['min']:.2f} & {stats['max']:.2f} & {stats['stddev']:.2f} \\\\"
            )

    latex.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize garbled-gpt2 benchmark results"
    )
    parser.add_argument(
        "--csv-pattern",
        default="benchmark_*.csv",
        help="Glob pattern for CSV files to process",
    )
    parser.add_argument(
        "--output",
        default="benchmark_summary",
        help="Output filename prefix (without extension)",
    )

    args = parser.parse_args()

    # Read CSV files
    garbler_results, evaluator_results = read_csv_files(args.csv_pattern)

    if not garbler_results and not evaluator_results:
        print(f"No CSV files found matching pattern: {args.csv_pattern}")
        return

    print(
        f"Found {len(garbler_results)} garbler results and {len(evaluator_results)} evaluator results"
    )

    # Generate outputs
    markdown_content = generate_markdown_summary(garbler_results, evaluator_results)
    latex_content = generate_latex_table(garbler_results, evaluator_results)

    # Write outputs
    markdown_file = f"{args.output}.md"
    latex_file = f"{args.output}.tex"

    with open(markdown_file, "w") as f:
        f.write(markdown_content)
    print(f"Markdown summary written to {markdown_file}")

    with open(latex_file, "w") as f:
        f.write(latex_content)
    print(f"LaTeX table written to {latex_file}")


if __name__ == "__main__":
    main()
