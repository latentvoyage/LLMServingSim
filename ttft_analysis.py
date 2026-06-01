"""
TTFT analysis script for LLMServingSim output CSVs.

Reads the per-request CSV produced by `python -m serving --output <csv>`
and reports TTFT P50, P75, P95, P99, plus a CDF plot.

Usage:
    python ttft_analysis.py <output.csv> [--plot ttft_cdf.png]

Example workflow:
    # 1. Generate workload
    python -m workloads.generators.multimodal --num-requests 300 --rate 5 \
        --output workloads/multimodal-10k-300-sps5.jsonl

    # 2. Run simulation (requires ASTRA-Sim compiled)
    python -m serving \
        --cluster-config configs/cluster/single_node_encoder_pd_instance.json \
        --workload workloads/multimodal-10k-300-sps5.jsonl \
        --output outputs/multimodal_run.csv

    # 3. Analyze TTFT
    python ttft_analysis.py outputs/multimodal_run.csv --plot ttft_cdf.png
"""

import argparse
import csv
import sys
import numpy as np


def load_results(csv_path):
    """Load per-request results from sim output CSV."""
    requests = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttft_ns = float(row['TTFT'])
            latency_ns = float(row['latency'])
            tpot_ns = float(row.get('TPOT', 0))
            requests.append({
                'instance_id': row.get('instance id', ''),
                'request_id': row.get('request id', ''),
                'input': int(row.get('input', 0)),
                'output': int(row.get('output', 0)),
                'ttft_ms': ttft_ns / 1e6,
                'latency_ms': latency_ns / 1e6,
                'tpot_ms': tpot_ns / 1e6,
            })
    return requests


def print_stats(requests):
    """Print TTFT statistics."""
    ttfts = np.array([r['ttft_ms'] for r in requests])
    latencies = np.array([r['latency_ms'] for r in requests])

    print("=" * 60)
    print(f"TTFT Analysis — {len(requests)} requests")
    print("=" * 60)
    print(f"\n{'Metric':<12}{'P50':<12}{'P75':<12}{'P95':<12}{'P99':<12}{'Mean':<12}")
    print("-" * 60)
    print(f"{'TTFT (ms)':<12}"
          f"{np.percentile(ttfts, 50):<12.2f}"
          f"{np.percentile(ttfts, 75):<12.2f}"
          f"{np.percentile(ttfts, 95):<12.2f}"
          f"{np.percentile(ttfts, 99):<12.2f}"
          f"{np.mean(ttfts):<12.2f}")
    print(f"{'E2E (ms)':<12}"
          f"{np.percentile(latencies, 50):<12.2f}"
          f"{np.percentile(latencies, 75):<12.2f}"
          f"{np.percentile(latencies, 95):<12.2f}"
          f"{np.percentile(latencies, 99):<12.2f}"
          f"{np.mean(latencies):<12.2f}")

    tpots = np.array([r['tpot_ms'] for r in requests if r['tpot_ms'] > 0])
    if len(tpots) > 0:
        print(f"{'TPOT (ms)':<12}"
              f"{np.percentile(tpots, 50):<12.2f}"
              f"{np.percentile(tpots, 75):<12.2f}"
              f"{np.percentile(tpots, 95):<12.2f}"
              f"{np.percentile(tpots, 99):<12.2f}"
              f"{np.mean(tpots):<12.2f}")
    print()


def plot_cdf(requests, output_path):
    """Generate TTFT CDF plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ttfts = np.sort([r['ttft_ms'] for r in requests])
        cdf = np.arange(1, len(ttfts) + 1) / len(ttfts)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(ttfts, cdf, color='#1f77b4', linewidth=2)
        ax.set_xlabel('TTFT (ms)', fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title(f'TTFT CDF — {len(requests)} requests', fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.50, color='gray', linestyle='--', alpha=0.4)
        ax.axhline(0.75, color='gray', linestyle='--', alpha=0.4)
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.4)

        # Annotate percentiles
        p50 = np.percentile(ttfts, 50)
        p95 = np.percentile(ttfts, 95)
        ax.axvline(p50, color='green', linestyle=':', alpha=0.6, label=f'P50={p50:.1f}ms')
        ax.axvline(p95, color='red', linestyle=':', alpha=0.6, label=f'P95={p95:.1f}ms')
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"CDF plot saved to {output_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


def main():
    parser = argparse.ArgumentParser(description="Analyze TTFT from LLMServingSim output CSV")
    parser.add_argument("csv", help="Path to simulation output CSV")
    parser.add_argument("--plot", type=str, default=None, help="Output CDF plot path (e.g., ttft_cdf.png)")
    args = parser.parse_args()

    requests = load_results(args.csv)
    if not requests:
        print(f"No requests found in {args.csv}")
        sys.exit(1)

    print_stats(requests)

    if args.plot:
        plot_cdf(requests, args.plot)


if __name__ == "__main__":
    main()
