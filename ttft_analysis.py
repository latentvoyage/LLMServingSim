"""TTFT CDF analysis for multimodal disaggregated serving.

Variable resolution, variable images, normal distributions for text/output.
Multi-instance: 2 encoders, 2 prefill, 2 decode. Prefix caching modeled.
"""
import random
import numpy as np
import json

import sys
sys.path.insert(0, '.')
from serving.core.encoder_model import EncoderLatencyModel

seed = 42
patch_size = 14
image_prob = 0.8
num_requests = 300

# Variable resolutions with probabilities (models like LLaVA-OneVision, Qwen-VL)
RESOLUTIONS = [224, 336, 448, 672, 896]
RES_PROBS = [0.10, 0.35, 0.30, 0.15, 0.10]  # 336 and 448 most common

# Image count: discrete 1-8, geometrically decaying (most requests have 1-3)
MAX_IMAGES = 8

# Text input: Normal(μ=10000, σ=3000), truncated to [1000, 30000]
TEXT_MU, TEXT_SIGMA = 10000.0, 3000.0
TEXT_MIN, TEXT_MAX = 1000, 30000

# Output tokens: Normal(μ=200, σ=80), truncated to [20, 600]
OUT_MU, OUT_SIGMA = 200.0, 80.0
OUT_MIN, OUT_MAX = 20, 600

# Prefix cache hit ratio: Normal(μ=0.40, σ=0.10), truncated to [0.0, 0.80]
CACHE_MU, CACHE_SIGMA = 0.40, 0.10
CACHE_MIN, CACHE_MAX = 0.0, 0.80

# Multi-instance config
NUM_ENCODERS = 2
NUM_PREFILLS = 2
NUM_DECODES = 2

encoder = EncoderLatencyModel('H100')
link_bw_gbps = 400

# Prefill throughput: tokens/s for H100 (from profiling, ~40k tok/s for Llama-8B scale)
PREFILL_TOKS_PER_SEC = 40000.0


def _truncated_normal(rng, mu, sigma, lo, hi, size=1):
    """Sample from truncated normal via rejection (integer)."""
    samples = []
    while len(samples) < size:
        x = rng.normal(mu, sigma)
        if lo <= x <= hi:
            samples.append(int(round(x)))
    if size == 1:
        return samples[0]
    return samples


def _truncated_normal_float(rng, mu, sigma, lo, hi):
    """Sample a single float from truncated normal via rejection."""
    while True:
        x = rng.normal(mu, sigma)
        if lo <= x <= hi:
            return float(x)


def _sample_num_images(rng):
    """Sample number of images with geometric decay: P(k) ∝ 0.5^(k-1), k=1..8"""
    probs = np.array([0.5**(k-1) for k in range(1, MAX_IMAGES + 1)])
    probs /= probs.sum()
    return rng.choice(np.arange(1, MAX_IMAGES + 1), p=probs)


def simulate_ttft(rate_rps, num_req=300):
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)

    # Generate arrivals (Poisson)
    inter_arrivals_ms = rng.exponential(1000.0 / rate_rps, num_req)
    arrivals_ms = np.cumsum(inter_arrivals_ms)

    requests = []
    for i in range(num_req):
        has_image = py_rng.random() < image_prob
        if has_image:
            n_img = int(_sample_num_images(rng))
            resolutions = rng.choice(RESOLUTIONS, size=n_img, p=RES_PROBS).tolist()
        else:
            n_img = 0
            resolutions = []

        text_in = _truncated_normal(rng, TEXT_MU, TEXT_SIGMA, TEXT_MIN, TEXT_MAX)
        out_toks = _truncated_normal(rng, OUT_MU, OUT_SIGMA, OUT_MIN, OUT_MAX)

        # Cache hit ratio for this request
        cache_hit = _truncated_normal_float(rng, CACHE_MU, CACHE_SIGMA, CACHE_MIN, CACHE_MAX)

        img_toks = sum((r // patch_size) ** 2 for r in resolutions)

        requests.append({
            'arrival_ms': arrivals_ms[i],
            'n_img': n_img,
            'resolutions': resolutions,
            'img_toks': img_toks,
            'text_in': text_in,
            'out_toks': out_toks,
            'cache_hit': cache_hit,
        })

    # Multi-instance simulation: 2E + 2P + 2D
    # Process requests in arrival order, routing to least-loaded instance at each stage
    from collections import Counter

    encoder_free_at_ms = [0.0] * NUM_ENCODERS
    prefill_free_at_ms = [0.0] * NUM_PREFILLS

    ttft_list = [0.0] * num_req

    for i, req in enumerate(requests):
        arr = req['arrival_ms']

        if req['n_img'] == 0:
            # Text-only: skip encoder, go directly to prefill
            best_p = int(np.argmin(prefill_free_at_ms))
            prefill_start = max(arr, prefill_free_at_ms[best_p])
            queue_wait_p = prefill_start - arr

            # Prefill with caching
            total_input = req['text_in']
            cached_tokens = int(total_input * req['cache_hit'])
            compute_tokens = total_input - cached_tokens
            prefill_ms = compute_tokens / PREFILL_TOKS_PER_SEC * 1000.0

            prefill_free_at_ms[best_p] = prefill_start + prefill_ms
            ttft_list[i] = queue_wait_p + prefill_ms
        else:
            # Image request: encoder → transfer → prefill
            # Route to least-loaded encoder
            best_e = int(np.argmin(encoder_free_at_ms))
            encoder_start = max(arr, encoder_free_at_ms[best_e])
            queue_wait_e = encoder_start - arr

            # Encoder compute: group images by resolution
            res_counts = Counter(req['resolutions'])
            enc_ms = 0.0
            for res, count in res_counts.items():
                lat_us = encoder.estimate_total_latency_us(count, resolution=res)
                enc_ms += lat_us / 1000.0

            encoder_free_at_ms[best_e] = encoder_start + enc_ms

            # Transfer embeddings
            transfer_bytes = sum((r // patch_size) ** 2 * 4096 * 2 for r in req['resolutions'])
            transfer_ms = transfer_bytes / (link_bw_gbps * 1e9) * 1000

            # Embeddings ready time
            embeddings_ready_ms = encoder_start + enc_ms + transfer_ms

            # Route to least-loaded prefill instance
            best_p = int(np.argmin(prefill_free_at_ms))
            prefill_start = max(embeddings_ready_ms, prefill_free_at_ms[best_p])
            queue_wait_p = prefill_start - embeddings_ready_ms

            # Prefill with caching (only text tokens can be prefix-cached)
            total_input = req['img_toks'] + req['text_in']
            cached_tokens = int(req['text_in'] * req['cache_hit'])
            compute_tokens = total_input - cached_tokens
            prefill_ms = compute_tokens / PREFILL_TOKS_PER_SEC * 1000.0

            prefill_free_at_ms[best_p] = prefill_start + prefill_ms

            ttft_list[i] = queue_wait_e + enc_ms + transfer_ms + queue_wait_p + prefill_ms

    return np.array(ttft_list)


# Test multiple arrival rates (2P @ 40k tok/s saturates around ~13 rps with 6k compute tokens)
rates = [1, 2, 5, 8, 10, 12, 13]

print("=" * 78)
print("TTFT Statistics — Multi-Instance Disaggregated Serving with Prefix Caching")
print(f"  Instances: {NUM_ENCODERS}E + {NUM_PREFILLS}P + {NUM_DECODES}D")
print(f"  Resolutions: {RESOLUTIONS} (probs={RES_PROBS})")
print(f"  Images: 1-{MAX_IMAGES} (geometric decay), P(has_image)={image_prob}")
print(f"  Text input: Normal(μ={TEXT_MU:.0f}, σ={TEXT_SIGMA:.0f}), clipped [{TEXT_MIN}, {TEXT_MAX}]")
print(f"  Output:     Normal(μ={OUT_MU}, σ={OUT_SIGMA}), clipped [{OUT_MIN}, {OUT_MAX}]")
print(f"  Cache hit:  Normal(μ={CACHE_MU}, σ={CACHE_SIGMA}), clipped [{CACHE_MIN}, {CACHE_MAX}]")
print(f"  Prefill throughput: {PREFILL_TOKS_PER_SEC:.0f} tok/s per instance")
print(f"  N={num_requests} requests, H100, ViT-L/14, Poisson arrivals")
print("=" * 78)
header = f"{'Rate (rps)':<12}{'P50 (ms)':<12}{'P75 (ms)':<12}{'P95 (ms)':<12}{'P99 (ms)':<12}{'Mean (ms)':<12}"
print(header)
print("-" * 78)

all_results = {}
for rate in rates:
    ttfts = simulate_ttft(rate)
    p50 = np.percentile(ttfts, 50)
    p75 = np.percentile(ttfts, 75)
    p95 = np.percentile(ttfts, 95)
    p99 = np.percentile(ttfts, 99)
    mean = np.mean(ttfts)
    all_results[rate] = ttfts
    print(f"{rate:<12}{p50:<12.2f}{p75:<12.2f}{p95:<12.2f}{p99:<12.2f}{mean:<12.2f}")

# Save CDF data
cdf_data = {}
for rate, ttfts in all_results.items():
    sorted_t = np.sort(ttfts)
    cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    cdf_data[str(rate)] = {'ttft_ms': sorted_t.tolist(), 'cdf': cdf.tolist()}

with open('ttft_cdf_data.json', 'w') as f:
    json.dump(cdf_data, f)

# Generate CDF plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Left: CDF
    ax = axes[0]
    for i, rate in enumerate(rates):
        ttfts = all_results[rate]
        sorted_t = np.sort(ttfts)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax.plot(sorted_t, cdf, label=f'{rate} rps', color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel('TTFT (ms)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'TTFT CDF — {NUM_ENCODERS}E+{NUM_PREFILLS}P+{NUM_DECODES}D, Prefix Caching\n'
                 f'(10k text tokens, 40% cache hit, variable resolution)', fontsize=10)
    ax.legend(title='Arrival Rate', fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    # Right: Histogram of encoder latencies (to show smooth distribution)
    ax2 = axes[1]
    # Recompute encoder latencies for the distribution view
    rng_plot = np.random.default_rng(seed)
    py_rng_plot = random.Random(seed)
    enc_latencies = []
    for _ in range(num_requests):
        has_img = py_rng_plot.random() < image_prob
        if has_img:
            n_img = int(_sample_num_images(rng_plot))
            resolutions = rng_plot.choice(RESOLUTIONS, size=n_img, p=RES_PROBS).tolist()
            from collections import Counter
            rc = Counter(resolutions)
            lat = sum(encoder.estimate_total_latency_us(cnt, resolution=res) / 1000.0
                      for res, cnt in rc.items())
            enc_latencies.append(lat)
    ax2.hist(enc_latencies, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Encoder Latency (ms)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Encoder Latency Distribution (N={len(enc_latencies)})\n'
                  f'Resolutions {RESOLUTIONS}, 1-{MAX_IMAGES} imgs (geometric)', fontsize=10)
    ax2.axvline(np.median(enc_latencies), color='red', linestyle='--', label=f'Median={np.median(enc_latencies):.2f}ms')
    ax2.axvline(np.mean(enc_latencies), color='orange', linestyle='--', label=f'Mean={np.mean(enc_latencies):.2f}ms')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ttft_cdf_plot.png', dpi=150)
    print(f"\nCDF plot saved to ttft_cdf_plot.png")
    print(f"\nEncoder latency stats: median={np.median(enc_latencies):.2f}ms, "
          f"mean={np.mean(enc_latencies):.2f}ms, "
          f"min={np.min(enc_latencies):.2f}ms, max={np.max(enc_latencies):.2f}ms")
except ImportError:
    print("\nmatplotlib not available — CDF data saved to JSON only")
