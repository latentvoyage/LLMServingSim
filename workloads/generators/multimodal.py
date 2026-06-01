"""
Multimodal workload generator for LLMServingSim.

Generates JSONL traces with:
- Variable image counts (1-8, geometric decay) and resolutions (224-896px)
- Text input ~N(10000, 3000) tokens with shared prefixes for ~40% cache hit
- Output ~N(200, 80) tokens
- Poisson arrivals

Cache hit strategy: requests share a common "system prompt" prefix of ~4000 tokens
(~40% of the 10k mean input). This produces ~40% prefix cache hit ratio when
prefix caching is enabled with block_size=16.

Usage:
    python -m workloads.generators.multimodal \
        --num-requests 300 \
        --rate 5 \
        --output workloads/multimodal-10k-300-sps5.jsonl
"""

import argparse
import json
import numpy as np
from pathlib import Path


# Variable resolutions (models like LLaVA-OneVision, Qwen-VL)
RESOLUTIONS = [224, 336, 448, 672, 896]
RES_PROBS = [0.10, 0.35, 0.30, 0.15, 0.10]

MAX_IMAGES = 8


def _truncated_normal_int(rng, mu, sigma, lo, hi):
    """Sample integer from truncated normal."""
    while True:
        x = rng.normal(mu, sigma)
        if lo <= x <= hi:
            return int(round(x))


def _truncated_normal_float(rng, mu, sigma, lo, hi):
    """Sample float from truncated normal."""
    while True:
        x = rng.normal(mu, sigma)
        if lo <= x <= hi:
            return float(x)


def _sample_num_images(rng):
    """Geometric decay: P(k) ∝ 0.5^(k-1), k=1..MAX_IMAGES."""
    probs = np.array([0.5**(k-1) for k in range(1, MAX_IMAGES + 1)])
    probs /= probs.sum()
    return int(rng.choice(np.arange(1, MAX_IMAGES + 1), p=probs))


def generate_multimodal_trace(
    num_requests: int = 300,
    rate_rps: float = 5.0,
    patch_size: int = 14,
    image_prob: float = 0.8,
    text_mu: float = 10000.0,
    text_sigma: float = 3000.0,
    text_min: int = 1000,
    text_max: int = 30000,
    output_mu: float = 200.0,
    output_sigma: float = 80.0,
    output_min: int = 20,
    output_max: int = 600,
    cache_hit_target: float = 0.40,
    cache_sigma: float = 0.10,
    num_prefix_groups: int = 5,
    seed: int = 42,
):
    """Generate a multimodal workload trace with prefix sharing for cache hits.

    Cache strategy: Requests are assigned to one of `num_prefix_groups` groups.
    Each group shares a common prefix of length ~(cache_hit_target * text_length).
    Requests within a group share this prefix + have unique suffixes.
    The shared prefix length per-request is sampled from N(cache_hit_target, cache_sigma).

    This mimics real workloads where many users share a system prompt but have
    unique queries (e.g., same chatbot template, different questions).

    Args:
        num_requests: Number of requests to generate.
        rate_rps: Request arrival rate (requests per second).
        patch_size: ViT patch size.
        image_prob: Probability a request contains at least one image.
        text_mu/sigma/min/max: Text input token count distribution.
        output_mu/sigma/min/max: Output token count distribution.
        cache_hit_target: Target prefix cache hit ratio (~0.40 = 40%).
        cache_sigma: Variance in per-request cache hit ratio.
        num_prefix_groups: Number of distinct "system prompt" groups.
        seed: Random seed.

    Yields:
        dict: One request per yield.
    """
    rng = np.random.default_rng(seed)

    # Generate shared prefix token sequences for each group
    # Each group's prefix is a different "system prompt" template
    # Use a global token counter to ensure unique IDs
    token_counter = 1

    # Pre-generate group prefixes (large enough for max possible shared length)
    max_shared_len = int(text_max * 0.8)  # max possible shared prefix
    group_prefixes = []
    for _ in range(num_prefix_groups):
        prefix = list(range(token_counter, token_counter + max_shared_len))
        token_counter += max_shared_len
        group_prefixes.append(prefix)

    # Poisson arrivals
    inter_arrivals = rng.exponential(1e9 / rate_rps, num_requests)
    arrivals = np.cumsum(inter_arrivals).astype(int)

    for i in range(num_requests):
        # Image sampling
        has_image = rng.random() < image_prob
        if has_image:
            n_img = _sample_num_images(rng)
            resolutions = rng.choice(RESOLUTIONS, size=n_img, p=RES_PROBS).tolist()
        else:
            n_img = 0
            resolutions = []

        image_tokens = sum((r // patch_size) ** 2 for r in resolutions)

        # Text tokens (Normal distribution)
        text_input = _truncated_normal_int(rng, text_mu, text_sigma, text_min, text_max)
        output_toks = _truncated_normal_int(rng, output_mu, output_sigma, output_min, output_max)

        # Cache hit ratio for this request
        cache_ratio = _truncated_normal_float(rng, cache_hit_target, cache_sigma, 0.0, 0.80)

        # Determine shared prefix length (block-aligned to 16)
        shared_len = int(text_input * cache_ratio)
        shared_len = (shared_len // 16) * 16  # align to block_size=16

        # Assign to a prefix group (round-robin with some randomness)
        group_id = i % num_prefix_groups

        # Build input_tok_ids: shared_prefix + unique_suffix
        shared_prefix = group_prefixes[group_id][:shared_len]
        unique_len = text_input - shared_len
        unique_suffix = list(range(token_counter, token_counter + unique_len))
        token_counter += unique_len

        input_tok_ids = shared_prefix + unique_suffix

        # Output token IDs (unique per request)
        output_tok_ids = list(range(token_counter, token_counter + output_toks))
        token_counter += output_toks

        # Total input for the sim = text + image embeddings
        total_input = text_input + image_tokens

        # Determine primary resolution (for the image_resolution field)
        # Use the most common resolution in this request's images
        if resolutions:
            from collections import Counter
            res_counter = Counter(resolutions)
            primary_res = res_counter.most_common(1)[0][0]
        else:
            primary_res = 0

        yield {
            "input_toks": total_input,
            "output_toks": output_toks,
            "arrival_time_ns": int(arrivals[i]),
            "input_tok_ids": input_tok_ids,
            "output_tok_ids": output_tok_ids,
            "image_tokens": image_tokens,
            "num_images": n_img,
            "image_resolution": primary_res,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate multimodal workload traces")
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--rate", type=float, default=5.0, help="Requests per second")
    parser.add_argument("--patch-size", type=int, default=14, help="ViT patch size")
    parser.add_argument("--image-prob", type=float, default=0.8, help="Prob of image in request")
    parser.add_argument("--text-mu", type=float, default=10000.0, help="Mean text input tokens")
    parser.add_argument("--text-sigma", type=float, default=3000.0, help="Std text input tokens")
    parser.add_argument("--output-mu", type=float, default=200.0, help="Mean output tokens")
    parser.add_argument("--output-sigma", type=float, default=80.0, help="Std output tokens")
    parser.add_argument("--cache-hit-target", type=float, default=0.40, help="Target cache hit ratio")
    parser.add_argument("--num-prefix-groups", type=int, default=5, help="Number of shared prefix groups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="workloads/multimodal-10k-300-sps5.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for req in generate_multimodal_trace(
            num_requests=args.num_requests,
            rate_rps=args.rate,
            patch_size=args.patch_size,
            image_prob=args.image_prob,
            text_mu=args.text_mu,
            text_sigma=args.text_sigma,
            output_mu=args.output_mu,
            output_sigma=args.output_sigma,
            cache_hit_target=args.cache_hit_target,
            num_prefix_groups=args.num_prefix_groups,
            seed=args.seed,
        ):
            f.write(json.dumps(req) + "\n")
            count += 1

    print(f"Generated {count} requests → {output_path}")
    # Summary stats
    print(f"  Text input: N(μ={args.text_mu:.0f}, σ={args.text_sigma:.0f})")
    print(f"  Output: N(μ={args.output_mu:.0f}, σ={args.output_sigma:.0f})")
    print(f"  Images: 1-{MAX_IMAGES} (geometric), P(has_image)={args.image_prob}")
    print(f"  Resolutions: {RESOLUTIONS}")
    print(f"  Cache hit target: {args.cache_hit_target*100:.0f}% ({args.num_prefix_groups} prefix groups)")


if __name__ == "__main__":
    main()
