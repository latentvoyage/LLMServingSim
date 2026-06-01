"""
Multimodal workload generator for LLMServingSim.

Generates JSONL traces with image tokens and text tokens for
encoder-prefill-decode disaggregated serving simulation.

Usage:
    python -m workloads.generators.multimodal \
        --num-requests 100 \
        --rate 10 \
        --output workloads/multimodal-100-sps10.jsonl
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path


def generate_multimodal_trace(
    num_requests: int,
    rate_rps: float,
    resolution: int = 336,
    patch_size: int = 14,
    max_images: int = 4,
    image_prob: float = 0.8,
    text_input_range: tuple = (20, 200),
    output_range: tuple = (50, 300),
    seed: int = 42,
):
    """Generate a multimodal workload trace.

    Args:
        num_requests: Number of requests to generate.
        rate_rps: Request arrival rate (requests per second).
        resolution: Image resolution in pixels (square).
        patch_size: ViT patch size.
        max_images: Max images per request.
        image_prob: Probability a request contains at least one image.
        text_input_range: (min, max) text input token range.
        output_range: (min, max) output token range.
        seed: Random seed.

    Yields:
        dict: One request per yield.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    patches_per_image = (resolution // patch_size) ** 2  # e.g. 576 for 336/14

    # Poisson arrivals
    inter_arrivals = rng.exponential(1e9 / rate_rps, num_requests)
    arrivals = np.cumsum(inter_arrivals).astype(int)

    for i, arrival in enumerate(arrivals):
        has_image = py_rng.random() < image_prob
        if has_image:
            num_images = py_rng.randint(1, max_images)
        else:
            num_images = 0

        image_tokens = num_images * patches_per_image
        text_input = py_rng.randint(*text_input_range)
        output_toks = py_rng.randint(*output_range)

        yield {
            "input_toks": text_input + image_tokens,  # Total input = text + image embeddings
            "output_toks": output_toks,
            "arrival_time_ns": int(arrival),
            "image_tokens": image_tokens,
            "image_resolution": resolution,
            "num_images": num_images,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate multimodal workload traces")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--rate", type=float, default=10.0, help="Requests per second")
    parser.add_argument("--resolution", type=int, default=336, help="Image resolution (px)")
    parser.add_argument("--patch-size", type=int, default=14, help="ViT patch size")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--image-prob", type=float, default=0.8, help="Prob of image in request")
    parser.add_argument("--text-input-min", type=int, default=20)
    parser.add_argument("--text-input-max", type=int, default=200)
    parser.add_argument("--output-min", type=int, default=50)
    parser.add_argument("--output-max", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="workloads/multimodal-100-sps10.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for req in generate_multimodal_trace(
            num_requests=args.num_requests,
            rate_rps=args.rate,
            resolution=args.resolution,
            patch_size=args.patch_size,
            max_images=args.max_images,
            image_prob=args.image_prob,
            text_input_range=(args.text_input_min, args.text_input_max),
            output_range=(args.output_min, args.output_max),
            seed=args.seed,
        ):
            f.write(json.dumps(req) + "\n")
            count += 1

    print(f"Generated {count} multimodal requests to {output_path}")


if __name__ == "__main__":
    main()
