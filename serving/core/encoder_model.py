"""
Analytical encoder latency model for vision transformers (ViT).

Uses a per-operation roofline model to determine whether each kernel is
compute-bound or memory-bound, then applies tile/wave quantization to
estimate realistic GPU utilization. This avoids the inaccuracy of a single
flat utilization factor.

Reference latencies (ViT-L/14 on H100, batch=1): ~3-5ms measured.
"""

import math


# Hardware specs needed for roofline analysis
DEFAULT_HARDWARE = {
    "H100": {
        "peak_tflops_bf16": 989.0,     # BF16 Tensor Core TFLOPS
        "mem_bw_gbps": 3350.0,         # HBM3 bandwidth GB/s
        "num_sms": 132,                # Streaming Multiprocessors
        "l2_cache_bytes": 50 * 1024 * 1024,  # 50 MB L2
        "kernel_launch_us": 3.5,       # Average kernel launch overhead (us)
        "gemm_tile_m": 64,             # Typical GEMM tile dimension (M)
        "gemm_tile_n": 64,             # Typical GEMM tile dimension (N)
    },
    "A100": {
        "peak_tflops_bf16": 312.0,
        "mem_bw_gbps": 2039.0,
        "num_sms": 108,
        "l2_cache_bytes": 40 * 1024 * 1024,
        "kernel_launch_us": 4.0,
        "gemm_tile_m": 64,
        "gemm_tile_n": 64,
    },
    "RTXPRO6000": {
        "peak_tflops_bf16": 209.5,
        "mem_bw_gbps": 1152.0,
        "num_sms": 96,
        "l2_cache_bytes": 96 * 1024 * 1024,
        "kernel_launch_us": 4.5,
        "gemm_tile_m": 64,
        "gemm_tile_n": 64,
    },
}

# Default ViT architecture configs
DEFAULT_VIT_CONFIGS = {
    "vit-large-patch14": {
        "hidden_size": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "intermediate_size": 4096,
        "patch_size": 14,
        "image_size": 336,
    },
    "vit-huge-patch14": {
        "hidden_size": 1280,
        "num_heads": 16,
        "num_layers": 32,
        "intermediate_size": 5120,
        "patch_size": 14,
        "image_size": 448,
    },
    "siglip-so400m": {
        "hidden_size": 1152,
        "num_heads": 16,
        "num_layers": 27,
        "intermediate_size": 4304,
        "patch_size": 14,
        "image_size": 384,
    },
}


class EncoderLatencyModel:
    """Analytical latency estimator for vision encoder (ViT-style).

    Uses a roofline model per operation: each kernel is either compute-bound
    (time = FLOPs / effective_peak) or memory-bound (time = bytes / bandwidth),
    whichever is larger. GEMM utilization accounts for tile/wave quantization
    on the specific GPU's SM count.
    """

    def __init__(self, hardware: str, vit_config: dict = None, vit_name: str = "vit-large-patch14",
                 calibration_factor: float = 1.8):
        """
        Args:
            hardware: Hardware name (must be in DEFAULT_HARDWARE or provide custom).
            vit_config: Dict with {hidden_size, num_heads, num_layers, intermediate_size, patch_size, image_size}.
            vit_name: Name of a default ViT config to use if vit_config is None.
            calibration_factor: Multiplier to close gap between roofline estimate and
                reality. Accounts for pipeline bubbles, unmodeled ops, runtime overhead.
                Default 1.8 calibrated against measured ViT-L/14 on H100 (~3.5ms real
                vs ~1.9ms roofline). Set to 1.0 to get raw roofline estimate.
        """
        if hardware in DEFAULT_HARDWARE:
            self.hw = DEFAULT_HARDWARE[hardware]
        else:
            raise ValueError(f"Unknown hardware '{hardware}'. Known: {list(DEFAULT_HARDWARE.keys())}")

        if vit_config is not None:
            self.vit = vit_config
        elif vit_name in DEFAULT_VIT_CONFIGS:
            self.vit = DEFAULT_VIT_CONFIGS[vit_name]
        else:
            raise ValueError(f"Unknown vit_name '{vit_name}'. Known: {list(DEFAULT_VIT_CONFIGS.keys())}")

        # Derived architecture params
        self.hidden = self.vit["hidden_size"]
        self.heads = self.vit["num_heads"]
        self.head_dim = self.hidden // self.heads
        self.ffn_dim = self.vit["intermediate_size"]
        self.num_layers = self.vit["num_layers"]
        self.patch_size = self.vit["patch_size"]
        self.image_size = self.vit["image_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.calibration_factor = calibration_factor

        # Roofline ridge point: ops/byte where compute and memory are balanced
        # Below this → memory-bound, above → compute-bound
        self.ridge_point = (self.hw["peak_tflops_bf16"] * 1e12) / (self.hw["mem_bw_gbps"] * 1e9)

    # -----------------------------------------------------------------------
    # Roofline core: per-operation latency estimation
    # -----------------------------------------------------------------------

    def _gemm_latency_us(self, M: int, N: int, K: int, num_batches: int = 1) -> float:
        """Estimate GEMM latency using roofline + tile/wave quantization.

        For batched GEMM (e.g., batched attention), num_batches = batch * heads.
        For standard GEMM, num_batches = 1 and M includes the batch dimension.

        Args:
            M: Rows of output (typically batch*seq for linear layers)
            N: Columns of output (output features)
            K: Shared dimension (input features)
            num_batches: Number of independent GEMM instances (for batched ops)
        """
        fp_bytes = 2  # bf16

        # FLOPs for one GEMM instance
        flops_per = 2 * M * N * K
        total_flops = flops_per * num_batches

        # Memory traffic per GEMM (assumes weights loaded from HBM, activations may be cached)
        # Input A: M*K, Weight B: K*N, Output C: M*N
        bytes_per = (M * K + K * N + M * N) * fp_bytes
        total_bytes = bytes_per * num_batches

        # Arithmetic intensity
        ai = total_flops / total_bytes if total_bytes > 0 else float('inf')

        # Tile/wave quantization: how efficiently do tiles map to SMs?
        tile_m = self.hw["gemm_tile_m"]
        tile_n = self.hw["gemm_tile_n"]
        num_sms = self.hw["num_sms"]

        tiles_m = math.ceil(M / tile_m)
        tiles_n = math.ceil(N / tile_n)
        total_tiles = tiles_m * tiles_n * num_batches

        # Wave quantization: tiles are dispatched in waves of num_sms
        num_waves = math.ceil(total_tiles / num_sms)
        # Effective tiles (including partial last wave)
        ideal_tiles = num_waves * num_sms
        wave_efficiency = total_tiles / ideal_tiles if ideal_tiles > 0 else 1.0

        # Tile padding efficiency: how much of each tile is useful work
        used_m = M / (tiles_m * tile_m)  # fraction of tile rows used
        used_n = N / (tiles_n * tile_n)  # fraction of tile cols used
        tile_efficiency = used_m * used_n

        # Combined GPU efficiency for this GEMM
        gpu_efficiency = wave_efficiency * tile_efficiency
        # Clamp to reasonable range (never below 15% — there's always some overhead)
        gpu_efficiency = max(gpu_efficiency, 0.15)

        # Roofline: time is max of compute-bound and memory-bound
        compute_time_s = total_flops / (self.hw["peak_tflops_bf16"] * 1e12 * gpu_efficiency)
        memory_time_s = total_bytes / (self.hw["mem_bw_gbps"] * 1e9)

        # The binding constraint
        time_s = max(compute_time_s, memory_time_s)

        # Kernel launch overhead
        time_us = time_s * 1e6 + self.hw["kernel_launch_us"]
        return time_us

    def _flash_attention_latency_us(self, batch: int, heads: int, seq_len: int) -> float:
        """Estimate FlashAttention-2 latency for bidirectional (encoder) attention.

        FlashAttention tiles the computation to avoid materializing the full N×N
        attention matrix. Memory access is O(N) per query tile, not O(N²).

        For encoder (bidirectional, no causal mask), the work is:
        - Load Q tiles, stream K/V tiles, accumulate output
        - FLOPs still O(N²) per head, but memory is O(N)
        """
        fp_bytes = 2
        # FLOPs: 2 * batch * heads * seq² * head_dim (scores) + same for weighted sum
        flops = 2 * 2 * batch * heads * seq_len * seq_len * self.head_dim

        # FlashAttention memory access pattern:
        # Q: batch * heads * seq * head_dim (loaded once)
        # K, V: batch * heads * seq * head_dim (streamed in tiles)
        # O: batch * heads * seq * head_dim (written once)
        # Total ≈ 4 * batch * heads * seq * head_dim * fp_bytes
        # Plus softmax intermediate (on-chip, not HBM)
        bytes_accessed = 4 * batch * heads * seq_len * self.head_dim * fp_bytes

        # FlashAttention-specific efficiency:
        # Block sizes are typically 128 for seq dimension
        fa_block = 128
        num_q_blocks = math.ceil(seq_len / fa_block)
        num_kv_blocks = math.ceil(seq_len / fa_block)
        total_blocks = batch * heads * num_q_blocks * num_kv_blocks

        # Wave efficiency over SMs
        num_sms = self.hw["num_sms"]
        # FA parallelizes over batch * heads * q_blocks
        parallel_dim = batch * heads * num_q_blocks
        num_waves = math.ceil(parallel_dim / num_sms)
        wave_eff = parallel_dim / (num_waves * num_sms)

        # FA typically achieves 60-80% of peak on compute-bound attention
        fa_efficiency = wave_eff * 0.75  # 75% base efficiency for FA kernel
        fa_efficiency = max(fa_efficiency, 0.15)

        compute_time_s = flops / (self.hw["peak_tflops_bf16"] * 1e12 * fa_efficiency)
        memory_time_s = bytes_accessed / (self.hw["mem_bw_gbps"] * 1e9)

        time_us = max(compute_time_s, memory_time_s) * 1e6 + self.hw["kernel_launch_us"]
        return time_us

    def _elementwise_latency_us(self, num_elements: int, reads: int = 1, writes: int = 1) -> float:
        """Memory-bound elementwise op (LayerNorm, GELU, residual add).

        These ops have negligible compute — entirely limited by memory bandwidth.
        """
        fp_bytes = 2
        bytes_accessed = num_elements * fp_bytes * (reads + writes)
        time_s = bytes_accessed / (self.hw["mem_bw_gbps"] * 1e9)
        time_us = time_s * 1e6 + self.hw["kernel_launch_us"]
        return time_us

    # -----------------------------------------------------------------------
    # Per-layer latency (architecture-aware)
    # -----------------------------------------------------------------------

    def _patch_embed_us(self, num_images: int) -> float:
        """Patch embedding conv2d, reshaped as GEMM."""
        # Conv2d(3, hidden, kernel=patch_size, stride=patch_size) on each patch
        # Equivalent to GEMM: M=num_images*num_patches, K=patch_size²*3, N=hidden
        M = num_images * self.num_patches
        K = self.patch_size * self.patch_size * 3
        N = self.hidden
        return self._gemm_latency_us(M, N, K)

    def _qkv_proj_us(self, num_images: int) -> float:
        """QKV linear projection (fused as one GEMM in practice)."""
        seq_len = self.num_patches + 1
        M = num_images * seq_len
        K = self.hidden
        N = 3 * self.hidden  # fused QKV
        return self._gemm_latency_us(M, N, K)

    def _attention_us(self, num_images: int) -> float:
        """Self-attention (FlashAttention-2 for encoder)."""
        seq_len = self.num_patches + 1
        return self._flash_attention_latency_us(num_images, self.heads, seq_len)

    def _o_proj_us(self, num_images: int) -> float:
        """Output projection after attention."""
        seq_len = self.num_patches + 1
        M = num_images * seq_len
        K = self.hidden
        N = self.hidden
        return self._gemm_latency_us(M, N, K)

    def _mlp_us(self, num_images: int) -> float:
        """MLP block: up_proj + GELU + down_proj (2 GEMMs + 1 activation)."""
        seq_len = self.num_patches + 1
        M = num_images * seq_len
        # Up projection
        up = self._gemm_latency_us(M, self.ffn_dim, self.hidden)
        # GELU activation (elementwise, memory-bound)
        gelu = self._elementwise_latency_us(num_images * seq_len * self.ffn_dim)
        # Down projection
        down = self._gemm_latency_us(M, self.hidden, self.ffn_dim)
        return up + gelu + down

    def _layernorm_us(self, num_images: int) -> float:
        """LayerNorm: read input + write output + stats computation."""
        seq_len = self.num_patches + 1
        # Reads input once for mean, once for variance, writes output
        num_elements = num_images * seq_len * self.hidden
        return self._elementwise_latency_us(num_elements, reads=2, writes=1)

    def _projection_us(self, num_images: int, llm_hidden: int) -> float:
        """Final projection from ViT hidden to LLM embedding space."""
        M = num_images * self.num_patches
        K = self.hidden
        N = llm_hidden
        return self._gemm_latency_us(M, N, K)

    # Variable-resolution helpers (accept explicit seq_len / num_patches)

    def _patch_embed_us_v(self, num_images: int, num_patches: int) -> float:
        M = num_images * num_patches
        K = self.patch_size * self.patch_size * 3
        N = self.hidden
        return self._gemm_latency_us(M, N, K)

    def _qkv_proj_us_v(self, num_images: int, seq_len: int) -> float:
        M = num_images * seq_len
        return self._gemm_latency_us(M, 3 * self.hidden, self.hidden)

    def _o_proj_us_v(self, num_images: int, seq_len: int) -> float:
        M = num_images * seq_len
        return self._gemm_latency_us(M, self.hidden, self.hidden)

    def _mlp_us_v(self, num_images: int, seq_len: int) -> float:
        M = num_images * seq_len
        up = self._gemm_latency_us(M, self.ffn_dim, self.hidden)
        gelu = self._elementwise_latency_us(num_images * seq_len * self.ffn_dim)
        down = self._gemm_latency_us(M, self.hidden, self.ffn_dim)
        return up + gelu + down

    def _layernorm_us_v(self, num_images: int, seq_len: int) -> float:
        num_elements = num_images * seq_len * self.hidden
        return self._elementwise_latency_us(num_elements, reads=2, writes=1)

    # -----------------------------------------------------------------------
    # Total encoder latency
    # -----------------------------------------------------------------------

    def estimate_total_latency_us(self, num_images: int, llm_hidden: int = 4096,
                                   resolution: int = None) -> float:
        """Estimate total encoder latency in microseconds.

        Args:
            num_images: Number of images in the batch.
            llm_hidden: LLM hidden size for the final projection layer.
            resolution: Override image resolution (pixels). If None, uses the
                model's default image_size.

        Returns:
            Total latency in microseconds.
        """
        if num_images == 0:
            return 0.0

        # Allow per-call resolution override
        if resolution is not None:
            num_patches = (resolution // self.patch_size) ** 2
        else:
            num_patches = self.num_patches

        total_us = 0.0

        # Patch embedding
        total_us += self._patch_embed_us_v(num_images, num_patches)

        # Transformer layers (each: LN → Attn → residual → LN → MLP → residual)
        seq_len = num_patches + 1
        for _ in range(self.num_layers):
            total_us += self._layernorm_us_v(num_images, seq_len)
            total_us += self._qkv_proj_us_v(num_images, seq_len)
            total_us += self._flash_attention_latency_us(num_images, self.heads, seq_len)
            total_us += self._o_proj_us_v(num_images, seq_len)
            total_us += self._elementwise_latency_us(num_images * seq_len * self.hidden)
            total_us += self._layernorm_us_v(num_images, seq_len)
            total_us += self._mlp_us_v(num_images, seq_len)
            total_us += self._elementwise_latency_us(num_images * seq_len * self.hidden)

        # Final LayerNorm
        total_us += self._layernorm_us_v(num_images, seq_len)

        # Projection to LLM hidden space
        M = num_images * num_patches
        total_us += self._gemm_latency_us(M, llm_hidden, self.hidden)

        return total_us * self.calibration_factor

    def estimate_total_latency_ns(self, num_images: int, llm_hidden: int = 4096,
                                   resolution: int = None) -> int:
        """Estimate total encoder latency in nanoseconds."""
        return int(self.estimate_total_latency_us(num_images, llm_hidden, resolution) * 1000)

    def estimate_per_layer_latency_us(self, num_images: int, llm_hidden: int = 4096) -> list:
        """Return per-layer breakdown for trace generation.

        Returns list of (layer_name, latency_us) tuples.
        """
        if num_images == 0:
            return []

        layers = []

        # Patch embedding
        layers.append(("patch_embed", self._patch_embed_us(num_images)))

        # Transformer layers
        for i in range(self.num_layers):
            layers.append((f"encoder_layernorm1_{i}", self._layernorm_us(num_images)))
            attn_us = (self._qkv_proj_us(num_images) +
                       self._attention_us(num_images) +
                       self._o_proj_us(num_images))
            layers.append((f"encoder_attention_{i}", attn_us))
            layers.append((f"encoder_layernorm2_{i}", self._layernorm_us(num_images)))
            layers.append((f"encoder_mlp_{i}", self._mlp_us(num_images)))

        # Final layernorm
        layers.append(("encoder_final_layernorm", self._layernorm_us(num_images)))

        # Projection
        layers.append(("encoder_projection", self._projection_us(num_images, llm_hidden)))

        # Apply calibration to each layer proportionally
        if self.calibration_factor != 1.0:
            layers = [(name, lat * self.calibration_factor) for name, lat in layers]

        return layers

    def get_encoder_output_size_bytes(self, num_images: int, fp: int = 2) -> int:
        """Size of encoder output to transfer to prefill instance.

        encoder output = num_images * num_patches * llm_hidden * fp
        But since projection maps to llm_hidden, we use that.
        """
        return num_images * self.num_patches * self.hidden * fp

    def get_encoder_weight_bytes(self, fp: int = 2) -> int:
        """Total encoder model weight size in bytes (approximately).

        Includes: patch_embed + num_layers * (attention + MLP) + final_ln + projection.
        """
        # Patch embedding: patch_size^2 * 3 * hidden
        patch_embed = self.patch_size ** 2 * 3 * self.hidden

        # Per layer: QKV (3*hidden*hidden) + O (hidden*hidden) + MLP (hidden*ffn + ffn*hidden) + 2*LN (2*hidden)
        per_layer = (3 * self.hidden * self.hidden +
                     self.hidden * self.hidden +
                     self.hidden * self.ffn_dim +
                     self.ffn_dim * self.hidden +
                     2 * self.hidden)

        # Final LN + projection (hidden * hidden)
        final = self.hidden + self.hidden * self.hidden

        total_params = patch_embed + self.num_layers * per_layer + final
        return total_params * fp
