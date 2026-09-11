# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for unified_attention_sparse_mla: MLA attention where each
query token attends to a per-token top-k selection of KV-cache entries (the
DeepSeek-style sparse attention path) instead of its whole context.

Shape conventions follow the op's docstring and its correctness test: the KV
cache is a single latent head of ``kv_lora_rank + rope_rank`` (512 + 64 = 576),
the output keeps only the ``kv_lora_rank`` half, and ``topk_indices`` addresses
the *flattened* KV cache (block_table[i, abs // block_size] * block_size +
abs % block_size), not the block table.

Work scales with top_k, not with the context length: s_k only sets how far the
selected entries are spread across the cache. Both regimes the kernel compiles
separately are covered by the default sweep -- decode (s_q == 1, which turns on
the ALL_DECODE specialization) and chunked prefill (s_q > 1).

Correctness is not re-checked here; see
op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py.

Usage examples
--------------
# Sweep default shapes, report time / TFLOPS / BW
python bench_unified_attention_sparse_mla.py

# Single shape: batch=32 s_q=1 s_k=8192 h_q=16 top_k=2048
python bench_unified_attention_sparse_mla.py --shape 32 1 8192 16 2048

# KV cache block size (top-k tiling granularity follows it)
python bench_unified_attention_sparse_mla.py --block-size 16

# Save CSV
python bench_unified_attention_sparse_mla.py -o
"""

import argparse
import os
import sys

# Skip CK/HIP native .so loading -- Triton kernels only.
os.environ.setdefault("AITER_TRITON_ONLY", "1")

# Ensure repo root is on the path when running this script directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# (batch, s_q, s_k, h_q, top_k)
# h_q = 16 is the per-rank head count of a 128-head MLA model at TP8, so it is
# the shape a single GPU actually runs; h_q = 128 covers TP1. The decode rows
# sweep batch (the only parallelism decode has), then top_k at a fixed batch,
# and the last two rows are chunked prefill.
DEFAULT_SHAPES = [
    (1, 1, 8192, 16, 2048),
    (8, 1, 8192, 16, 2048),
    (32, 1, 8192, 16, 2048),
    (128, 1, 8192, 16, 2048),
    (1, 1, 8192, 128, 2048),
    (32, 1, 8192, 128, 2048),
    (32, 1, 8192, 16, 512),
    (32, 1, 8192, 16, 1024),
    (1, 1024, 8192, 16, 2048),
    (1, 4096, 16384, 16, 2048),
]

KV_LORA_RANK = 512
ROPE_RANK = 64
HEAD_DIM = KV_LORA_RANK + ROPE_RANK  # q/k head dim
H_KV = 1  # the latent KV head
# The wrapper hardcodes BLOCK_M, and the grid is one program per
# (query token, BLOCK_M heads), so h_q must be a whole number of head blocks.
BLOCK_M = 16
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _make_inputs(batch, s_q, s_k, h_q, top_k, block_size, seed=0):
    """Build one unified_attention_sparse_mla call's worth of random inputs."""
    torch.manual_seed(seed)
    total_q = batch * s_q
    blocks_per_seq = triton.cdiv(s_k, block_size)
    num_blocks = batch * blocks_per_seq

    # Shuffled block table: a sequence's blocks are scattered across the cache,
    # as they are once the allocator has been running for a while.
    block_table = torch.randperm(num_blocks, device=DEVICE, dtype=torch.int32).view(
        batch, blocks_per_seq
    )

    kv = torch.randn(
        (num_blocks, block_size, H_KV, HEAD_DIM), device=DEVICE, dtype=DTYPE
    )
    q = torch.randn((total_q, h_q, HEAD_DIM), device=DEVICE, dtype=DTYPE)
    out = torch.empty((total_q, h_q, KV_LORA_RANK), device=DEVICE, dtype=DTYPE)

    seqused_k = torch.full((batch,), s_k, device=DEVICE, dtype=torch.int32)
    cu_seqlens_q = torch.arange(
        0, (batch + 1) * s_q, s_q, device=DEVICE, dtype=torch.int32
    )

    # top-k picks distinct positions per query token, which then get translated
    # into flat cache slots through the block table. Short of s_k entries the
    # tail is -1, the padding value the kernel masks off.
    keep = min(top_k, s_k)
    abs_idx = torch.argsort(torch.rand((total_q, s_k), device=DEVICE), dim=-1)[
        :, :keep
    ].to(torch.int32)
    seq_of_token = torch.arange(total_q, device=DEVICE) // s_q
    flat_idx = (
        block_table[seq_of_token.unsqueeze(1), abs_idx // block_size].to(torch.int64)
        * block_size
        + abs_idx % block_size
    ).to(torch.int32)
    if keep < top_k:
        pad = torch.full((total_q, top_k - keep), -1, device=DEVICE, dtype=torch.int32)
        flat_idx = torch.cat([flat_idx, pad], dim=1)
    # Selected entries arrive in no particular order.
    flat_idx = flat_idx.gather(
        1, torch.argsort(torch.rand((total_q, top_k), device=DEVICE), dim=-1)
    )

    return {
        "q": q,
        "kv": kv,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": s_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": s_k,
        "softmax_scale": KV_LORA_RANK**-0.5,
        "topk_indices": flat_idx,
        "block_table": block_table,
        "kv_lora_rank": KV_LORA_RANK,
    }


def _flops(total_q, h_q, top_k):
    # Per query token and head: QK over the top_k selected entries, then PV.
    return 2 * total_q * h_q * top_k * (HEAD_DIM + KV_LORA_RANK)


def _bytes(total_q, h_q, top_k):
    """Traffic the kernel issues, not the compulsory minimum.

    Each program serves BLOCK_M heads of one token, so the same token's top_k
    entries are fetched once per head block -- that re-read is what the runtime
    actually pays for (L2 absorbs part of it), and counting the cache entries
    only once would report a bandwidth the kernel never has to sustain.
    """
    elem = torch.tensor([], dtype=DTYPE).element_size()
    head_blocks = max(1, h_q // BLOCK_M)
    kv_read = total_q * head_blocks * top_k * HEAD_DIM * elem
    q_read = total_q * h_q * HEAD_DIM * elem
    o_write = total_q * h_q * KV_LORA_RANK * elem
    return kv_read + q_read + o_write


def run_benchmark(args):
    if args.shape is not None:
        x_vals_list = [tuple(args.shape)]
    else:
        x_vals_list = DEFAULT_SHAPES

    header = (
        f"{'B':>4} {'s_q':>6} {'s_k':>6} {'h_q':>4} {'top_k':>6}  "
        f"{'Time_us':>10}  {'TFLOPS':>8}  {'BW(GB/s)':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for batch, s_q, s_k, h_q, top_k in x_vals_list:
        if h_q % BLOCK_M:
            raise SystemExit(
                f"h_q must be a multiple of BLOCK_M={BLOCK_M}, got h_q={h_q}"
            )
        inputs = _make_inputs(batch, s_q, s_k, h_q, top_k, args.block_size)

        def fn(inputs=inputs):
            return unified_attention_sparse_mla(**inputs)

        ms = triton.testing.do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms)
        total_q = batch * s_q
        us = ms * 1e3
        tflops = _flops(total_q, h_q, top_k) / ms * 1e-9
        bw = _bytes(total_q, h_q, top_k) / (ms * 1e-3) * 1e-9

        print(
            f"{batch:>4} {s_q:>6} {s_k:>6} {h_q:>4} {top_k:>6}  "
            f"{us:>10.2f}  {tflops:>8.2f}  {bw:>10.1f}"
        )
        rows.append((batch, s_q, s_k, h_q, top_k, us, tflops, bw))

    if args.o:
        import csv

        fname = f"{get_caller_name_no_ext()}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["B", "s_q", "s_k", "h_q", "top_k", "Time_us", "TFLOPS", "BW_GBs"]
            )
            w.writerows(rows)
        print(f"\nSaved to {fname}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark unified_attention_sparse_mla",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        metavar=("BATCH", "S_Q", "S_K", "H_Q", "TOP_K"),
        help="Single shape to benchmark instead of the default sweep.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="KV cache block size; also the kernel's top-k tiling granularity.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write results to a CSV file in the current directory.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=300.0,
        help="Warmup budget in ms (time-based, ensures JIT/autotune completes).",
    )
    parser.add_argument(
        "--rep-ms",
        type=float,
        default=500.0,
        help="Measurement budget in ms passed to triton.testing.do_bench.",
    )
    return parser.parse_args(args=args)


def main(args=None):
    run_benchmark(parse_args(args=args))


if __name__ == "__main__":
    main()
