# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for gather_kv_b_proj: gather compressed-KV rows out of a
paged/DCP cache, dequantize, apply kv_b_proj, and split into k_nope/v -- one
kernel for the whole MLA "decompress cached KV" chain.

Default sweep matches the Kimi-K3 TP8 shape family exercised by
`aiter::unified_attention_with_output_base` when DCP has a cached prefix:
block_size=1 (the "flat" token-granular cache -> `_triton_gather_kv_b_proj_flat`),
tp_k_head_num=12 (=96 heads / TP8), qk_nope_head_dim=v_head_dim=128, fp8 KV
cache + fp8 kv_b_proj weight with per-128x128-block scale, weight preshuffled.
See k3_op_param_reference.md's `_triton_gather_kv_b_proj_flat` section for the
op/kernel-parameter derivation this sweep is based on.

Usage examples
--------------
# Sweep default shapes (K3 TP8 flat-path family), report time / TFLOPS / BW
python bench_gather_kv_b_proj.py

# Single shape: batch=1 block_size=1 tp_k_head_num=12 avg_kv_length=8192
python bench_gather_kv_b_proj.py --shape 1 1 12 8192

# Continuous-batching (non-flat) path with block_size > 1
python bench_gather_kv_b_proj.py --shape 8 16 12 8192

# bf16 KV cache / weight instead of fp8
python bench_gather_kv_b_proj.py --ktype bf16 --weight-dtype bf16

# Per-row (instead of per-128x128-block) weight scale
python bench_gather_kv_b_proj.py --scale-mode per_row

# Save CSV
python bench_gather_kv_b_proj.py -o
"""

import argparse
import os
import sys

# NOTE: unlike most triton-only benches, this one imports `aiter.ops.shuffle`
# (for `shuffle_weight`, the same helper used to preshuffle kv_b_proj weights
# in production), which pulls in the non-triton `aiter.jit` machinery. Don't
# set AITER_TRITON_ONLY here.

# Ensure repo root is on the path when running this script directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj
from aiter.utility import dtypes
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# (batch_size, block_size, tp_k_head_num, avg_kv_length)
# block_size=1 rows are the K3 DCP "flat" path (dispatches to
# `_triton_gather_kv_b_proj_flat`); avg_kv_length sweeps the traced prefill
# chunk contexts (544/3072/8192/16384/20000) from k3_op_param_reference.md.
# The larger-batch rows exercise the decode-style continuous-batching shape
# that the same wrapper serves outside of the single-prefill-chunk trace.
DEFAULT_SHAPES = [
    (1, 1, 12, 544),
    (1, 1, 12, 3072),
    (1, 1, 12, 8192),
    (1, 1, 12, 16384),
    (1, 1, 12, 20000),
    (8, 1, 12, 8192),
    (16, 1, 12, 8192),
    (32, 1, 12, 2048),
]

KV_C_DIM = 512
KV_PE_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
DEVICE = "cuda"

arg_to_torch_dtype = {
    "fp8": dtypes.fp8,
    "bf16": torch.bfloat16,
}


def _elem_bytes(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def _make_inputs(
    batch_size,
    block_size,
    tp_k_head_num,
    avg_kv_length,
    k_dtype,
    weight_dtype,
    scale_mode,
    weight_preshuffle,
    seed=0,
):
    """Build one gather_kv_b_proj call's worth of random inputs."""
    torch.manual_seed(seed)
    num_block = max(1, 2 * avg_kv_length // block_size)
    weight_n = tp_k_head_num * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)

    k_buffer = torch.randn(
        (num_block, block_size, KV_C_DIM + KV_PE_DIM),
        device=DEVICE,
        dtype=torch.float32,
    ).to(k_dtype)
    k_scale = torch.randn(1, device=DEVICE, dtype=torch.float32).abs()

    var_ratio = 0.2
    context_lens = torch.randint(
        int((1 - var_ratio) * avg_kv_length),
        int((1 + var_ratio) * avg_kv_length) + 1,
        (batch_size,),
    ).to(device=DEVICE, dtype=torch.int32)
    context_blocks = torch.div(
        context_lens + block_size - 1, block_size, rounding_mode="trunc"
    )

    kv_indptr = torch.zeros((batch_size + 1,), device=DEVICE, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(context_blocks, dim=0)
    kv_prefix_sum_context_lens = torch.zeros(
        (batch_size + 1,), device=DEVICE, dtype=torch.int32
    )
    kv_prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

    kv_indices = torch.zeros(
        int(kv_indptr[-1].item()), device=DEVICE, dtype=torch.int32
    )
    for b in range(batch_size):
        ctx_blk = int(context_blocks[b].item())
        kv_indices[kv_indptr[b] : kv_indptr[b + 1]] = torch.randperm(
            num_block, device=DEVICE
        )[:ctx_blk]

    kv_proj_weight = torch.randn(
        (weight_n, KV_C_DIM), device=DEVICE, dtype=torch.float32
    ).to(weight_dtype)
    if weight_dtype == torch.bfloat16:
        # Unquantized weight: all-ones scale so the matmul result is unscaled.
        kv_proj_scale = (
            torch.ones((weight_n, 1), device=DEVICE, dtype=torch.float32)
            if scale_mode == "per_row"
            else torch.ones(
                (weight_n // 128, KV_C_DIM // 128), device=DEVICE, dtype=torch.float32
            )
        )
    elif scale_mode == "per_row":
        kv_proj_scale = torch.randn(
            (weight_n, 1), device=DEVICE, dtype=torch.float32
        ).abs()
    else:
        kv_proj_scale = torch.randn(
            (weight_n // 128, KV_C_DIM // 128), device=DEVICE, dtype=torch.float32
        ).abs()

    if weight_preshuffle:
        kv_proj_weight = shuffle_weight(kv_proj_weight)

    total_kv = int(kv_prefix_sum_context_lens[-1].item())
    k_prefix = torch.zeros(
        (total_kv, tp_k_head_num, QK_NOPE_HEAD_DIM + KV_PE_DIM),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    v_prefix = torch.zeros(
        (total_kv, tp_k_head_num, V_HEAD_DIM), device=DEVICE, dtype=torch.bfloat16
    )

    inputs = {
        "k_buffer": k_buffer,
        "k_scale": k_scale,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_prefix_sum_context_lens": kv_prefix_sum_context_lens,
        "kv_proj_weight": kv_proj_weight,
        "kv_proj_scale": kv_proj_scale,
        "k_prefix": k_prefix,
        "v_prefix": v_prefix,
        "weight_preshuffle": weight_preshuffle,
    }
    return inputs, total_kv, weight_n


def _flops(total_kv, weight_n):
    # kv_c [total_kv, KV_C_DIM] @ kv_proj_weight.T [KV_C_DIM, weight_n]
    return 2 * total_kv * weight_n * KV_C_DIM


def _bytes(total_kv, weight_n, k_elem_bytes, weight_elem_bytes):
    read = total_kv * (KV_C_DIM + KV_PE_DIM) * k_elem_bytes  # gathered compressed KV
    read += weight_n * KV_C_DIM * weight_elem_bytes  # kv_b_proj weight
    write = total_kv * weight_n * 2  # k_prefix + v_prefix, bf16
    return read + write


def run_benchmark(args):
    k_dtype = arg_to_torch_dtype[args.ktype]
    weight_dtype = arg_to_torch_dtype[args.weight_dtype]
    weight_preshuffle = not args.no_preshuffle

    if args.shape is not None:
        batch_size, block_size, tp_k_head_num, avg_kv_length = args.shape
        x_vals_list = [(batch_size, block_size, tp_k_head_num, avg_kv_length)]
    else:
        x_vals_list = DEFAULT_SHAPES

    header = (
        f"{'B':>4} {'blk':>4} {'tpH':>4} {'kv_len':>7}  {'Time(us)':>10}  "
        f"{'TFLOPS':>8}  {'BW(GB/s)':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for batch_size, block_size, tp_k_head_num, avg_kv_length in x_vals_list:
        inputs, total_kv, weight_n = _make_inputs(
            batch_size,
            block_size,
            tp_k_head_num,
            avg_kv_length,
            k_dtype,
            weight_dtype,
            args.scale_mode,
            weight_preshuffle,
        )

        def fn(inputs=inputs):
            return gather_kv_b_proj(**inputs)

        ms = triton.testing.do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms)
        us = ms * 1e3
        tflops = _flops(total_kv, weight_n) / ms * 1e-9
        bw = (
            _bytes(total_kv, weight_n, _elem_bytes(k_dtype), _elem_bytes(weight_dtype))
            / (ms * 1e-3)
            * 1e-9
        )

        print(
            f"{batch_size:>4} {block_size:>4} {tp_k_head_num:>4} {avg_kv_length:>7}  "
            f"{us:>10.2f}  {tflops:>8.2f}  {bw:>10.1f}"
        )
        rows.append(
            (batch_size, block_size, tp_k_head_num, avg_kv_length, us, tflops, bw)
        )

    if args.o:
        import csv

        fname = f"{get_caller_name_no_ext()}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "B",
                    "block_size",
                    "tp_k_head_num",
                    "avg_kv_length",
                    "Time_us",
                    "TFLOPS",
                    "BW_GBs",
                ]
            )
            w.writerows(rows)
        print(f"\nSaved to {fname}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark gather_kv_b_proj",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("BATCH", "BLOCK_SIZE", "TP_K_HEAD_NUM", "AVG_KV_LENGTH"),
        help="Single shape to benchmark instead of the default sweep.",
    )
    parser.add_argument(
        "--ktype",
        type=str,
        default="fp8",
        choices=list(arg_to_torch_dtype),
        help="KV cache dtype.",
    )
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default="fp8",
        choices=list(arg_to_torch_dtype),
        help="kv_b_proj weight dtype.",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        default="block",
        choices=["block", "per_row"],
        help="Weight scale granularity: per-128x128-block (K3 online ptpc_fp8) or per-row.",
    )
    parser.add_argument(
        "--no-preshuffle",
        action="store_true",
        help="Disable kv_b_proj weight preshuffle (K3 production runs preshuffled).",
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
