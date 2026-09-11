# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the Gluon MLA *decode* kernel (gfx950 / CDNA4).

`mla_gluon` is a unified entry serving both MLA decode and DeepSeek-V4 sparse
prefill. Only the prefill side (`has_pe=False`) was covered by
`bench_sparse_attention_dsv4.py`; this benchmark covers the decode side
(`has_pe=True`, stage-1 + stage-2 reduce, or the stage-1-only fast path when
NUM_KV_SPLITS==1), which previously had correctness coverage in
`op_tests/test_mla.py` but no performance coverage.

The wrapper dispatches on (nhead, KV dtype) into three regimes, each with its
own launch geometry, so the default shape list exercises all three:

  bh16bn64  : nhead <= 16, bf16 KV        (BLOCK_N=64)
  bh16bn128 : nhead <= 16, fp8_e4m3 KV    (BLOCK_N=128, requires batch_size==1)
  bh64      : nhead in (64, 128), bf16 KV (BLOCK_H=64, requires batch % 64 == 0)

plus `bh16bn64_lse`, which is bh16bn64 with the merged fp32 lse also returned.
The shape list covers the reference configs published in the gluon README, and
each shape reports both TFLOPS and GB/s so either README unit can be compared.

Usage:
  python op_tests/op_benchmarks/triton/bench_mla_gluon_decode.py
  python op_tests/op_benchmarks/triton/bench_mla_gluon_decode.py --regime bh64
  python op_tests/op_benchmarks/triton/bench_mla_gluon_decode.py --metric bandwidth
  python op_tests/op_benchmarks/triton/bench_mla_gluon_decode.py -check -o
"""

import torch
import triton

from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)

try:
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon

    HAS_GLUON = get_gfx() == "gfx950"
except ImportError:
    mla_gluon = None
    HAS_GLUON = False


KV_LORA_RANK = 512  # head_dim_ckv, the NoPE / V dim
QK_ROPE_HEAD_DIM = 64  # head_dim_kpe
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576, shared kv_c row

# Physical KV pool, in PAGE_SIZE=1 pages. Same `kv_max_sz = 65536 * 32` budget
# test_mla.py uses: what a serving framework has left for the cache after the
# weights are loaded.
KV_POOL_PAGES = 65536 * 32

# (batch, nhead, ctx_len) per regime. Constraints come from the mla_gluon
# wrapper asserts; bh64 additionally needs min_kv_seq_len to exceed
# NUM_KV_SPLITS * (3 * BLOCK_N + NUM_KV_SPLITS), satisfied by ctx >= 1024.
# Shapes tagged `README ref` mirror the reference configs in
# `aiter/ops/triton/gluon/README.md`, so this benchmark can be compared directly
# against the numbers published there (bh64 in TFLOPS, bh16* in TB/s):
#   bh64            test_mla.py -c 16384 -b 64 128 -n 64,1 128,1   ~563 TFLOPS
#   bh16bn128       test_mla.py -c 10000000 -b 1 -n 16,1 -kvd fp8  ~4.58 TB/s
#   bh16bn64        test_mla.py -c 10000000 -b 1 -n 16,1           ~5.33 TB/s
#   bh16bn64_lse    test_mla.py -c 100000  -b 4 -n 16,1 -lse       ~4.31 TB/s
REGIME_SHAPES = {
    "bh16bn64": [
        (1, 16, 1024),
        (1, 16, 8192),
        (16, 16, 8192),
        (64, 16, 8192),
        (128, 16, 8192),
        (128, 16, 16384),
        (1, 16, 10000000),  # README ref
    ],
    "bh16bn128": [
        (1, 16, 1024),
        (1, 16, 8192),
        (1, 16, 16384),
        (1, 16, 10000000),  # README ref
    ],
    "bh64": [
        (64, 64, 8192),
        (64, 64, 16384),  # README ref
        (128, 64, 8192),
        (64, 128, 8192),
        (256, 64, 8192),
        (128, 64, 16384),  # README ref
        (64, 128, 16384),  # README ref
        (128, 128, 16384),  # README ref
    ],
    # Same kernel/layout as bh16bn64, but with the merged fp32 lse also written
    # out; kept as its own regime so the extra output shows up in the report.
    "bh16bn64_lse": [
        (4, 16, 100000),  # README ref
    ],
}

REGIME_KV_DTYPE = {
    "bh16bn64": torch.bfloat16,
    "bh16bn128": torch.float8_e4m3fn,
    "bh64": torch.bfloat16,
    "bh16bn64_lse": torch.bfloat16,
}

REGIME_RETURN_LSE = {"bh16bn64_lse"}


def _build_inputs(batch, nhead, ctx, kv_dtype, device):
    """Uniform-length decode inputs in the shared-KV / 2-D block-table layout.

    The physical page pool is fixed at KV_POOL_PAGES and page ids are drawn with
    replacement, exactly as `op_tests/test_mla.py` does. Sizing the pool to the
    logical context instead would make long-context shapes stream a cache no
    serving framework could allocate, and the numbers would no longer line up
    with the README reference values that test_mla.py produced.
    """
    seq_lens_kv = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    total_kv = batch * ctx
    num_page = KV_POOL_PAGES
    kv_indices = torch.randint(
        0, num_page, (total_kv,), dtype=torch.int32, device=device
    )
    page_table = kv_indices.view(batch, ctx)

    kv_c = torch.randn((num_page, QK_HEAD_DIM), dtype=torch.bfloat16, device=device)
    if kv_dtype != torch.bfloat16:
        kv_c = kv_c.to(kv_dtype)

    q = torch.randn((batch, nhead, QK_HEAD_DIM), dtype=torch.bfloat16, device=device)
    q_nope = q[:, :, :KV_LORA_RANK]
    q_pe = q[:, :, KV_LORA_RANK:]
    o = torch.empty((batch, nhead, KV_LORA_RANK), dtype=torch.bfloat16, device=device)
    return q_nope, q_pe, kv_c, o, page_table, seq_lens_kv


def _launch(q_nope, q_pe, kv_c, o, page_table, seq_lens_kv, sm_scale, ctx, lse=False):
    return mla_gluon(
        q_nope,
        q_pe,
        kv_c,
        o,
        page_table,
        seq_lens_kv,
        sm_scale,
        use_2d_view=True,
        min_kv_seq_len=ctx,
        return_lse=lse,
    )


def _ref_decode(q_nope, q_pe, kv_c, page_table, seq_lens_kv, sm_scale):
    """Absorbed-MLA decode reference: V is the NoPE prefix of the KV row."""
    batch, nhead, _ = q_nope.shape
    q = torch.cat([q_nope.float(), q_pe.float()], dim=-1)
    out = torch.empty((batch, nhead, KV_LORA_RANK), dtype=torch.float32)
    for b in range(batch):
        s = int(seq_lens_kv[b])
        k = kv_c[page_table[b, :s].long()].float()
        p = torch.softmax((q[b] @ k.t()) * sm_scale, dim=-1)
        out[b] = p @ k[:, :KV_LORA_RANK]
    return out


def check_correctness(regimes, device):
    """Small-shape torch-reference gate so a broken kernel fails loudly."""
    print("\n========== CORRECTNESS ==========")
    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)
    for regime in regimes:
        # bh64 needs batch % 64 == 0 and nhead in (64, 128); keep ctx small.
        batch, nhead, ctx = (64, 64, 1024) if regime == "bh64" else (1, 16, 256)
        kv_dtype = REGIME_KV_DTYPE[regime]
        lse = regime in REGIME_RETURN_LSE
        torch.manual_seed(0)
        q_nope, q_pe, kv_c, o, page_table, seq_lens_kv = _build_inputs(
            batch, nhead, ctx, kv_dtype, device
        )
        _launch(q_nope, q_pe, kv_c, o, page_table, seq_lens_kv, sm_scale, ctx, lse)
        torch.cuda.synchronize()
        ref = _ref_decode(q_nope, q_pe, kv_c, page_table, seq_lens_kv, sm_scale)
        # fp8 KV quantizes the gathered K, so allow a looser tolerance there.
        tol = 1e-1 if kv_dtype == torch.float8_e4m3fn else 2e-2
        max_diff = (o.float() - ref).abs().max().item()
        torch.testing.assert_close(o.float(), ref, atol=tol, rtol=tol)
        print(
            f"  {regime:10s} B={batch:<4d} H={nhead:<4d} ctx={ctx:<6d}: "
            f"OK (max|delta|={max_diff:.4f})"
        )


def benchmark(args):
    # Single-token labels: spaces break table parsing, and report tooling shows
    # only the text ahead of a "(", which would hide the unit.
    unit_by_metric = {
        "throughput": "TFLOPS",
        "time": "Time_ms",
        "bandwidth": "GB/s",
    }
    # The README quotes bh64 in TFLOPS and bh16* in TB/s, so the default run
    # reports both columns and neither regime needs a second invocation.
    metrics = ["throughput", "bandwidth"] if args.metric == "both" else [args.metric]

    # Numeric columns first: downstream table parsers require the leading cell
    # of a data row to be a number, so the string `regime` column goes last.
    x_vals_list = [
        (batch, nhead, ctx, regime)
        for regime in args.regimes
        for (batch, nhead, ctx) in REGIME_SHAPES[regime]
    ]

    config = triton.testing.Benchmark(
        x_names=["B", "H", "ctx", "regime"],
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=metrics,
        line_names=[unit_by_metric[m] for m in metrics],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")][: len(metrics)],
        ylabel="Performance",
        plot_name=get_caller_name_no_ext(),
        args={"device": args.device},
    )

    @triton.testing.perf_report([config])
    def bench_mla_gluon(B, H, ctx, regime, metric, device):
        kv_dtype = REGIME_KV_DTYPE[regime]
        lse = regime in REGIME_RETURN_LSE
        torch.manual_seed(0)
        q_nope, q_pe, kv_c, o, page_table, seq_lens_kv = _build_inputs(
            B, H, ctx, kv_dtype, device
        )
        sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

        ms = triton.testing.do_bench(
            lambda: _launch(
                q_nope, q_pe, kv_c, o, page_table, seq_lens_kv, sm_scale, ctx, lse
            ),
            warmup=25,
            rep=100,
        )

        if metric == "time":
            return ms
        # QK over the full 576-wide row plus PV over the 512-wide NoPE prefix.
        flops = 2.0 * B * H * ctx * (QK_HEAD_DIM + KV_LORA_RANK)
        if metric == "throughput":
            return flops / ms * 1e-9
        if metric == "bandwidth":
            mem = (
                B * ctx * QK_HEAD_DIM * kv_c.element_size()
                + q_nope.numel() * q_nope.element_size()
                + q_pe.numel() * q_pe.element_size()
                + o.numel() * o.element_size()
            )
            return mem / (ms * 1e-3) * 1e-9
        raise NotImplementedError(f"{metric} is not supported")

    bench_mla_gluon.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args(args: list[str] | None = None):
    parser = get_parser("Gluon MLA decode")
    # "both" is not a --metric choice; argparse only validates choices for values
    # that come off the command line, so this default just means "TFLOPS + GB/s".
    parser.set_defaults(metric="both")
    parser.add_argument(
        "--regime",
        type=str,
        choices=["all", *REGIME_SHAPES],
        default="all",
        help="Which mla_gluon dispatch regime to benchmark.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("B", "H", "CTX"),
        help="User-defined (batch, nhead, ctx_len); regime is inferred from --regime.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "-check",
        action="store_true",
        help="Run a small-shape torch-reference correctness gate before benchmarking.",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parsed = parser.parse_args(args=args)

    if parsed.regime == "all":
        parsed.regimes = list(REGIME_SHAPES)
    else:
        parsed.regimes = [parsed.regime]

    if parsed.shape:
        if parsed.regime == "all":
            parser.error("--shape requires an explicit --regime")
        REGIME_SHAPES[parsed.regime] = [tuple(parsed.shape)]
    return parsed


def run_bench(args):
    torch.set_default_device(args.device)
    if args.check:
        check_correctness(args.regimes, args.device)
    benchmark(args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if not HAS_GLUON:
        print(
            "mla_gluon is unavailable (requires gfx950 / CDNA4 and Triton >= 3.6); "
            "skipping Gluon MLA decode benchmark."
        )
        return
    if parsed_args.print_vgpr:
        print_vgpr(lambda: run_bench(parsed_args), table_start=get_caller_name_no_ext())
        return
    run_bench(parsed_args)


if __name__ == "__main__":
    main()
