"""
Benchmark the GEMM shapes that GLM-5, DeepSeek-V4, Kimi-K3 and Qwen3.5 actually run
through Triton kernels on gfx950.

Why this file exists
--------------------
bench_gemm_a16w16.py and bench_gemm_a8w8_blockscale.py already cover these two ops, but
their default sweep is a synthetic family -- (N,K)=(1280,8192) with an M ladder, plus two
(4096,4096,*) corners. No production model runs any of those. So the ops were monitored
while every shape the models actually execute was not.

The shapes below come from aiter/configs/model_configs/*_tuned_gemm.csv, filtered to
gfx == gfx950 and libtype == triton. That libtype filter is the important part: those CSVs
also contain ck / cktile / asm / flydsl / opus / torch rows, and only the `triton` rows are
shapes where the tuner picked a Triton kernel, i.e. where a Triton compiler regression
actually reaches production. Everything else in those files is a real model shape served by
some other backend, and is out of scope here.

Deliberately NOT included
-------------------------
- MoE (glm5_fp4/mxfp8/ptpc, dsv4_fp8fp4, kimik3_a4w4/a8w4/a16w4, qwen3_5 fp4): every
  gfx950 row dispatches to FlyDSL, CK 2-stage or a prebuilt HIP fmoe symbol -- never a
  Triton kernel. See aiter/fused_moe.py.
- A8W8 bpreshuffle GEMM (kimik3, glm5.2, qwen3.5-mxfp4): ck / cktile / flydsl only.
- kimik3 A4W4 blockscale GEMM: prebuilt HIP/ASM symbols.
- dsv4 batched mxscale BMM: libtype=opus only.
- Qwen3.5 gated-delta-net chunk shapes: the default code path IS Triton, but those
  model_configs CSVs carry zero gfx950 rows (gfx942 tuning only), so nothing here is
  gfx950-validated and there is no upstream bench to reuse.

Only the Triton timing helpers of the upstream benches are reused (bench_gemm_fn), so the
numbers stay directly comparable to the generic sweeps and no kernel-invocation logic is
duplicated here.
"""

import argparse
import sys

import triton

import aiter.ops.triton.utils._triton.arch_info as arch_info
from op_tests.op_benchmarks.triton.bench_gemm_a16w16 import (
    bench_gemm_fn as bench_a16w16_fn,
)
from op_tests.op_benchmarks.triton.bench_gemm_a8w8_blockscale import (
    bench_gemm_fn as bench_a8w8_blockscale_fn,
    triton_gemm_a8w8_blockscale_preshuffle,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# (M, N, K) per model, from *_bf16_tuned_gemm.csv where gfx=gfx950 and libtype=triton.
# Production dispatch: aiter/tuned_gemm.py solMap["triton"] -> triton_gemm -> gemm_a16w16.
BF16_SHAPES = {
    "glm5.3": [
        (1, 1024, 128), (1, 4096, 1024), (1, 4096, 1536),
        (2, 288, 4096), (2, 1024, 128), (2, 4096, 1024), (2, 4096, 1536),
        (4, 288, 4096), (4, 1024, 128), (4, 4096, 1024), (4, 4096, 1536),
        (8, 1024, 128), (8, 4096, 1024),
        (16, 8, 4096), (16, 32, 4096), (16, 288, 4096), (16, 1024, 128),
        (16, 3072, 4096), (16, 4096, 1024), (16, 4096, 1536),
        (32, 1024, 128), (32, 4096, 1024), (32, 4096, 1536),
        (64, 1024, 128), (64, 4096, 1024), (64, 4096, 1536),
        (128, 1024, 128), (128, 4096, 1024),
    ],
    "glm5": [
        (1, 3584, 512),
        (2, 2048, 2048), (2, 3584, 512),
        (4, 2048, 2048),
        (8, 3584, 512),
        (10, 3584, 512), (10, 6144, 2048), (10, 7168, 512),
        (10, 19360, 6144), (10, 38720, 6144),
        (12, 2048, 2048), (12, 3584, 512), (12, 38720, 6144),
        (14, 7168, 512),
        (16, 4096, 2048), (16, 38720, 6144),
        (24, 6144, 2048), (24, 6144, 4096), (24, 7168, 512),
        (32, 6144, 3072), (32, 6144, 4096), (32, 38720, 6144),
        (64, 6144, 3072), (64, 6144, 4096),
        (128, 6144, 3072),
    ],
    "kimik3": [
        (2, 2304, 1536),
        (16, 2304, 1536), (16, 3072, 512), (16, 7168, 768),
        (32, 7168, 768), (32, 7168, 1536), (32, 7168, 4224),
        (64, 7168, 1536), (64, 7168, 3584),
        (256, 7168, 768), (256, 7168, 1536), (256, 7168, 3584),
    ],
    "qwen3.5-397b": [
        (1, 4096, 512), (2, 4096, 512),
        (16, 4096, 512), (16, 8192, 4096),
        (32, 8192, 4096), (64, 8192, 4096), (128, 8192, 4096),
        (512, 4096, 512),
    ],
    "dsv4": [
        (16, 64640, 4096),
    ],
}

# (M, N, K) from dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv, gfx950 + libtype=triton.
# Production dispatch: aiter/ops/gemm_op_a8w8.py -> _gemm_a8w8_blockscale_preshuffle_triton.
# dsv4 is the only one of the four models with Triton rows for this op.
A8W8_BLOCKSCALE_PRESHUFFLE_SHAPES = {
    "dsv4": [
        (1, 2048, 7168), (1, 7168, 16384), (1, 16384, 1536), (1, 65536, 1536),
        (2, 2048, 7168), (2, 7168, 16384), (2, 65536, 1536),
        (4, 2048, 7168), (4, 7168, 16384), (4, 16384, 1536), (4, 65536, 1536),
        (8, 2048, 7168), (8, 7168, 16384), (8, 16384, 1536), (8, 65536, 1536),
        (16, 2048, 7168), (16, 7168, 16384), (16, 16384, 1536),
        (32, 2048, 7168), (32, 7168, 16384),
        (48, 2048, 7168),
        (64, 2048, 7168), (64, 7168, 16384),
        (80, 2048, 7168),
        (128, 2048, 7168),
    ],
}

METRIC_TO_UNIT = {"throughput": "TFLOPS", "time": "Time_ms"}


def _x_vals(shapes_by_model, models):
    """Flatten {model: [(M,N,K)]} into perf_report x_vals rows, keeping model as a column
    so a regression can be attributed to the model whose layer it belongs to."""
    rows = []
    for model, shapes in shapes_by_model.items():
        if models and model not in models:
            continue
        for m, n, k in shapes:
            rows.append([model, m, n, k])
    return rows


def _benchmark(plot_name, x_vals, metrics):
    return triton.testing.Benchmark(
        x_names=["model", "M", "N", "K"],
        x_vals=x_vals,
        line_arg="metric",
        line_vals=metrics,
        line_names=[METRIC_TO_UNIT[m] for m in metrics],
        styles=[("green", "-"), ("blue", "-")][: len(metrics)],
        ylabel="/".join(METRIC_TO_UNIT[m] for m in metrics),
        plot_name=plot_name,
        args={},
    )


def run_bf16(args, metrics):
    x_vals = _x_vals(BF16_SHAPES, args.models)
    if not x_vals:
        return
    # Pin backend="triton": production picked the Triton kernel for these rows, and the
    # auto path would silently choose Gluon on gfx1250, which is a different kernel.
    @triton.testing.perf_report([_benchmark("bench_gemm_a16w16_model_shapes", x_vals, metrics)])
    def bench(model, M, N, K, metric, **kwargs):
        return bench_a16w16_fn(M, N, K, metric, args.layout, "triton")

    bench.run(save_path="." if args.o else None, print_data=True)


def run_a8w8_blockscale_preshuffle(args, metrics):
    x_vals = _x_vals(A8W8_BLOCKSCALE_PRESHUFFLE_SHAPES, args.models)
    if not x_vals:
        return
    if not arch_info.is_fp8_avail():
        print(
            f"Skipping a8w8_blockscale_preshuffle: fp8 not available on {arch_info.get_arch()}"
        )
        return

    @triton.testing.perf_report(
        [_benchmark("bench_gemm_a8w8_blockscale_preshuffle_model_shapes", x_vals, metrics)]
    )
    def bench(model, M, N, K, metric, **kwargs):
        return bench_a8w8_blockscale_fn(
            M,
            N,
            K,
            metric,
            args.layout,
            triton_gemm_a8w8_blockscale_preshuffle,
            shuffle=True,
        )

    bench.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog=f"Benchmark {get_caller_name_no_ext()}",
        description="Triton GEMM shapes really used by glm5 / dsv4 / kimik3 / qwen3.5 on gfx950",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--op",
        choices=["all", "bf16", "a8w8_blockscale_preshuffle"],
        default="all",
        help="Which op family to benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict to these model keys (default: all). "
        f"BF16: {sorted(BF16_SHAPES)}; a8w8: {sorted(A8W8_BLOCKSCALE_PRESHUFFLE_SHAPES)}",
    )
    parser.add_argument(
        "--metric",
        choices=["throughput", "time", "both"],
        default="both",
        help="Metric(s) to report.",
    )
    parser.add_argument(
        "--layout",
        choices=["TT", "TN", "NT", "NN"],
        default="TN",
        help="Input/weight layout.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = ["throughput", "time"] if args.metric == "both" else [args.metric]

    arch = arch_info.get_arch()
    if arch != "gfx950":
        # Not fatal: the kernels are arch-agnostic. But the shapes were selected from
        # gfx950 tuning rows, so on another arch production may not use Triton at all.
        print(
            f"WARNING: these shapes are the gfx950 libtype=triton rows; running on {arch}"
        )

    if args.op in ("all", "bf16"):
        run_bf16(args, metrics)
    if args.op in ("all", "a8w8_blockscale_preshuffle"):
        run_a8w8_blockscale_preshuffle(args, metrics)


if __name__ == "__main__":
    sys.exit(main())
