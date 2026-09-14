"""
Benchmark the gfx950 kernel configurations published as LLVM IR dumps in
/data/triton-perf-ir-0824.

Why this file exists
--------------------
Those dumps are the reference corpus the LLVM team does static codegen/regalloc
analysis against, so a Triton or LLVM change that moves one of them matters even
when every generic sweep stays flat. Most of the corpus is already monitored:
mla_gluon's four bh64 shapes, flash_kda (1,16384,12,128,128), gluon pa_decode's
sliding-window config and both gmm shapes all sit in some bench's default sweep.

The cases below are the ones that do not. Each needs a non-default argument --
a different op mode, a shape outside the sweep, or a flag that selects another
kernel variant -- and the daily runs every bench script bare, so none of them
would ever execute.

Why subprocess instead of importing the benches
-----------------------------------------------
Several of these scripts write a fixed-name CSV (bench_rmsnorm.csv,
bench_fp8_mqa_logits.csv, bench_pa_decode_gluon_normal_bf16_blk64_ql4.csv) that
the daily already produces from the same script's default sweep. Re-running the
script in this directory would overwrite that file and the default-sweep result
would be lost. Each case therefore runs in its own temporary working directory
and its CSV is moved out under a bench_perf_ir_* name, so both survive.

Running the documented command verbatim also keeps this file honest: the
commands are copied from the corpus README, so a case that stops reproducing its
dump shows up here rather than silently drifting.

Deliberately NOT included
-------------------------
- bench_gemm_* / bench_fused_gemm_* dumps: excluded by request.
- moe_gluon_gemm1_bf16: the corpus README states it was built from a patched
  triton + patched llvm and is not reproducible on a stock toolchain. There is
  also no in-tree bf16 MoE caller, so there is nothing for a daily to run.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# Per-case timeout. The slowest case here (mha at seq 16384) takes a couple of
# minutes; anything past this is a hang, and the daily should not sit on it.
CASE_TIMEOUT_S = 1800


class Case:
    # Most benches take a bare `-o`; bench_mha.py's `-o` takes the output
    # directory instead, so the flag is per-case rather than appended blindly.
    def __init__(self, name, script, argv, env=None, note="", out_args=("-o",)):
        self.name = name
        self.script = script
        # Kept as one string so it stays diffable against the corpus README.
        self.argv = shlex.split(argv)
        self.env = env or {}
        self.note = note
        self.out_args = list(out_args)

    @property
    def csv_name(self):
        return f"bench_perf_ir_{self.name}.csv"


# Commands are verbatim from the corpus README; the dump each one produced is
# named in `note`.
CASES = [
    Case(
        "attn_res_gate",
        "bench_attn_res.py",
        # Default is --op fwd over the prefill sweep, which never builds the
        # gate kernel this dump came from.
        "--op gate -N 65536 -D 7168 -L 9 --add-hidden --onorm --metric bandwidth",
        note="attn_res_n65536d7168l9_5.2tbs (attnres_fwd_kernel)",
    ),
    Case(
        "mha_fwd_causal",
        "bench_mha.py",
        "-fn fwd --dtype bf16 -b 1 -hq 16 -hk 16 -sq 16384 -d 128 -causal 1",
        # The dump was taken with packed-fop scalarization on; without it the
        # kernel this corpus tracks is not the one that gets built.
        env={"AMDGCN_SCALARIZE_PACKED_FOPS": "1"},
        note="fa_s16384h16d128_causal_709tflops (_mha_fwd_gluon_kernel)",
        out_args=("-o", "."),
    ),
    Case(
        "mha_fwd_noncausal",
        "bench_mha.py",
        "-fn fwd --dtype bf16 -b 1 -hq 16 -hk 16 -sq 16384 -d 128 -causal 0",
        env={"AMDGCN_SCALARIZE_PACKED_FOPS": "1"},
        note="fa_s16384h16d128_noncausal_961tflops (_mha_fwd_gluon_kernel)",
        out_args=("-o", "."),
    ),
    Case(
        "rmsnorm_m32768_n16384",
        "bench_rmsnorm.py",
        # get_x_vals() tops out at N=1280 for large M; (32768, 16384) is the
        # large-M large-N corner and is not in it.
        "-M 32768 --shape 32768 16384 --metric bandwidth",
        note="rmsnorm_m32768_n16384_5tbs (_rms_norm_kernel)",
    ),
    Case(
        "fp8_mqa_logits_clean0",
        "bench_fp8_mqa_logits.py",
        # Every other argument here is already the script default; only
        # --clean_logits 0 selects the variant this dump came from.
        "--batch_size 1 --seq_q_l 4096 --seq_kv_l 4096 --num_heads_q 64 --head_dim 128 --clean_logits 0",
        note="mqa_logits_b1s4096_1474tflops (fp8_mqa_logits)",
    ),
    Case(
        "pa_decode_gluon_h64x8",
        "bench_pa_decode_gluon.py",
        # The daily's bare run sweeps num_heads (64, 4); this dump is the
        # (64, 8) GQA ratio, which is a different BLOCK_H specialization.
        "--mode normal --compute_type bf16 --num_heads 64 8 --head_dim 128 --batch_size 128 --context_length 8192 --block_size 64 --query_length 4 --quant_mode per_tensor",
        note="gluon_pa_decode_bf16_b128h64x8_3.5tbs",
    ),
]


def run_case(case, keep_csv):
    script = BENCH_DIR / case.script
    if not script.is_file():
        print(f"SKIP {case.name}: {case.script} not found in {BENCH_DIR}")
        return None

    env = {**os.environ, **case.env}
    # Each case gets its own cwd so the CSV it drops is unambiguous and cannot
    # collide with the daily's own run of the same script.
    with tempfile.TemporaryDirectory(prefix=f"perf_ir_{case.name}_") as workdir:
        cmd = [sys.executable, "-u", str(script), *case.argv, *case.out_args]
        print(
            f"\n{'=' * 70}\n{case.name}  <- {case.note}\n$ {' '.join(cmd)}\n{'=' * 70}"
        )
        proc = subprocess.run(
            cmd, cwd=workdir, env=env, timeout=CASE_TIMEOUT_S, check=False
        )
        if proc.returncode != 0:
            print(f"FAILED {case.name}: exit {proc.returncode}")
            return False

        produced = sorted(Path(workdir).glob("*.csv"))
        if len(produced) != 1:
            names = [p.name for p in produced] or ["<none>"]
            print(f"FAILED {case.name}: expected exactly one CSV, got {names}")
            return False

        if keep_csv:
            dest = BENCH_DIR / case.csv_name
            shutil.copyfile(produced[0], dest)
            print(f"{case.name}: {produced[0].name} -> {dest.name}")

    return True


def main():
    parser = argparse.ArgumentParser(
        prog="Benchmark /data/triton-perf-ir-0824 reference cases",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[c.name for c in CASES],
        help="Run only the named case (repeatable). Default: all.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV files"
    )
    args = parser.parse_args()

    cases = CASES if not args.case else [c for c in CASES if c.name in args.case]

    failed, skipped = [], []
    for case in cases:
        result = run_case(case, keep_csv=args.o)
        if result is None:
            skipped.append(case.name)
        elif result is False:
            failed.append(case.name)

    print(
        f"\n{'=' * 70}\nperf-ir cases: {len(cases)} total, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )
    for name in skipped:
        print(f"  skipped: {name}")
    for name in failed:
        print(f"  failed:  {name}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
