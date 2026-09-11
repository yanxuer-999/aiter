#!/usr/bin/env python3
"""
All-in-one Triton Bench Test Runner & Excel Report Generator
=============================================================
Run all bench tests with multiple triton wheels, parse outputs, and generate
a comparison Excel report with per-sheet categorization.

Usage:
    # Run all tests with 3 wheels and generate Excel:
    python run_all_bench.py \
        --wheels /path/to/triton-3.6.0+git756afc06.whl \
                 /path/to/triton-3.6.0+git7409f166.whl \
                 /path/to/triton-3.7.0+gitd1660454.whl \
        --output /home/mejiang/w1/bench_test_results_latest.xlsx

    # Skip running, just regenerate Excel from existing logs:
    python run_all_bench.py \
        --wheels ... \
        --skip-run \
        --output /home/mejiang/w1/bench_test_results_latest.xlsx

    # Run only specific tests:
    python run_all_bench.py --wheels ... --tests bench_mha bench_gemm_a16w16

    # Run only a specific sheet category:
    python run_all_bench.py --wheels ... --sheet gemm

    # Compare the two AMD PyPI index builds (release_ baseline vs release_tmp new),
    # auto-detecting the ROCm version, and generate a comparison Excel with the
    # 性能变化统计 table:
    python run_all_bench.py --from-index --output bench_test_results.xlsx
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BENCH_DIR = PROJECT_ROOT / "op_tests" / "op_benchmarks" / "triton"

# ============================================================================
# Test Definitions
# ============================================================================
# (test_name, extra_args, sheet_category)
# Categories map to per-sheet output and to the 性能变化统计 stats table:
#   batched_gemm, gemm, moe, attention, normalization, routing_topk, other
# The full list is the set of bench_*.py under
# aiter/op_tests/op_benchmarks/triton on the `main` branch.

_MOE_GEMM_ARGS = "--shape 7168 4096 --experts 256 8"
# a8w8_blockscale's tuned gfx950 config picks a 288KB-LDS tile for every M>=256
# (> the 160KB hardware limit), independent of the GEMM shape, so it OutOfResources
# right after batch 128. Cap the batch sweep to the M<=128 buckets that fit.
_MOE_GEMM_BLOCKSCALE_ARGS = _MOE_GEMM_ARGS + " --M 1 2 4 8 16 32 64 128"

BENCH_TESTS = [
    # ----- Batched GEMM -----
    ("bench_batched_gemm_a16wfp4",                                                  "", "batched_gemm"),
    ("bench_batched_gemm_a8w8",                                                     "", "batched_gemm"),
    ("bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant", "", "batched_gemm"),
    ("bench_batched_gemm_afp4wfp4",                                                 "", "batched_gemm"),
    ("bench_batched_gemm_bf16",                                                     "", "batched_gemm"),
    # ----- GEMM -----
    ("bench_ff_a16w16_fused",                "", "gemm"),
    ("bench_fused_gemm_a8w8_blockscale_a16w16", "", "gemm"),
    ("bench_fused_gemm_afp4wfp4_a16w16",     "", "gemm"),
    ("bench_gemm_a16w16",                    "", "gemm"),
    ("bench_gemm_a16w16_gated",              "", "gemm"),
    ("bench_gemm_a16w16_gating",             "", "gemm"),
    ("bench_gemm_a16w8_blockscale",          "", "gemm"),
    ("bench_gemm_a8w8",                      "", "gemm"),
    ("bench_gemm_a8w8_blockscale",           "", "gemm"),
    ("bench_gemm_a8w8_per_token_scale",      "", "gemm"),
    ("bench_gemm_a8wfp4",                    "", "gemm"),
    ("bench_gemm_afp4wfp4",                  "", "gemm"),
    ("bench_gemm_afp4wfp4_pre_quant_atomic", "", "gemm"),
    ("bench_gmm",                            "", "gemm"),
    # ----- MoE -----
    # Upstream renamed the a4w4/a8w4 MoE benches to *_proton (a *_cudagraph
    # variant of each also exists, on a different timing methodology than the
    # rest of the suite). bench_moe / bench_moe_mx /
    # bench_moe_align_block_size / bench_moe_routing_sigmoid_top1_fused have no
    # script in this checkout at all -- listing them only buys a 0-second FAIL
    # per run, so they stay out until they come back.
    ("bench_moe_gemm_a16w4",                 _MOE_GEMM_ARGS, "moe"),
    ("bench_moe_gemm_a4w4_proton",           _MOE_GEMM_ARGS, "moe"),
    ("bench_moe_gemm_a8w4_proton",           _MOE_GEMM_ARGS, "moe"),
    ("bench_moe_gemm_a8w8",                  _MOE_GEMM_ARGS, "moe"),
    ("bench_moe_gemm_a8w8_blockscale",       _MOE_GEMM_BLOCKSCALE_ARGS, "moe"),
    ("bench_moe_gemm_int8_smoothquant",      _MOE_GEMM_ARGS, "moe"),
    # ----- Attention -----
    # Kimi-K3 runs attn_res_gate with add_hidden + output RMSNorm + fused
    # per-token FP8 output quant; the bench's default (--op fwd, no fusions) is a
    # different kernel configuration, so drive the production one explicitly.
    ("bench_attn_res",                       "--op gate --add-hidden --onorm --quant", "attention"),
    ("bench_batch_prefill",                  "-b 4 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 -causal", "attention"),
    ("bench_deepgemm_attention",             "", "attention"),
    ("bench_extend_attention",               "", "attention"),
    ("bench_fav3_sage",                      "-b 4 -hq 32 -sq 1024 -d 128", "attention"),
    ("bench_fav3_sage_mxfp4",                "-b 4 -hq 32 -sq 1024 -d 128", "attention"),
    ("bench_flash_kda",                      "", "attention"),
    ("bench_fp8_mqa_logits",                 "", "attention"),
    ("bench_fused_kda_decode",               "", "attention"),
    ("bench_gather_kv_b_proj",               "", "attention"),
    ("bench_hstu_attn",                      "", "attention"),
    # bench_la (bare lean-attention) hangs / times out and is effectively
    # covered by bench_la_paged_decode; excluded on purpose.
    ("bench_la_paged_decode",                "", "attention"),
    ("bench_mha",                            "", "attention"),
    ("bench_mhc",                            "", "attention"),
    ("bench_mla",                            "", "attention"),
    # x_names/x_vals_list are only bound on the --model path, so an argument-less
    # invocation dies with UnboundLocalError before it benchmarks anything.
    ("bench_mla_decode",                     "--model deepseek -b 1 --seqlen 8192", "attention"),
    ("bench_mla_decode_rope",                "", "attention"),
    # mla_gluon had no perf coverage at all: bench_mla* are the non-gluon Triton
    # kernels (bench_mla's "Gluon" is only a column label), and the one bench
    # that does call mla_gluon -- sparse_attention_dsv4, the has_pe=False
    # prefill leg -- was never in this list. These two cover both legs; the
    # dsv4 cfg is the reference config published in the gluon README.
    ("bench_mla_gluon_decode",               "", "attention"),
    ("bench_sparse_attention_dsv4",          "--prefill_cfgs 4096,128,4096,1024", "attention"),
    ("bench_pa_decode",                      "", "attention"),
    ("bench_pa_prefill",                     "", "attention"),
    ("bench_sage",                           "--b 4 --hq 32 --sq 1024 --d 128", "attention"),
    ("bench_unified_attention",              "", "attention"),
    ("bench_unified_attention_sparse_mla",   "", "attention"),
    # ----- Normalization -----
    ("bench_rmsnorm",                        "", "normalization"),
    # ----- Routing / TopK -----
    ("bench_topk",                           "", "routing_topk"),
    # ----- Other -----
    ("bench_cache_copy",                     "", "other"),
    # bench_rope is excluded on purpose: it contains no timing code at all (no
    # do_bench, no events) and prints only shape-derived Total flops/Total memory
    # plus a note telling you to use rocprof, because its kernels are too short
    # for do_bench. It can never produce a cross-wheel delta -- keeping it only
    # bought a PASS row with NO DATA. Measure rope separately:
    #   rocprofv2 --kernel-trace python bench_rope.py -l 'thd' -T 1 -H 128 -D 64 --two_inputs=true
]

# Display order and human labels for sheets / stats table.
CATEGORY_ORDER = [
    "batched_gemm", "gemm", "moe", "attention",
    "normalization", "routing_topk", "other",
]

CATEGORY_LABELS = {
    "batched_gemm":  "Batched GEMM",
    "gemm":          "GEMM",
    "moe":           "MoE",
    "attention":     "Attention",
    "normalization": "Normalization",
    "routing_topk":  "Routing/TopK",
    "other":         "Other",
}

SHEET_NAMES = {
    "batched_gemm":  "Batched GEMM Benchmarks",
    "gemm":          "GEMM Benchmarks",
    "moe":           "MoE Benchmarks",
    "attention":     "Attention Benchmarks",
    "normalization": "Normalization Benchmarks",
    "routing_topk":  "Routing TopK Benchmarks",
    "other":         "Other Benchmarks",
}

# Tests that appear in multiple sheets share the same log; only run once.
# Dedup by test name for execution, but include in all mapped sheets for reporting.
def get_unique_tests(tests):
    """Return deduplicated list for execution (first occurrence wins)."""
    seen = set()
    unique = []
    for name, args, _ in tests:
        if name not in seen:
            seen.add(name)
            unique.append((name, args))
    return unique


# ============================================================================
# Utility
# ============================================================================

def get_version_label(wheel_path):
    """Extract short git hash from wheel filename."""
    basename = os.path.basename(wheel_path)
    m = re.search(r'git([0-9a-f]+)', basename)
    return m.group(1) if m else basename.split("-")[1] if "-" in basename else basename


def detect_rocm_major_minor():
    """Detect ROCm major.minor from rocm-core (e.g. '7.1'). Returns None if unavailable."""
    out = subprocess.run(
        ["dpkg", "-l", "rocm-core"],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("ii"):
            ver = line.split()[2]               # e.g. 7.1.1.70101-38~24.04
            return ".".join(ver.split(".")[:2])  # -> 7.1
    return None


def clear_triton_cache():
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print("  Cleared ~/.triton/cache")


def print_triton_version():
    # Use installed package metadata, not triton.__version__ — the latter drops
    # the local version suffix (+amd.rocmX.gitHASH), which we need for the label.
    ver = subprocess.run(
        [sys.executable, "-c",
         "from importlib.metadata import version; print(version('triton'))"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"  triton={ver}")
    return ver


def install_wheel(wheel_path):
    """Install a triton wheel and clear cache."""
    label = get_version_label(wheel_path)
    print(f"\n{'=' * 60}")
    print(f"Installing {os.path.basename(wheel_path)}  (label: {label})")
    print(f"{'=' * 60}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", wheel_path,
         "--force-reinstall", "--no-deps", "-q"],
        check=True, capture_output=True,
    )
    clear_triton_cache()
    print_triton_version()
    return label


# AMD PyPI index variants to compare, in order: first = baseline, last = new.
#   release_     -> the current published release  (baseline)
#   release_tmp  -> the candidate build under test (new)
INDEX_VARIANTS = ["release_", "release_tmp"]


def index_url_for(index_name, mm):
    return f"https://pypi.amd.com/triton/{index_name}/rocm-{mm}.0/simple/"


def uninstall_all_triton():
    """Clean-uninstall every triton variant; a plain `pip install triton` will
    NOT upgrade an already-satisfied unconstrained requirement, so the existing
    triton would otherwise stick."""
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y",
         "triton", "pytorch-triton", "pytorch-triton-rocm", "triton-rocm",
         "amd-triton", "triton-kernels"],
        check=False,
    )


def fetch_index_label(index_name, mm):
    """Resolve an index variant's triton git-hash label WITHOUT installing,
    by reading the simple-index HTML. Used by --skip-run. Falls back to the
    index name if the page can't be read."""
    import urllib.request
    url = index_url_for(index_name, mm) + "triton/"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            html = resp.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  warn: could not fetch {url}: {e}")
        return index_name
    hashes = re.findall(r'triton-[^"<#]*?git([0-9a-f]+)', html)
    return hashes[0] if hashes else index_name


def install_from_index(index_name, mm):
    """Install triton + triton-kernels from one AMD PyPI index variant.

    Returns the short version label (git hash) derived from the installed
    triton wheel version, e.g. 'd0d77a509'.
    """
    index_url = index_url_for(index_name, mm)
    print(f"\n{'=' * 60}")
    print(f"Installing triton + triton-kernels from index '{index_name}'")
    print(f"  {index_url}")
    print(f"{'=' * 60}")
    uninstall_all_triton()
    for pkg in ("triton", "triton-kernels"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg,
             "--extra-index-url", index_url],
            check=True,
        )
    clear_triton_cache()
    ver = print_triton_version()
    m = re.search(r'git([0-9a-f]+)', ver)
    return m.group(1) if m else (ver or index_name)


# A TIMEOUT is often transient (a hung kernel autotune / contended GPU), so
# retry it a few more times before giving up. Total attempts = 1 + TIMEOUT_RETRIES.
TIMEOUT_RETRIES = 2


def _run_one(name, extra_args, log_file, env, timeout):
    """Run a single bench once. Returns (status, elapsed). Writes log_file."""
    script = BENCH_DIR / f"{name}.py"

    # bench_pa_decode_gluon outputs \r which corrupts stdout redirection;
    # wrap with `script` + `tr` to strip carriage returns.
    needs_cr_fix = ("pa_decode_gluon" in name)

    cmd = f"{sys.executable} {script}"
    if extra_args:
        cmd += f" {extra_args}"

    if needs_cr_fix:
        # Use script(1) to capture terminal output with \r, then strip \r
        shell_cmd = f"script -q -c '{cmd}' /dev/null 2>/dev/null | tr -d '\\r'"
    else:
        shell_cmd = cmd

    t0 = time.time()
    try:
        proc = subprocess.run(
            shell_cmd, shell=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        with open(log_file, "w") as f:
            f.write(proc.stdout)
            if proc.stderr and not needs_cr_fix:
                f.write(proc.stderr)
        return ("PASS" if proc.returncode == 0 else "FAIL"), elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        with open(log_file, "w") as f:
            f.write(f"TIMEOUT after {timeout}s\n")
        return "TIMEOUT", elapsed


def run_tests(tests, log_dir, timeout):
    """Run a list of (name, args) tests, save logs, return status dict.

    On TIMEOUT a test is retried up to TIMEOUT_RETRIES additional times; the
    first non-TIMEOUT result wins, otherwise the final TIMEOUT is recorded.
    """
    os.makedirs(log_dir, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    results = {}
    total = len(tests)

    for idx, (name, extra_args) in enumerate(tests, 1):
        log_file = os.path.join(log_dir, f"{name}.log")
        print(f"  [{idx}/{total}] {name} ...", end=" ", flush=True)

        status, elapsed = _run_one(name, extra_args, log_file, env, timeout)
        attempt = 1
        while status == "TIMEOUT" and attempt <= TIMEOUT_RETRIES:
            print(f"TIMEOUT ({elapsed:.0f}s); retry {attempt}/{TIMEOUT_RETRIES} ...",
                  end=" ", flush=True)
            # Clear the triton cache between tries in case a bad compiled
            # artifact is the cause of the hang.
            clear_triton_cache()
            status, elapsed = _run_one(name, extra_args, log_file, env, timeout)
            attempt += 1

        results[name] = status
        print(f"{status} ({elapsed:.0f}s)")

    return results


def run_tests_repeated(tests, version_dir, timeout, repeat):
    """Run the suite `repeat` times for one version and return an aggregated
    status dict. With repeat>1 each pass writes to <version_dir>/run<k>/; with
    repeat==1 it writes flat into <version_dir> (backward compatible).

    Per-data-point values are medianed at report time (see extract_data_aggregated);
    here we only aggregate PASS/FAIL/TIMEOUT status across passes (PASS wins if any
    pass produced a good result)."""
    os.makedirs(version_dir, exist_ok=True)
    per_rep = []
    for k in range(1, repeat + 1):
        rep_dir = os.path.join(version_dir, f"run{k}") if repeat > 1 else version_dir
        if repeat > 1:
            print(f"\n--- repeat {k}/{repeat} -> {os.path.basename(rep_dir)} ---")
        per_rep.append(run_tests(tests, rep_dir, timeout))

    agg = {}
    for name, _ in tests:
        sts = [r.get(name, "N/A") for r in per_rep]
        if "PASS" in sts:
            agg[name] = "PASS"
        elif all(s == "N/A" for s in sts):
            agg[name] = "N/A"
        elif all(s == "TIMEOUT" for s in sts):
            agg[name] = "TIMEOUT"
        else:
            agg[name] = "FAIL"
    return agg


# ============================================================================
# Log Parsers
# ============================================================================

def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    # Values carrying a unit suffix, e.g. bench_fused_kda_decode's "4.96x"
    # speedup column. Strip anything after the numeric prefix.
    m = re.match(r'^[-+]?\d*\.?\d+', str(s).strip())
    return float(m.group(0)) if m else None


METRIC_KEYWORDS = [
    "tflops", "throughput", "bandwidth", "time_", "time(ms)", "time (ms)",
    "triton(ms)", "triton (", "tbps", "kernel latency", "latency",
    "tflop", "gbps", "gb/s", "tb/s",
    # "_time" catches bench_attn_res's "sequence_time"/"packed_time"; "(us)"
    # catches "Time(us)" / "3-kernel(us)" / "Fused(us)"; "speedup" catches
    # bench_fused_kda_decode's trailing ratio column. Without these the columns
    # are classified as *shape* columns and their measured values end up inside
    # the param key, which makes the same shape look like a different row on
    # every run and silently breaks the cross-version diff.
    "_time", "(us)", "speedup",
    # "(ms)" catches bench_mla_decode's "mla_decode_fwd (ms)" — the specific
    # "time (ms)"/"triton(ms)" spellings above miss any kernel-named ms column,
    # so the whole table was classified no_data and the test PASSed with 0 rows.
    "(ms)",
]


def detect_output_type(filepath):
    """Detect log format: moe_gemm, pa_decode_gluon, tabular, or no_data."""
    if not os.path.exists(filepath):
        return "empty"
    with open(filepath) as f:
        content = f.read()  # read full file for reliable detection
    if re.search(r'batch:\s+\d+\s*\|.*TFLOPS:', content):
        return "moe_gemm"
    if re.search(r'^bench_pa_decode_gluon\S+:', content, re.MULTILINE):
        return "pa_decode_gluon"
    # A markdown separator row is the reliable tell: pandas tables never emit one.
    if re.search(r'^\|[\s:-]*-{2,}[\s:|-]*\|$', content, re.MULTILINE):
        return "pipe_table"
    lower = content.lower()
    if any(kw in lower for kw in METRIC_KEYWORDS):
        return "tabular"
    return "no_data"


def parse_tabular_output(filepath):
    """Parse pandas DataFrame-style tabular output.
    Returns list of data rows (each a list of strings) and header names.
    """
    rows, headers = [], []
    if not os.path.exists(filepath):
        return rows, headers
    with open(filepath) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(kw in lower for kw in METRIC_KEYWORDS):
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j].strip()
                if nxt and re.match(r'^\d', nxt):
                    header_idx = i
                    break
            if header_idx is not None:
                break

    if header_idx is None:
        return rows, headers

    header_line = lines[header_idx].strip()
    headers = [h.strip() for h in re.split(r'\s{2,}', header_line) if h.strip()]

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('[') or stripped.startswith('DEBUG'):
            continue
        if 'Traceback' in stripped or 'Error' in stripped:
            break
        parts = [p.strip() for p in re.split(r'\s{2,}', stripped) if p.strip()]
        if len(parts) >= 2:
            try:
                float(parts[0])
                rows.append(parts)
            except ValueError:
                continue

    return rows, headers


def parse_moe_gemm_output(filepath):
    """Parse MOE gemm pipe-delimited output.
    Returns list of dicts with keys: batch, total_latency_us, kernel_latency_us, TFLOPS, TBPS.
    """
    rows = []
    if not os.path.exists(filepath):
        return rows
    with open(filepath) as f:
        for line in f:
            m = re.match(
                r'batch:\s+(\d+)\s*\|\s*Total latency \(us\):\s*([\d.]+)\s*\|'
                r'\s*Kernel latency \(us\):\s*([\d.]+)\s*\|\s*TFLOPS:\s*([\d.]+)'
                r'\s*\|\s*TBPS:\s*([\d.]+)',
                line.strip(),
            )
            if m:
                rows.append({
                    "batch": int(m.group(1)),
                    "total_latency_us": float(m.group(2)),
                    "kernel_latency_us": float(m.group(3)),
                    "TFLOPS": float(m.group(4)),
                    "TBPS": float(m.group(5)),
                })
    return rows


def parse_pipe_table(test_name, filepath):
    """Parse markdown pipe tables (bench_sparse_attention_dsv4).

    Returns list of (benchmark, params, metric, value) tuples.

    Unlike the pandas tables, metric columns here are not all trailing: the row
    reads `Q H Kv topk | triton ms | triton TFLOPS | ...`. Split instead at the
    first non-integer cell, since shape columns are integers and every measured
    column is fractional.
    """
    rows = []
    if not os.path.exists(filepath):
        return rows
    with open(filepath) as f:
        lines = f.readlines()

    section, headers = None, None
    for line in lines:
        stripped = line.strip()
        m = re.match(r'^=+\s*([A-Za-z0-9_ /-]+?)\s*=+$', stripped)
        if m:
            section, headers = m.group(1).strip(), None
            continue
        if not (stripped.startswith('|') and stripped.endswith('|')):
            continue
        cells = [c.strip() for c in stripped.strip('|').split('|')]
        if all(re.fullmatch(r'[-:\s]*', c) for c in cells):
            continue  # separator row
        if headers is None:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue

        split_at = next(
            (i for i, c in enumerate(cells) if not re.fullmatch(r'-?\d+', c)),
            len(cells),
        )
        param_vals = [f"{headers[i]}={cells[i]}" for i in range(split_at)]
        param_str = " x ".join(param_vals) if param_vals else "-"
        if section:
            param_str = f"{section} | {param_str}"

        for i in range(split_at, len(cells)):
            # Values like "2.21x" carry a unit suffix; keep only the number.
            val = safe_float(re.sub(r'[^\d.eE+-]', '', cells[i]))
            if val is not None:
                rows.append((test_name, param_str, headers[i], val))
    return rows


def parse_pa_decode_gluon(filepath):
    """Parse pa_decode_gluon log with multiple named sub-tables.
    Returns list of (subtable_name, params, metric, value) tuples.
    """
    rows = []
    if not os.path.exists(filepath):
        return rows
    with open(filepath) as f:
        lines = f.readlines()

    current_subtable = None
    header = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(bench_pa_decode_gluon\S+):$', line)
        if m:
            current_subtable = m.group(1)
            header = None
            continue
        if current_subtable and 'batch_size' in line and 'context_length' in line:
            header = line
            continue
        if current_subtable and header and re.match(r'^\d+\s+', line):
            parts = line.split()
            if len(parts) >= 5:
                param_str = f"bs={parts[1]}, ctx={parts[2]}"
                rows.append((current_subtable, param_str, "Time_(ms)", float(parts[3])))
                rows.append((current_subtable, param_str, "Bandwidth_(TB/s)", float(parts[4])))
    return rows


# ============================================================================
# Unified data extraction: log -> list of (benchmark, params, metric, value)
# ============================================================================

def extract_data(test_name, filepath):
    """Extract all data points from a log file.
    Returns list of (benchmark_name, param_string, metric_name, float_value).
    """
    otype = detect_output_type(filepath)
    results = []

    if otype == "moe_gemm":
        for r in parse_moe_gemm_output(filepath):
            batch = r["batch"]
            for metric in ["TFLOPS", "kernel_latency_us", "total_latency_us", "TBPS"]:
                results.append((test_name, f"batch={batch}", metric, r[metric]))

    elif otype == "pa_decode_gluon":
        results = parse_pa_decode_gluon(filepath)

    elif otype == "pipe_table":
        results = parse_pipe_table(test_name, filepath)

    elif otype == "tabular":
        data_rows, headers = parse_tabular_output(filepath)
        if not headers or not data_rows:
            return results

        # Determine param vs metric columns
        metric_count = 0
        for h in reversed(headers):
            if any(kw in h.lower() for kw in METRIC_KEYWORDS):
                metric_count += 1
            else:
                break
        if metric_count == 0:
            metric_count = 1
        n_param_cols = len(headers) - metric_count

        metric_names = []
        for h in headers[n_param_cols:]:
            clean = re.sub(r'\s*\(.*\)', '', h).strip()
            metric_names.append(clean if clean else h)

        for rd in data_rows:
            # Metric values are always the last metric_count items in the row.
            # This handles cases where header and data column counts differ
            # (e.g. bench_mha: header "function dtype" = 1 col, data "fwd bf16" = 2 cols).
            n_metrics = len(metric_names)
            # pandas/perf_report tables print an unnamed index column, so the
            # data row has one field more than the header and rd[0] must be
            # dropped. Hand-rolled tables (bench_flash_kda, bench_gather_kv_b_proj,
            # bench_fused_kda_decode) have no index column, and dropping rd[0]
            # there silently eats their first shape column -- which can collapse
            # several distinct shapes onto one param key.
            has_index = len(rd) > len(headers)
            param_vals = rd[1 if has_index else 0:len(rd) - n_metrics]
            if len(param_vals) <= 6:
                param_str = " x ".join(param_vals)
            else:
                param_str = " | ".join(param_vals)

            for mi, mname in enumerate(metric_names):
                di = len(rd) - n_metrics + mi
                if di < len(rd):
                    val = safe_float(rd[di])
                    if val is not None:
                        results.append((test_name, param_str, mname, val))

    return results


# ---- Repeat-aware log helpers (median across repeated runs) ----------------

def rep_subdirs(version_dir):
    """Return sorted run<k>/ subdirs of a version log dir, or [] if flat."""
    if not os.path.isdir(version_dir):
        return []
    runs = sorted(
        (d for d in os.listdir(version_dir)
         if re.fullmatch(r'run\d+', d) and os.path.isdir(os.path.join(version_dir, d))),
        key=lambda d: int(d[3:]),
    )
    return [os.path.join(version_dir, d) for d in runs]


def test_log_paths(version_dir, test_name):
    """All existing log paths for a test under a version dir (per-rep or flat)."""
    subs = rep_subdirs(version_dir)
    if subs:
        paths = [os.path.join(s, f"{test_name}.log") for s in subs]
    else:
        paths = [os.path.join(version_dir, f"{test_name}.log")]
    return [p for p in paths if os.path.exists(p)]


def extract_data_aggregated(test_name, version_dir):
    """Extract data for a test across all repeat runs and median each
    (benchmark, params, metric) point. Falls back to a single flat log."""
    from statistics import median
    acc = OrderedDict()
    for p in test_log_paths(version_dir, test_name):
        for bench, params, metric, val in extract_data(test_name, p):
            acc.setdefault((bench, params, metric), []).append(val)
    return [(b, pm, mt, median(vs)) for (b, pm, mt), vs in acc.items()]


def status_of_log(path):
    if not os.path.exists(path):
        return "N/A"
    with open(path) as f:
        content = f.read(4096)
    if content.startswith("TIMEOUT"):
        return "TIMEOUT"
    if "Traceback" in content or "Error" in content or "can't open file" in content:
        return "FAIL"
    return "PASS"


# ============================================================================
# Excel Generation
# ============================================================================

# Thresholds for the 性能变化统计 buckets.
IMPROVE_T = 2.0     # |pct| >= 2%  -> improvement / regression
BIG_T = 10.0        # |pct| >= 10% -> significant
# A change point is "小值波动" (small-value noise) when its baseline magnitude is
# tiny in absolute terms, so a large % swing is not meaningful.
SMALL_VALUE_ABS = {
    True: 1.0,      # throughput-like metrics (TFLOPS / bandwidth): base < 1.0
    False: 0.01,    # latency/time metrics (ms): base < 0.01
}


def write_perf_stats_table(ws, change_records, labels, start_row,
                           header_font, section_font, green_fill, red_fill):
    """Write the 性能变化统计 table grouped by category.

    For throughput metrics higher is better, for latency/time lower is better;
    pct is normalized so positive always means "improvement".
    Columns: 分类 | Benchmark数 | 数据点数 | 提升(≥2%) | 显著提升(≥10%) |
             回退(≤-2%) | 显著回退(≤-10%) | 持平(-2%~2%) | 小值波动 | 提升占比
    """
    new_label, base_label = labels[-1], labels[0]

    title = f"性能变化统计 ({new_label} vs {base_label})"
    ws.cell(start_row, 1, title).font = section_font
    note_row = start_row + 1
    ws.cell(note_row, 1,
            "注：TFLOPS/Bandwidth 越大越好，Time/Latency 越小越好；"
            "提升=正向变化，回退=负向变化；小值波动=基准值过小导致的百分比噪声")

    hdr = ["分类", "Benchmark数", "数据点数", "提升(≥2%)", "显著提升(≥10%)",
           "回退(≤-2%)", "显著回退(≤-10%)", "持平(-2%~2%)", "小值波动", "提升占比"]
    hrow = note_row + 1
    for ci, h in enumerate(hdr, 1):
        ws.cell(hrow, ci, h).font = header_font

    def norm_pct(rec):
        """Improvement-positive percent: latency metrics get sign flipped."""
        return rec["pct"] if rec["is_throughput"] else -rec["pct"]

    def is_small(rec):
        return abs(rec["base"]) < SMALL_VALUE_ABS[rec["is_throughput"]]

    # totals accumulator
    tot = {k: 0 for k in ("bench", "pts", "imp", "bigimp", "reg", "bigreg", "flat", "small")}
    r = hrow + 1
    for cat in CATEGORY_ORDER:
        recs = [x for x in change_records if x["category"] == cat]
        if not recs:
            continue
        benches = len({x["test"] for x in recs})
        pts = len(recs)
        imp = bigimp = reg = bigreg = flat = small = 0
        for rec in recs:
            if is_small(rec):
                small += 1
                continue
            p = norm_pct(rec)
            if p >= BIG_T:
                bigimp += 1; imp += 1
            elif p >= IMPROVE_T:
                imp += 1
            elif p <= -BIG_T:
                bigreg += 1; reg += 1
            elif p <= -IMPROVE_T:
                reg += 1
            else:
                flat += 1
        considered = pts - small
        ratio = f"{round(imp / considered * 100)}%" if considered else "—"

        vals = [CATEGORY_LABELS[cat], benches, pts, imp, bigimp,
                reg, bigreg, flat, small, ratio]
        for ci, v in enumerate(vals, 1):
            ws.cell(r, ci, v)
        if imp:
            ws.cell(r, 4).fill = green_fill
        if reg:
            ws.cell(r, 6).fill = red_fill
        r += 1

        tot["bench"] += benches; tot["pts"] += pts; tot["imp"] += imp
        tot["bigimp"] += bigimp; tot["reg"] += reg; tot["bigreg"] += bigreg
        tot["flat"] += flat; tot["small"] += small

    considered = tot["pts"] - tot["small"]
    ratio = f"{round(tot['imp'] / considered * 100)}%" if considered else "—"
    total_row = ["合计", tot["bench"], tot["pts"], tot["imp"], tot["bigimp"],
                 tot["reg"], tot["bigreg"], tot["flat"], tot["small"], ratio]
    for ci, v in enumerate(total_row, 1):
        c = ws.cell(r, ci, v); c.font = header_font


def generate_excel(labels, log_dirs, all_results, output_path):
    """Generate the multi-sheet comparison Excel.

    labels:      list of version labels, e.g. ["756afc06", "7409f166", "d1660454"]
    log_dirs:    list of log directory paths, one per label
    all_results: list of status dicts, one per label
    output_path: where to save the xlsx
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    green_font = Font(color="006100")
    red_font = Font(color="9C0006")
    header_font = Font(bold=True, size=11)
    section_font = Font(bold=True, size=12)

    base_label = labels[0]  # baseline for change_%

    # Column layout: idx | benchmark | params | metric | label0 | label1 | ... | change_% | change_%(last vs second-to-last) | 备注
    # change_% = last vs first;  second change = last vs second-to-last
    col_headers = ["idx", "benchmark", "params", "metric"]
    col_headers += labels
    if len(labels) >= 2:
        col_headers.append("change_%")
    if len(labels) >= 3:
        col_headers.append(f"change_%({labels[-1]} vs {labels[-2]})")
    col_headers.append("备注")
    n_fixed = 4  # idx, benchmark, params, metric
    n_labels = len(labels)
    col_change1 = n_fixed + n_labels + 1 if len(labels) >= 2 else None
    col_change2 = n_fixed + n_labels + 2 if len(labels) >= 3 else None
    col_note = len(col_headers)

    def is_throughput_metric(metric):
        m = metric.lower()
        return any(kw in m for kw in ["tflops", "throughput", "bandwidth", "tbps", "tb/s", "gb/s", "gbps"])

    def color_change(ws, row, col, pct, metric):
        """Apply green/red coloring based on improvement/regression."""
        cell = ws.cell(row, col)
        if pct is None:
            return
        if is_throughput_metric(metric):
            if pct > 0:
                cell.fill, cell.font = green_fill, green_font
            elif pct < 0:
                cell.fill, cell.font = red_fill, red_font
        else:  # latency/time: lower is better
            if pct < 0:
                cell.fill, cell.font = green_fill, green_font
            elif pct > 0:
                cell.fill, cell.font = red_fill, red_font

    # ---------- Build data per sheet ----------
    sheet_order = CATEGORY_ORDER

    # Compute the starting global index of each category (1-based, cumulative
    # over unique tests in CATEGORY_ORDER) so idx is contiguous across sheets.
    def unique_tests_for(cat):
        seen, uniq = set(), []
        for n, a, c in BENCH_TESTS:
            if c == cat and n not in seen:
                seen.add(n)
                uniq.append((n, a))
        return uniq

    bench_idx_start = {}
    _acc = 1
    for _cat in CATEGORY_ORDER:
        bench_idx_start[_cat] = _acc
        _acc += len(unique_tests_for(_cat))

    # Per-data-point change records for the 性能变化统计 table:
    # list of dicts {category, test, metric, base, new, pct, is_throughput}
    change_records = []

    for sheet_key in sheet_order:
        sheet_name = SHEET_NAMES[sheet_key]
        ws = wb.create_sheet(sheet_name)

        # Write top-level header
        for ci, h in enumerate(col_headers, 1):
            c = ws.cell(1, ci, h)
            c.font = header_font

        row_idx = 2

        unique_tests = unique_tests_for(sheet_key)
        current_idx = bench_idx_start[sheet_key]

        for test_name, _ in unique_tests:
            # Collect data from all versions (median across repeat runs).
            per_wheel_data = []
            for i, label in enumerate(labels):
                data = extract_data_aggregated(test_name, log_dirs[i])
                per_wheel_data.append(data)

            # Build unified key set across all wheels
            # key = (benchmark, params, metric)
            all_keys = OrderedDict()
            for wd in per_wheel_data:
                for bench, params, metric, val in wd:
                    key = (bench, params, metric)
                    if key not in all_keys:
                        all_keys[key] = {}

            # Fill values
            for i, wd in enumerate(per_wheel_data):
                for bench, params, metric, val in wd:
                    all_keys[(bench, params, metric)][labels[i]] = val

            # Determine status/note
            statuses = [all_results[i].get(test_name, "N/A") for i in range(n_labels)]
            note = ""
            for i, st in enumerate(statuses):
                if st not in ("PASS", "N/A"):
                    note += f"{labels[i]}: {st}; "
                # Check log(s) for specific errors (first rep is representative)
                paths = test_log_paths(log_dirs[i], test_name)
                if paths:
                    with open(paths[0]) as f:
                        content = f.read(4096)
                    if "OutOfResources" in content:
                        note += f"{labels[i]}: OutOfResources; "
                    elif "SyntaxError" in content:
                        note += f"{labels[i]}: SyntaxError (Python 3.12?); "
            note = note.rstrip("; ")

            # Write section title row
            ws.cell(row_idx, 1, current_idx).font = section_font
            ws.cell(row_idx, 2, test_name).font = section_font
            current_idx += 1
            row_idx += 1

            # Write sub-header
            for ci, h in enumerate(col_headers, 1):
                ws.cell(row_idx, ci, h).font = header_font
            row_idx += 1

            if not all_keys:
                # No data - write a note row. Zero rows does not imply failure:
                # a bench can exit 0 and still print nothing this parser counts
                # as a metric (bench_rope deliberately emits no runtime at all),
                # so don't label a green test as a failure.
                if note:
                    reason = note
                elif all(st == "PASS" for st in statuses):
                    reason = "PASS but no parsable metric in output"
                else:
                    reason = "FAIL on all versions"
                ws.cell(row_idx, 2, test_name)
                ws.cell(row_idx, 3, "NO DATA")
                ws.cell(row_idx, col_note, reason)
                row_idx += 1
            else:
                for (bench, params, metric), vals in all_keys.items():
                    ws.cell(row_idx, 2, bench)
                    ws.cell(row_idx, 3, params)
                    ws.cell(row_idx, 4, metric)

                    # Write values for each wheel
                    for i, label in enumerate(labels):
                        v = vals.get(label)
                        if v is not None:
                            ws.cell(row_idx, n_fixed + i + 1, round(v, 6))

                    # change_%: last vs first  (new vs base)
                    if col_change1:
                        v_first = vals.get(labels[0])
                        v_last = vals.get(labels[-1])
                        if v_first is not None and v_last is not None and abs(v_first) > 1e-12:
                            pct = round(((v_last - v_first) / abs(v_first)) * 100, 2)
                            ws.cell(row_idx, col_change1, pct)
                            color_change(ws, row_idx, col_change1, pct, metric)
                            change_records.append({
                                "category": sheet_key,
                                "test": test_name,
                                "metric": metric,
                                "base": v_first,
                                "new": v_last,
                                "pct": pct,
                                "is_throughput": is_throughput_metric(metric),
                            })

                    # change_%: last vs second-to-last
                    if col_change2 and n_labels >= 3:
                        v_prev = vals.get(labels[-2])
                        v_last = vals.get(labels[-1])
                        if v_prev is not None and v_last is not None and abs(v_prev) > 1e-12:
                            pct2 = round(((v_last - v_prev) / abs(v_prev)) * 100, 2)
                            ws.cell(row_idx, col_change2, pct2)
                            color_change(ws, row_idx, col_change2, pct2, metric)

                    # Note (only on first data row of this test)
                    row_idx += 1

            # Add note on first data row of this test section (after header)
            if note:
                first_data_row = row_idx - len(all_keys) if all_keys else row_idx - 1
                ws.cell(first_data_row, col_note, note)

            row_idx += 1  # blank row between tests

    # ---------- Summary Sheet ----------
    ws_sum = wb.create_sheet("Summary")
    sum_headers = ["Test Name"] + [f"Status ({l})" for l in labels] + ["Note"]
    for ci, h in enumerate(sum_headers, 1):
        ws_sum.cell(1, ci, h).font = header_font

    all_tests_dedup = get_unique_tests(BENCH_TESTS)
    for tidx, (test_name, _) in enumerate(all_tests_dedup, 2):
        ws_sum.cell(tidx, 1, test_name)
        for i, label in enumerate(labels):
            st = all_results[i].get(test_name, "N/A")
            # Refine status from log(s)
            paths = test_log_paths(log_dirs[i], test_name)
            if st == "PASS" and paths:
                with open(paths[0]) as f:
                    content = f.read(4096)
                if "OutOfResources" in content:
                    st = "FAIL (OutOfResources)"
                elif "SyntaxError" in content:
                    st = "FAIL (SyntaxError)"
            cell = ws_sum.cell(tidx, 2 + i, st)
            if "PASS" in st:
                cell.fill = green_fill
            elif "FAIL" in st or "TIMEOUT" in st:
                cell.fill = red_fill

    # ---------- 性能变化统计 table (only when comparing >= 2 versions) ----------
    if len(labels) >= 2:
        write_perf_stats_table(
            ws_sum, change_records, labels,
            start_row=len(all_tests_dedup) + 4,
            header_font=header_font, section_font=section_font,
            green_fill=green_fill, red_fill=red_fill,
        )

    # Auto-fit column widths (approximate)
    for ws_sheet in wb.worksheets:
        for col_cells in ws_sheet.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                try:
                    if cell.value:
                        max_len = max(max_len, len(str(cell.value)))
                except Exception:
                    pass
            ws_sheet.column_dimensions[col_letter].width = min(max_len + 2, 60)

    wb.save(output_path)
    print(f"\nExcel saved to {output_path}")


# ============================================================================
# Status inference from logs (for --skip-run mode)
# ============================================================================

def infer_status_from_logs(log_dir, tests):
    """Infer PASS/FAIL/TIMEOUT from existing logs (rep-aware: aggregates across
    run<k>/ subdirs, PASS wins if any rep passed)."""
    results = {}
    for name, _ in tests:
        paths = test_log_paths(log_dir, name)
        if not paths:
            results[name] = "N/A"
            continue
        sts = [status_of_log(p) for p in paths]
        if "PASS" in sts:
            results[name] = "PASS"
        elif all(s == "TIMEOUT" for s in sts):
            results[name] = "TIMEOUT"
        else:
            results[name] = "FAIL"
    return results


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all triton bench tests and generate comparison Excel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wheels", nargs="+", default=None,
        help="Paths to triton wheel files (2 or more). First = baseline. "
             "Omit when using --from-index.",
    )
    parser.add_argument(
        "--from-index", action="store_true",
        help="Compare two triton builds pulled from the AMD PyPI indices "
             "(release_ as baseline, release_tmp as new), auto-detecting the "
             "ROCm version. Each is installed, all benches run, and a comparison "
             "Excel with change_% and the 性能变化统计 table is generated.",
    )
    parser.add_argument(
        "--output", default="bench_test_results.xlsx",
        help="Output Excel file path (default: bench_test_results.xlsx).",
    )
    parser.add_argument(
        "--timeout", type=int, default=1200,
        help="Timeout per test in seconds (default: 1200 = 20min).",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip running tests; regenerate Excel from existing logs.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Run each test N times per version and median each data point to "
             "stabilize the comparison (default: 1). Logs go to "
             "bench_logs/rerun_<hash>/run<k>/.",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Base log directory (default: bench_logs/ under project root).",
    )
    parser.add_argument(
        "--tests", nargs="+", default=None,
        help="Run only these specific tests (by name, e.g. bench_mha).",
    )
    parser.add_argument(
        "--sheet", choices=CATEGORY_ORDER,
        default=None,
        help="Run only tests from this category.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.from_index and not args.wheels:
        sys.exit("error: provide --wheels, or use --from-index for a single index install.")

    # Filter tests if requested
    tests = list(BENCH_TESTS)
    if args.sheet:
        tests = [(n, a, c) for n, a, c in tests if c == args.sheet]
    if args.tests:
        test_set = set(args.tests)
        tests = [(n, a, c) for n, a, c in tests if n in test_set]

    unique_tests = get_unique_tests(tests)

    # In index mode we have a single (yet unknown) version; install first to get its
    # label, unless --skip-run, in which case fall back to a fixed "index" label.
    index_mode = args.from_index
    base_log_dir = args.log_dir or str(PROJECT_ROOT / "bench_logs")
    all_results = []

    if index_mode:
        mm = detect_rocm_major_minor()
        if not mm:
            sys.exit("error: could not detect ROCm version from rocm-core (dpkg -l rocm-core)")
        print(f"Compare:  AMD PyPI indices {INDEX_VARIANTS} (rocm-{mm}.0)")
        print(f"Tests:    {len(unique_tests)}")
        print(f"Repeat:   {args.repeat}x (median)")
        print(f"Timeout:  {args.timeout}s")
        print(f"Output:   {args.output}")

        if args.skip_run:
            print("\n--skip-run: regenerating Excel from existing logs")
            labels = [fetch_index_label(v, mm) for v in INDEX_VARIANTS]
            log_dirs = [os.path.join(base_log_dir, f"rerun_{l}") for l in labels]
            for i, label in enumerate(labels):
                results = infer_status_from_logs(log_dirs[i], unique_tests)
                all_results.append(results)
                print(f"  {INDEX_VARIANTS[i]} ({label}): "
                      f"{sum(1 for v in results.values() if v=='PASS')} PASS")
        else:
            labels, log_dirs = [], []
            for variant in INDEX_VARIANTS:
                label = install_from_index(variant, mm)
                log_dir = os.path.join(base_log_dir, f"rerun_{label}")
                labels.append(label)
                log_dirs.append(log_dir)
                print(f"\nRunning {len(unique_tests)} tests x{args.repeat} with "
                      f"{variant} ({label}):")
                results = run_tests_repeated(unique_tests, log_dir, args.timeout, args.repeat)
                all_results.append(results)
    else:
        labels = [get_version_label(w) for w in args.wheels]
        log_dirs = [os.path.join(base_log_dir, f"rerun_{l}") for l in labels]

        print(f"Wheels:  {len(args.wheels)}")
        for i, w in enumerate(args.wheels):
            print(f"  [{i+1}] {os.path.basename(w)}  ->  label={labels[i]}")
        print(f"Tests:   {len(unique_tests)}")
        print(f"Timeout: {args.timeout}s")
        print(f"Output:  {args.output}")
        print(f"Logs:    {base_log_dir}/rerun_<label>/")

        if args.skip_run:
            print("\n--skip-run: regenerating Excel from existing logs")
            for i, label in enumerate(labels):
                results = infer_status_from_logs(log_dirs[i], unique_tests)
                all_results.append(results)
                p = sum(1 for v in results.values() if v == "PASS")
                f = sum(1 for v in results.values() if v in ("FAIL",))
                t = sum(1 for v in results.values() if v == "TIMEOUT")
                print(f"  {label}: {p} PASS, {f} FAIL, {t} TIMEOUT")
        else:
            for i, wheel in enumerate(args.wheels):
                install_wheel(wheel)
                print(f"\nRunning {len(unique_tests)} tests x{args.repeat} with {labels[i]}:")
                results = run_tests_repeated(unique_tests, log_dirs[i], args.timeout, args.repeat)
                all_results.append(results)

    # Always regenerate the full Excel (all sheets) to keep it coherent
    generate_excel(labels, log_dirs, all_results, args.output)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for i, label in enumerate(labels):
        r = all_results[i]
        p = sum(1 for v in r.values() if v == "PASS")
        f = sum(1 for v in r.values() if v in ("FAIL",))
        t = sum(1 for v in r.values() if v == "TIMEOUT")
        print(f"  {label}: {p} PASS, {f} FAIL, {t} TIMEOUT")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

