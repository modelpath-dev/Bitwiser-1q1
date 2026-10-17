"""Resource-aware GPU-accelerated inference via llama-completion."""
from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import psutil

from prompts import format_prompt

ROOT = Path(__file__).resolve().parent
LLAMA_BIN = ROOT / "vendor" / "llama.cpp" / "build" / "bin"
# llama-completion is the non-interactive single-prompt binary. llama-cli is
# always interactive when the model ships a chat template (Mistral, Llama-3,
# Gemma) — its `-no-cnv` flag is documented but errors at runtime
# ("--no-conversation is not supported by llama-cli; please use llama-completion
# instead"). llama-completion runs the prompt and exits cleanly.
CLI_BIN = LLAMA_BIN / "llama-completion"

# llama-completion appends this marker to stdout when generation ends. We
# buffer it out of the user-visible stream.
EOF_MARKER = "> EOF by user"

# llama.cpp's perf summary on stderr, e.g.
#   llama_perf_context_print:        eval time =  812.30 ms /   127 runs   ( ... )
# "runs" (not "tokens") distinguishes it from the "prompt eval time" line.
EVAL_PERF_RE = re.compile(r" eval time =\s*([\d.]+) ms /\s*(\d+) runs")

# Metal shares system RAM, so every layer fits on the GPU. A discrete GPU has
# its own (often small) VRAM: forcing all layers there fails to load anything
# bigger than VRAM, so elsewhere we leave llama.cpp's default `-ngl auto` with
# `--fit on`, which offloads as many layers as fit.
GPU_LAYERS = "99" if sys.platform == "darwin" else "auto"
GPU_LAYER_ARGS = ["-ngl", GPU_LAYERS]

# `--list-devices` prints e.g. "  CUDA0: NVIDIA GeForce GTX 1650 (4096 MiB, ...)".
_DEVICE_RE = re.compile(r"^\s*([A-Za-z]+)\d+:\s*(.+?)\s*\(")
_BACKEND_NAMES = {"MTL": "Metal", "CUDA": "CUDA", "Vulkan": "Vulkan"}


@functools.cache
def compute_backend() -> str:
    """Accelerator llama.cpp was built for, as shown in the UI: "CUDA · <gpu>",
    "Metal · <chip>", or "CPU" when no GPU device is available."""
    if not CLI_BIN.exists():
        return "llama.cpp not built"
    try:
        res = subprocess.run(
            [str(CLI_BIN), "--list-devices"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    for line in (res.stdout + res.stderr).splitlines():
        m = _DEVICE_RE.match(line)
        if m:
            return f"{_BACKEND_NAMES.get(m.group(1), m.group(1))} · {m.group(2)}"
    return "CPU"


@dataclass
class RunStats:
    tokens: int | None  # generated tokens per llama.cpp; None if it never printed perf
    elapsed_s: float
    peak_rss_bytes: int
    eval_ms: float | None = None

    @property
    def tokens_per_s(self) -> float | None:
        """Decode speed as llama.cpp measures it (excludes model load and prompt eval)."""
        if not self.tokens or not self.eval_ms:
            return None
        return self.tokens / (self.eval_ms / 1000.0)


def _physical_cores() -> int:
    n = psutil.cpu_count(logical=False)
    return n if n and n > 0 else 4


def optimal_thread_count() -> int:
    """Use all performance cores. On Apple Silicon, psutil reports physical cores;
    we use all of them (efficiency cores are mostly counterproductive for inference)."""
    cores = _physical_cores()
    # Heuristic: on M-series, P-cores ~= total physical - 4 efficiency cores for older
    # chips, but newer chips (M4/M5) flip the ratio. Use cores - 2 with a floor of 4.
    return max(4, cores - 2)


def model_will_fit(path: str | Path) -> tuple[bool, int, int]:
    """Return (fits, model_bytes, ram_bytes). Treats <0.85x available RAM as fitting."""
    size = Path(path).stat().st_size
    avail = psutil.virtual_memory().available
    return size < int(avail * 0.85), size, avail


def build_command(
    model_path: str,
    prompt: str,
    threads: int | None = None,
    n_predict: int = 256,
    ctx: int = 4096,
    stop_tokens: list[str] | None = None,
) -> list[str]:
    if not CLI_BIN.exists():
        raise FileNotFoundError(f"llama-completion not found at {CLI_BIN}. Run ./setup.sh first.")
    cmd: list[str] = [
        str(CLI_BIN),
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx),
        "-t", str(threads or optimal_thread_count()),
        *GPU_LAYER_ARGS,
        "--no-display-prompt",
        "--simple-io",
    ]
    if stop_tokens:
        for s in stop_tokens:
            cmd.extend(["-r", s])
    return cmd


def _kill_proc(proc: subprocess.Popen) -> None:
    """Best-effort terminate, then SIGKILL after 2s. Always reaps."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        pass


def run_streaming(
    model_path: str,
    model_name: str,
    architecture: str | None,
    user_prompt: str,
    n_predict: int = 256,
    ctx: int = 4096,
    timeout_s: float = 120.0,
    runaway_blank_chars: int = 400,
    proc_sink: list | None = None,
) -> Iterator[tuple[str, RunStats | None]]:
    """Yield (token_chunk, None) as text streams in. Final yield is ('', RunStats).

    The Popen handle is appended to `proc_sink` (if provided) so the caller can
    kill it if the consumer (SSE client) disconnects. Also enforces:
      - hard wall-clock timeout (`timeout_s`)
      - runaway-whitespace guard (kills if >`runaway_blank_chars` in a row)
      - generator-level cleanup: `finally` block always reaps the subprocess."""
    fits, size, avail = model_will_fit(model_path)
    if not fits:
        # Weights are always mmap'd (llama.cpp's default), so an oversized
        # model still runs; the kernel pages it from disk.
        yield (
            f"[warning] model is {size/1e9:.1f} GB but only {avail/1e9:.1f} GB RAM is "
            f"free; weights will page from disk (will be slow).\n",
            None,
        )

    formatted, stop = format_prompt(model_name, architecture, user_prompt)
    cmd = build_command(
        model_path,
        formatted,
        n_predict=n_predict,
        ctx=ctx,
        stop_tokens=stop,
    )

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if proc_sink is not None:
        proc_sink.append(proc)
    assert proc.stdout is not None and proc.stderr is not None

    # stderr must be drained continuously or llama.cpp blocks on a full pipe.
    # We only keep the perf summary line it prints on exit.
    eval_perf: list[tuple[float, int]] = []

    def _drain_stderr() -> None:
        for line in proc.stderr:
            m = EVAL_PERF_RE.search(line)
            if m:
                eval_perf.append((float(m.group(1)), int(m.group(2))))

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # stdout.read(1) blocks while the model loads or stalls, so the wall-clock
    # limit is enforced from a timer; killing the process unblocks the read.
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        _kill_proc(proc)

    timer = threading.Timer(timeout_s, _on_timeout)
    timer.daemon = True
    timer.start()

    peak_rss = 0
    chars = 0
    consecutive_blank = 0
    try:
        ps = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        ps = None

    # Tail buffer: hold back the last `tail_len` chars from the stream so we
    # can drop the trailing "> EOF by user" marker after the process exits.
    tail_len = len(EOF_MARKER) + 4  # margin for surrounding newlines/spaces
    tail: list[str] = []

    def _flush_oldest() -> str | None:
        if len(tail) > tail_len:
            return tail.pop(0)
        return None

    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            chars += 1

            if ch.strip():
                consecutive_blank = 0
            else:
                consecutive_blank += 1
                if consecutive_blank >= runaway_blank_chars:
                    _kill_proc(proc)
                    yield (
                        f"\n[runaway] killed after {runaway_blank_chars} "
                        f"consecutive whitespace chars (degraded quantization)\n",
                        None,
                    )
                    break

            tail.append(ch)
            old = _flush_oldest()
            if old is not None:
                yield (old, None)

            if ps is not None and chars % 16 == 0:
                try:
                    rss = ps.memory_info().rss
                    if rss > peak_rss:
                        peak_rss = rss
                except psutil.NoSuchProcess:
                    pass

        # Process exited: decide whether the held-back tail is the EOF marker
        # (and should be dropped) or genuine output (and should be flushed).
        trailing = "".join(tail)
        if EOF_MARKER in trailing:
            keep = trailing.split(EOF_MARKER, 1)[0].rstrip()
            if keep:
                yield (keep, None)
        else:
            if trailing:
                yield (trailing, None)
        if timed_out.is_set():
            yield (f"\n[timeout] killed after {timeout_s:.0f}s\n", None)
    finally:
        timer.cancel()
        # If the consumer abandoned us (SSE client disconnected, etc.) this
        # finally runs and reaps the subprocess so the model unloads from RAM.
        _kill_proc(proc)

    elapsed = time.perf_counter() - start
    stderr_thread.join(timeout=2.0)
    eval_ms, tokens = eval_perf[-1] if eval_perf else (None, None)
    yield ("", RunStats(tokens=tokens, elapsed_s=elapsed, peak_rss_bytes=peak_rss, eval_ms=eval_ms))


def run_blocking(
    model_path: str,
    model_name: str,
    architecture: str | None,
    user_prompt: str,
    n_predict: int = 256,
) -> tuple[str, RunStats]:
    text_parts: list[str] = []
    stats: RunStats | None = None
    for chunk, s in run_streaming(model_path, model_name, architecture, user_prompt, n_predict):
        if s is not None:
            stats = s
        else:
            text_parts.append(chunk)
    assert stats is not None
    return "".join(text_parts), stats
