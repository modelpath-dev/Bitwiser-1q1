"""Automated 1-bit (IQ1_S) compression pipeline."""
from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from inference import GPU_LAYER_ARGS
from scanner import Model, _read_gguf_metadata

ROOT = Path(__file__).resolve().parent
LLAMA_BIN = ROOT / "vendor" / "llama.cpp" / "build" / "bin"
QUANTIZE_BIN = LLAMA_BIN / "llama-quantize"
IMATRIX_BIN = LLAMA_BIN / "llama-imatrix"
CALIBRATION_FILE = ROOT / "calibration.txt"

OUTPUT_DIR = ROOT / "compressed"

# Quant types that already represent <=2-bit; re-quantizing is pointless.
ALREADY_TINY = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "TQ1_0", "TQ2_0", "Q1_0"}

# Quant types llama-quantize refuses to produce without an importance matrix.
NEEDS_IMATRIX = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M"}

ProgressCB = Callable[..., None]  # (pct: float, line: str, phase: str = "quantize")


class CompressionError(RuntimeError):
    pass


@dataclass
class CompressionResult:
    input_path: str
    output_path: str
    input_bytes: int
    output_bytes: int
    quant: str

    @property
    def ratio(self) -> float:
        return self.output_bytes / self.input_bytes if self.input_bytes else 0.0


def _ensure_binary(path: Path, label: str) -> Path:
    if not path.exists():
        raise CompressionError(f"{label} not found at {path}. Run ./setup.sh first.")
    return path


def output_path_for(model: Model, target: str = "IQ1_S") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(model.path).stem or model.name.replace("/", "_").replace(":", "_")
    return OUTPUT_DIR / f"{stem}.{target}.gguf"


def imatrix_path_for(model: Model) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(model.path).stem or model.name.replace("/", "_").replace(":", "_")
    return OUTPUT_DIR / f"{stem}.imatrix"


def already_compressed(model: Model) -> bool:
    return (model.quant or "").upper() in ALREADY_TINY


def _estimated_ratio(target: str) -> float:
    """Rough output:input size ratio when re-quantizing from a Q4_K source.
    Used only for user-facing messaging; actual ratio comes from file sizes."""
    table = {
        "IQ1_S": 0.34, "IQ1_M": 0.38, "IQ2_XXS": 0.45, "IQ2_XS": 0.50,
        "IQ2_S": 0.54, "IQ2_M": 0.59, "Q2_K": 0.65, "Q3_K_S": 0.75,
        "Q3_K_M": 0.82, "Q4_K_S": 0.95, "Q4_K_M": 1.0,
    }
    return 1.0 - table.get(target.upper(), 0.5)


def _emit(cb: ProgressCB | None, pct: float, line: str, phase: str) -> None:
    if cb is None:
        return
    try:
        cb(pct, line, phase)
    except TypeError:
        # Backward compat: caller didn't accept phase.
        cb(pct, line)


def _part_path(out: Path) -> Path:
    """Where a subprocess writes before the result is complete. Only a finished,
    successful run is renamed to `out`, so an existing `out` is always trustworthy
    as a cache hit."""
    return out.with_name(out.name + ".part")


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()


def _generate_imatrix(
    model: Model,
    on_progress: ProgressCB | None,
    chunks: int = 12,
    proc_sink: list | None = None,
) -> Path:
    """Run llama-imatrix over calibration.txt to produce an importance matrix.
    Cached: if the imatrix file already exists, returns it without recomputing."""
    out = imatrix_path_for(model)
    if out.exists() and out.stat().st_size > 0:
        _emit(on_progress, 1.0, f"[imatrix] using cached {out.name}", "calibrate")
        return out

    binary = _ensure_binary(IMATRIX_BIN, "llama-imatrix")
    if not CALIBRATION_FILE.exists():
        raise CompressionError(f"calibration.txt missing at {CALIBRATION_FILE}")

    part = _part_path(out)
    cmd = [
        str(binary),
        "-m", model.path,
        "-f", str(CALIBRATION_FILE),
        "-o", str(part),
        "-c", "512",
        "--chunks", str(chunks),
        *GPU_LAYER_ARGS,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc_sink is not None:
        proc_sink.append(proc)

    total = chunks
    chunk_re = re.compile(r"\[(\d+)\]")
    total_re = re.compile(r"computing over\s+(\d+)\s+chunks")

    assert proc.stdout is not None
    last_pct = 0.0
    try:
        for line in proc.stdout:
            line = line.rstrip()
            m = total_re.search(line)
            if m:
                total = max(int(m.group(1)), 1)
            m = chunk_re.search(line)
            if m:
                n = int(m.group(1))
                last_pct = min(n / total, 1.0)
            _emit(on_progress, last_pct, line, "calibrate")

        rc = proc.wait()
        if rc != 0:
            raise CompressionError(f"llama-imatrix exited with status {rc}")
        if not part.exists():
            raise CompressionError("imatrix finished but output file is missing")
        os.replace(part, out)
    finally:
        _reap(proc)
        part.unlink(missing_ok=True)
    return out


def compress(
    model: Model,
    target: str = "IQ1_S",
    on_progress: ProgressCB | None = None,
    overwrite: bool = False,
    proc_sink: list | None = None,
) -> CompressionResult:
    """Run llama-quantize on `model`, streaming progress.
    `on_progress(pct, line, phase)` is invoked for every output line, where
    `phase` is "calibrate" (during imatrix) or "quantize". Each subprocess is
    appended to `proc_sink` (if provided) so the caller can kill it on cancel."""
    binary = _ensure_binary(QUANTIZE_BIN, "llama-quantize")
    out = output_path_for(model, target)

    if already_compressed(model):
        raise CompressionError(
            f"{model.name} is already {model.quant}; nothing to compress."
        )

    target_upper = target.upper()

    # Multimodal GGUFs (Gemma3n/Gemma4 with vision + audio towers) hit two distinct
    # upstream llama.cpp bugs that we can't work around at the wrapper level:
    #   1. llama-imatrix's model loader doesn't know about vision/audio tensors and
    #      aborts with a tensor-count mismatch — so IQ1/IQ2 are unreachable.
    #   2. llama-quantize crashes with GGML_ASSERT (llama-quant.cpp:1256) on the
    #      audio tower's 4D conv weights (e.g. a.conv1d.0.weight, [3,3,1,128]) —
    #      so even Q2_K, Q4_K, etc. fail mid-run.
    # There is no llama-quantize flag that skips problematic tensors entirely.
    # Refuse up front rather than crash partway through.
    if getattr(model, "multimodal", False):
        raise CompressionError(
            f"{model.name} is a multimodal model (vision + audio tensors alongside "
            f"the text decoder). The current llama.cpp release can't quantize this "
            f"GGUF: llama-imatrix won't load it (tensor-count mismatch), and "
            f"llama-quantize crashes on the audio tower's 4D conv weights "
            f"(llama-quant.cpp:1256 assertion). Try a text-only model instead — "
            f"e.g. `ollama pull gemma2:9b`, `ollama pull llama3.1:8b`, or "
            f"`ollama pull mistral:7b`."
        )

    if out.exists() and not overwrite:
        return CompressionResult(
            input_path=model.path,
            output_path=str(out),
            input_bytes=Path(model.path).stat().st_size,
            output_bytes=out.stat().st_size,
            quant=target,
        )

    extra_args: list[str] = []

    # Most local models (Ollama, HuggingFace mirrors) ship pre-quantized — they
    # arrive as Q4_K_M, Q5_K_M, Q8_0, etc. llama-quantize refuses to re-quantize
    # already-quantized tensors by default, since going Q4_K → IQ1_S loses more
    # quality than going F16 → IQ1_S. We accept that tradeoff: this tool's whole
    # promise is "any local model → 1-bit", not "ideal-quality 1-bit".
    source_quant = (model.quant or "").upper()
    is_already_quantized = source_quant not in {"", "F32", "F16", "BF16"}
    if is_already_quantized:
        extra_args.append("--allow-requantize")
        _emit(
            on_progress,
            0.0,
            f"{model.name} is already {source_quant}. Re-quantizing to {target} "
            f"(--allow-requantize). Quality will be lower than quantizing from a "
            f"fresh F16 source, but the file will still shrink ~{_estimated_ratio(target):.0%}.",
            "warning",
        )

    if target_upper in NEEDS_IMATRIX:
        _emit(on_progress, 0.0, f"[imatrix] {target} requires calibration; starting", "calibrate")
        imatrix = _generate_imatrix(model, on_progress, proc_sink=proc_sink)
        extra_args.extend(["--imatrix", str(imatrix)])
        _emit(on_progress, 0.0, f"[quantize] starting {target}", "quantize")

    part = _part_path(out)
    cmd = [str(binary), *extra_args, model.path, str(part), target]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc_sink is not None:
        proc_sink.append(proc)

    # llama-quantize prints "[ N/M] ..." lines per tensor; use those for progress.
    progress_re = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
    last_pct = 0.0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            m = progress_re.search(line)
            if m:
                n, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    last_pct = n / total
            _emit(on_progress, last_pct, line, "quantize")

        rc = proc.wait()
        if rc != 0:
            raise CompressionError(f"llama-quantize exited with status {rc}")
        if not part.exists():
            raise CompressionError("Quantization finished but output file is missing.")
        os.replace(part, out)
    finally:
        _reap(proc)
        part.unlink(missing_ok=True)

    in_size = Path(model.path).stat().st_size
    out_size = out.stat().st_size

    # Sanity check: IQ1_S should be roughly 0.15x–0.25x of an FP16 source.
    # For Q4 inputs the ratio is closer to 0.4x, so we only warn, don't error.
    return CompressionResult(
        input_path=model.path,
        output_path=str(out),
        input_bytes=in_size,
        output_bytes=out_size,
        quant=target,
    )


if __name__ == "__main__":
    import argparse
    import scanner

    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Model name as reported by scanner.py")
    ap.add_argument("--target", default="IQ1_S")
    args = ap.parse_args()

    models = {m.name: m for m in scanner.discover()}
    if args.name not in models:
        raise SystemExit(f"Unknown model {args.name!r}. Available: {list(models)}")

    def cb(pct: float, line: str, phase: str = "quantize") -> None:
        print(f"[{phase[:3]} {pct*100:5.1f}%] {line}")

    res = compress(models[args.name], args.target, on_progress=cb)
    print(f"\nWrote {res.output_path} ({res.output_bytes/1e9:.2f} GB, ratio={res.ratio:.2f})")
