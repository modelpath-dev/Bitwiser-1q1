"""Model discovery for Ollama blobs and local GGUF files."""
from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

OLLAMA_MEDIA_MODEL = "application/vnd.ollama.image.model"

GGUF_MAGIC = b"GGUF"


@dataclass
class Model:
    name: str
    source: str  # "ollama" or "gguf"
    path: str  # absolute path to GGUF (or ollama blob, which is GGUF)
    size_bytes: int
    family: str | None = None  # llama, gemma, mistral, phi, qwen, ...
    quant: str | None = None  # F16, Q4_K_M, IQ1_S, ...
    architecture: str | None = None
    multimodal: bool = False  # has vision/audio tensors (e.g. Gemma3n/Gemma4)

    def to_dict(self) -> dict:
        return asdict(self)


def _read_gguf_metadata(path: Path) -> dict:
    """Return a small subset of GGUF metadata: architecture, quant, name."""
    out: dict = {}
    try:
        with path.open("rb") as f:
            head = f.read(4)
            if head != GGUF_MAGIC:
                return out
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return out
            f.read(8)  # tensor count
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", errors="replace")

            # GGUF value type ids
            type_readers = {
                0: lambda: struct.unpack("<B", f.read(1))[0],   # uint8
                1: lambda: struct.unpack("<b", f.read(1))[0],   # int8
                2: lambda: struct.unpack("<H", f.read(2))[0],
                3: lambda: struct.unpack("<h", f.read(2))[0],
                4: lambda: struct.unpack("<I", f.read(4))[0],
                5: lambda: struct.unpack("<i", f.read(4))[0],
                6: lambda: struct.unpack("<f", f.read(4))[0],
                7: lambda: struct.unpack("<B", f.read(1))[0],   # bool
                8: read_str,
                10: lambda: struct.unpack("<Q", f.read(8))[0],
                11: lambda: struct.unpack("<q", f.read(8))[0],
                12: lambda: struct.unpack("<d", f.read(8))[0],
            }

            def read_value(t: int):
                if t == 9:  # array
                    inner = struct.unpack("<I", f.read(4))[0]
                    n = struct.unpack("<Q", f.read(8))[0]
                    # We don't need array contents — skip them quickly.
                    for _ in range(n):
                        read_value(inner)
                    return None
                reader = type_readers.get(t)
                if reader is None:
                    raise ValueError(f"Unknown GGUF type {t}")
                return reader()

            wanted = {
                "general.architecture",
                "general.name",
                "general.file_type",
                "general.quantization_version",
            }
            multimodal = False
            for _ in range(kv_count):
                key = read_str()
                t = struct.unpack("<I", f.read(4))[0]
                val = read_value(t)
                if not multimodal:
                    parts = key.split(".")
                    if "vision" in parts or "audio" in parts:
                        multimodal = True
                if key in wanted:
                    out[key] = val
            out["multimodal"] = multimodal
    except (OSError, ValueError, struct.error):
        return out

    # Map LLAMA_FTYPE enum (GGUF general.file_type) -> readable quant name.
    file_type_map = {
        0: "F32", 1: "F16",
        2: "Q4_0", 3: "Q4_1",
        7: "Q8_0",
        8: "Q5_0", 9: "Q5_1",
        10: "Q2_K",
        11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
        14: "Q4_K_S", 15: "Q4_K_M",
        16: "Q5_K_S", 17: "Q5_K_M",
        18: "Q6_K",
        19: "IQ2_XXS", 20: "IQ2_XS",
        21: "Q2_K_S",
        22: "IQ3_XS", 23: "IQ3_XXS",
        24: "IQ1_S",
        25: "IQ4_NL",
        26: "IQ3_S", 27: "IQ3_M",
        28: "IQ2_S", 29: "IQ2_M",
        30: "IQ4_XS",
        31: "IQ1_M",
    }
    if "general.file_type" in out:
        out["quant"] = file_type_map.get(int(out["general.file_type"]), f"FT{out['general.file_type']}")
    return out


def _build_model(name: str, source: str, path: Path) -> Model:
    meta = _read_gguf_metadata(path)
    return Model(
        name=name,
        source=source,
        path=str(path),
        size_bytes=path.stat().st_size,
        family=meta.get("general.architecture"),
        quant=meta.get("quant"),
        architecture=meta.get("general.architecture"),
        multimodal=bool(meta.get("multimodal")),
    )


def _ollama_roots() -> list[Path]:
    """Ollama model stores: $OLLAMA_MODELS if set, the per-user store (macOS,
    `ollama serve`), and the system store used by the Linux installer's service."""
    roots = [Path.home() / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models")]
    if os.environ.get("OLLAMA_MODELS"):
        roots.insert(0, Path(os.environ["OLLAMA_MODELS"]))
    return roots


def scan_ollama() -> list[Model]:
    out: list[Model] = []
    seen: set[Path] = set()
    for root in _ollama_roots():
        # $OLLAMA_MODELS often points at one of the default stores.
        real = root.resolve()
        if real in seen:
            continue
        seen.add(real)
        out.extend(_scan_ollama_root(root))
    return out


def _scan_ollama_root(root: Path) -> list[Model]:
    out: list[Model] = []
    manifests_root = root / "manifests"
    blobs_root = root / "blobs"
    if not manifests_root.exists() or not blobs_root.exists():
        return out

    for manifest_path in manifests_root.rglob("*"):
        if not manifest_path.is_file():
            continue
        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        for layer in data.get("layers", []):
            if layer.get("mediaType") != OLLAMA_MEDIA_MODEL:
                continue
            digest = layer.get("digest", "").replace(":", "-")
            blob = blobs_root / digest
            if not blob.exists():
                continue
            # name = registry/owner/repo:tag
            rel = manifest_path.relative_to(manifests_root)
            parts = rel.parts
            tag = parts[-1]
            repo = "/".join(parts[1:-1]) if len(parts) > 2 else parts[-2]
            name = f"{repo}:{tag}"
            out.append(_build_model(name, "ollama", blob))
    return out


def scan_local(roots: list[Path]) -> list[Model]:
    out: list[Model] = []
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue
        for path in root.rglob("*.gguf"):
            real = str(path.resolve())
            if real in seen:
                continue
            seen.add(real)
            out.append(_build_model(path.name, "gguf", path))
    return out


def discover(extra_dirs: list[Path] | None = None) -> list[Model]:
    """Return all compressible models found on this machine."""
    roots: list[Path] = [Path.home() / "Downloads", Path.home() / "Models"]
    if extra_dirs:
        roots.extend(extra_dirs)
    return scan_ollama() + scan_local(roots)


if __name__ == "__main__":
    models = discover()
    print(json.dumps([m.to_dict() for m in models], indent=2))
