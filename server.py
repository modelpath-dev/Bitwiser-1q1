"""BitWiser HTTP API. Wraps the local Python modules for the React frontend."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import compressor
import inference
import scanner

ROOT = Path(__file__).resolve().parent
WEB_DIST = ROOT / "web" / "dist"
COMPRESSED_DIR = ROOT / "compressed"


app = FastAPI(title="BitWiser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_index() -> dict[str, scanner.Model]:
    return {m.name: m for m in scanner.discover()}


@app.get("/api/models")
def list_models() -> dict:
    models = scanner.discover()
    return {
        "models": [m.to_dict() for m in models],
        "system": {
            "physical_cores": inference._physical_cores(),
            "optimal_threads": inference.optimal_thread_count(),
            "backend": inference.compute_backend(),
            "gpu_layers": inference.GPU_LAYERS,
        },
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------- compression ----------
class CompressRequest(BaseModel):
    name: str
    target: str = "IQ1_S"


def _stream_compress(model: scanner.Model, target: str):
    """Run llama-quantize in a thread, push events into a queue, drain to SSE."""
    q: queue.Queue = queue.Queue()
    DONE = object()
    proc_sink: list = []  # populated by compressor.compress as each subprocess starts

    def worker() -> None:
        def cb(pct: float, line: str, phase: str = "quantize") -> None:
            if phase == "warning":
                q.put({"type": "warning", "message": line})
            else:
                q.put({"type": "progress", "pct": pct, "line": line, "phase": phase})

        try:
            res = compressor.compress(model, target=target, on_progress=cb, proc_sink=proc_sink)
        except compressor.CompressionError as e:
            q.put({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        else:
            q.put(
                {
                    "type": "done",
                    "output_path": res.output_path,
                    "input_bytes": res.input_bytes,
                    "output_bytes": res.output_bytes,
                    "ratio": res.ratio,
                    "quant": res.quant,
                }
            )
        finally:
            q.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        loop = asyncio.get_running_loop()
        try:
            while True:
                evt = await loop.run_in_executor(None, q.get)
                if evt is DONE:
                    return
                yield _sse(evt)
        finally:
            # Client disconnected — stop llama-imatrix / llama-quantize rather
            # than letting a multi-minute job run on with nobody watching.
            for proc in proc_sink:
                inference._kill_proc(proc)

    return gen


@app.post("/api/compress")
def compress_model(req: CompressRequest):
    models = _model_index()
    if req.name not in models:
        raise HTTPException(404, f"Unknown model {req.name!r}")
    gen = _stream_compress(models[req.name], req.target)
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- inference ----------
class InferenceRequest(BaseModel):
    model_name: str
    model_path: str
    architecture: str | None = None
    prompt: str
    n_predict: int = 256


def _stream_inference(req: InferenceRequest):
    q: queue.Queue = queue.Queue()
    DONE = object()
    proc_sink: list = []  # populated by inference.run_streaming once the proc starts

    def worker() -> None:
        try:
            for chunk, stats in inference.run_streaming(
                req.model_path,
                req.model_name,
                req.architecture,
                req.prompt,
                n_predict=req.n_predict,
                proc_sink=proc_sink,
            ):
                if stats is not None:
                    q.put(
                        {
                            "type": "done",
                            "tokens": stats.tokens,
                            "elapsed_s": stats.elapsed_s,
                            "tokens_per_s": stats.tokens_per_s,
                            "peak_rss_bytes": stats.peak_rss_bytes,
                        }
                    )
                elif chunk.lstrip().startswith(("[warning]", "[timeout]", "[runaway]")):
                    q.put({"type": "warning", "message": chunk.strip()})
                else:
                    q.put({"type": "token", "text": chunk})
        except FileNotFoundError as e:
            q.put({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            q.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        loop = asyncio.get_running_loop()
        try:
            while True:
                evt = await loop.run_in_executor(None, q.get)
                if evt is DONE:
                    return
                yield _sse(evt)
        finally:
            # Client disconnected (closed tab, hit cancel, ran another query) —
            # kill the llama-cli subprocess so the model unloads from RAM.
            for proc in proc_sink:
                inference._kill_proc(proc)

    return gen


@app.post("/api/inference")
def run_inference(req: InferenceRequest):
    gen = _stream_inference(req)
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- download ----------
@app.get("/api/download/{filename}")
def download(filename: str, as_name: str | None = None):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "invalid filename")
    path = (COMPRESSED_DIR / filename).resolve()
    root = COMPRESSED_DIR.resolve()
    # Prevent path traversal: resolved file must live inside compressed/.
    if not str(path).startswith(str(root) + "/"):
        raise HTTPException(403, "forbidden")
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(
        path,
        filename=as_name or path.name,
        media_type="application/octet-stream",
    )


# ---------- static frontend (when built) ----------
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
