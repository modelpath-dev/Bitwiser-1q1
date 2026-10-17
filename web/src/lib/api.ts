export type Model = {
  name: string;
  source: "ollama" | "gguf";
  path: string;
  size_bytes: number;
  family: string | null;
  quant: string | null;
  architecture: string | null;
  multimodal: boolean;
};

export type SystemInfo = {
  physical_cores: number;
  optimal_threads: number;
  backend: string; // e.g. "CUDA · NVIDIA GeForce GTX 1650", "Metal · Apple M3 Max", "CPU"
  gpu_layers: string; // value passed to -ngl: "99" on macOS, "auto" elsewhere
};

export type CompressPhase = "calibrate" | "quantize";

export type CompressEvent =
  | { type: "progress"; pct: number; line: string; phase: CompressPhase }
  | { type: "warning"; message: string }
  | {
      type: "done";
      output_path: string;
      input_bytes: number;
      output_bytes: number;
      ratio: number;
      quant: string;
    }
  | { type: "error"; message: string };

export type InferenceEvent =
  | { type: "token"; text: string }
  | { type: "warning"; message: string }
  | {
      type: "done";
      // null when llama.cpp exited without printing its perf summary (killed early)
      tokens: number | null;
      elapsed_s: number;
      tokens_per_s: number | null;
      peak_rss_bytes: number;
    }
  | { type: "error"; message: string };

export async function fetchModels(): Promise<{ models: Model[]; system: SystemInfo }> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error(`models: ${res.status}`);
  return res.json();
}

/** Stream SSE events from a POST endpoint. Yields parsed JSON for each `data:` chunk. */
export async function* sseStream<T>(
  url: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`${url}: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    // SSE event delimiter is a blank line.
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of raw.split("\n")) {
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          if (payload) {
            try {
              yield JSON.parse(payload) as T;
            } catch {
              // ignore malformed
            }
          }
        }
      }
    }
  }
}

export function compressStream(
  name: string,
  signal?: AbortSignal,
  target = "IQ1_S",
) {
  return sseStream<CompressEvent>("/api/compress", { name, target }, signal);
}

export function inferenceStream(
  body: {
    model_name: string;
    model_path: string;
    architecture: string | null;
    prompt: string;
    n_predict: number;
  },
  signal?: AbortSignal,
) {
  return sseStream<InferenceEvent>("/api/inference", body, signal);
}
