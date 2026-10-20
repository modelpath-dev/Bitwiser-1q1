import { useEffect, useRef, useState } from "react";
import { compressStream, type CompressPhase, type Model } from "../lib/api";
import { humanizeBytes } from "../lib/format";

const ALREADY_TINY = new Set(["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "TQ1_0", "TQ2_0", "Q1_0"]);

export type CompressedArtifact = {
  inputBytes: number;
  outputBytes: number;
  outputPath: string;
  ratio: number;
  quant: string;
};

type Props = {
  model: Model;
  onCompressed: (a: CompressedArtifact) => void;
};

export function ShrinkPanel({ model, onCompressed }: Props) {
  const [running, setRunning] = useState(false);
  const [pct, setPct] = useState(0);
  const [phase, setPhase] = useState<CompressPhase>("quantize");
  const [logs, setLogs] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<CompressedArtifact | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const isTiny = ALREADY_TINY.has((model.quant ?? "").toUpperCase());
  const isMultimodal = model.multimodal;
  const blocked = isTiny || isMultimodal;

  // Abort on unmount: the server kills llama-imatrix / llama-quantize when the
  // stream closes, and a stale stream must never call onCompressed.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function start() {
    setRunning(true);
    setPct(0);
    setPhase("quantize");
    setLogs([]);
    setWarnings([]);
    setError(null);
    setDone(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      for await (const evt of compressStream(model.name, ctrl.signal)) {
        if (evt.type === "progress") {
          setPct(evt.pct);
          setPhase(evt.phase);
          setLogs((prev) => {
            const next = [...prev, evt.line];
            return next.length > 200 ? next.slice(-200) : next;
          });
        } else if (evt.type === "warning") {
          setWarnings((prev) => [...prev, evt.message]);
        } else if (evt.type === "error") {
          setError(evt.message);
        } else if (evt.type === "done") {
          const artifact: CompressedArtifact = {
            inputBytes: evt.input_bytes,
            outputBytes: evt.output_bytes,
            outputPath: evt.output_path,
            ratio: evt.ratio,
            quant: evt.quant,
          };
          setDone(artifact);
          setPct(1);
          onCompressed(artifact);
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message);
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  return (
    <div className="glass-strong p-6">
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[--color-accent-300]">
            <span className="pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-[--color-accent-500]" />
            Step 01 · Shrink
          </div>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">
            Compress to{" "}
            <span
              className="bg-clip-text text-transparent"
              style={{
                backgroundImage:
                  "linear-gradient(135deg, var(--color-accent-400), var(--color-cyan-glow))",
              }}
            >
              IQ1_S
            </span>
          </h2>
          <p className="mt-1 max-w-xl text-sm text-[--color-ink-300]">
            Importance Quantization — ~1.56 bits per weight. Output lands in{" "}
            <code className="font-mono text-[--color-ink-100]">compressed/</code>.
          </p>
        </div>

        <div className="flex shrink-0 gap-2">
          {running ? (
            <button onClick={cancel} className="btn-ghost">
              Cancel
            </button>
          ) : (
            <button
              onClick={start}
              disabled={blocked}
              className="btn-primary"
            >
              {done ? "Re-shrink" : "Shrink to 1-bit"}
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path
                  d="M5 12h14M13 6l6 6-6 6"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
        </div>
      </div>

      {isTiny && !running && (
        <div className="mt-5 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-sm text-amber-200">
          This model is already <code className="font-mono">{model.quant}</code> —
          there's nothing left to compress.
        </div>
      )}

      {isMultimodal && !running && (
        <div className="mt-5 rounded-2xl border border-amber-400/25 bg-amber-400/[0.04] p-4 text-sm text-amber-100">
          <div className="flex items-start gap-3">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" className="mt-0.5 shrink-0">
              <path
                d="M9 1.5l8 14H1L9 1.5zM9 6.5v4M9 12.5v.5"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <div className="space-y-2 leading-relaxed">
              <p>
                <strong className="font-semibold">Multimodal model — not supported by llama.cpp's quantizer.</strong>{" "}
                This GGUF includes vision and audio tensors alongside the text decoder.
              </p>
              <ul className="ml-4 list-disc space-y-1 text-amber-100/85">
                <li>
                  <code className="font-mono text-xs">llama-imatrix</code> can't load
                  it (tensor-count mismatch), so{" "}
                  <code className="font-mono text-xs">IQ1_S</code> /{" "}
                  <code className="font-mono text-xs">IQ2_*</code> are unreachable.
                </li>
                <li>
                  <code className="font-mono text-xs">llama-quantize</code> crashes
                  on the audio tower's 4D conv weights (
                  <code className="font-mono text-xs">llama-quant.cpp:1256</code>{" "}
                  assertion), so <code className="font-mono text-xs">Q2_K</code> /{" "}
                  <code className="font-mono text-xs">Q4_K</code> also fail.
                </li>
              </ul>
              <p className="text-amber-100/85">
                Try a text-only model — e.g.{" "}
                <code className="font-mono text-xs">ollama pull gemma2:9b</code>,{" "}
                <code className="font-mono text-xs">ollama pull llama3.1:8b</code>, or{" "}
                <code className="font-mono text-xs">ollama pull mistral:7b</code>.
              </p>
            </div>
          </div>
        </div>
      )}

      {(running || done || error) && (
        <div className="mt-6 space-y-4">
          {/* warnings (e.g. multimodal fallback) */}
          {warnings.length > 0 && (
            <div className="space-y-2">
              {warnings.map((w, i) => (
                <div
                  key={i}
                  className="flex gap-3 rounded-xl border border-amber-400/30 bg-amber-400/[0.06] p-3 text-sm text-amber-100"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    className="mt-0.5 shrink-0"
                  >
                    <path
                      d="M8 1.5l7 12.5H1L8 1.5zM8 6v3.5M8 11.5v.5"
                      stroke="currentColor"
                      strokeWidth="1.4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span className="leading-relaxed">{w}</span>
                </div>
              ))}
            </div>
          )}

          {/* phase indicator */}
          {(running || done) && !error && (
            <div className="flex items-center gap-2 text-xs">
              <PhaseChip
                label="01 · Calibrate"
                hint="imatrix"
                state={
                  phase === "calibrate" && running
                    ? "active"
                    : "done"
                }
              />
              <div className="h-px flex-1 bg-white/8" />
              <PhaseChip
                label="02 · Quantize"
                hint="IQ1_S"
                state={
                  done
                    ? "done"
                    : phase === "quantize" && running
                      ? "active"
                      : "pending"
                }
              />
            </div>
          )}

          {/* progress bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-[--color-ink-300]">
              <span className="font-mono">
                {error
                  ? "error"
                  : done
                    ? "complete"
                    : `${phase === "calibrate" ? "calibrating" : "quantizing"} · ${(pct * 100).toFixed(1)}%`}
              </span>
              <span className="font-mono">
                {done
                  ? `${humanizeBytes(done.inputBytes)} → ${humanizeBytes(done.outputBytes)}`
                  : "—"}
              </span>
            </div>
            <div className="relative h-2 overflow-hidden rounded-full bg-white/[0.06]">
              <div
                className="progress-fill absolute inset-y-0 left-0"
                style={{ width: `${Math.max(pct, error ? 1 : 0) * 100}%` }}
              />
              {running && (
                <div className="shimmer absolute inset-0" />
              )}
            </div>
          </div>

          {error && (
            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
              {error}
            </div>
          )}

          {/* logs */}
          {logs.length > 0 && (
            <details className="group rounded-xl border border-white/8 bg-black/30">
              <summary className="cursor-pointer select-none px-4 py-2.5 text-xs font-medium text-[--color-ink-300] transition-colors hover:text-[--color-ink-100]">
                Quantizer log
                <span className="ml-2 font-mono text-[--color-ink-400]">
                  ({logs.length})
                </span>
              </summary>
              <pre className="max-h-64 overflow-auto px-4 pb-4 font-mono text-[11px] leading-relaxed text-[--color-ink-300]">
                {logs.slice(-80).join("\n")}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function PhaseChip({
  label,
  hint,
  state,
}: {
  label: string;
  hint: string;
  state: "pending" | "active" | "done";
}) {
  const tone =
    state === "active"
      ? "border-[--color-accent-500]/40 bg-[--color-accent-500]/10 text-[--color-accent-300]"
      : state === "done"
        ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
        : "border-white/8 bg-white/[0.03] text-[--color-ink-400]";
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 ${tone}`}
    >
      {state === "active" ? (
        <span className="pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-[--color-accent-500]" />
      ) : state === "done" ? (
        <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
          <path
            d="M2.5 6.5L5 9l4.5-5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : (
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-current opacity-40" />
      )}
      <span className="text-[11px] font-medium tracking-wide">{label}</span>
      <span className="hidden font-mono text-[10px] opacity-60 sm:inline">
        {hint}
      </span>
    </span>
  );
}
