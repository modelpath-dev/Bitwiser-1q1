import { useRef, useState } from "react";
import { inferenceStream, type Model } from "../lib/api";
import type { CompressedArtifact } from "./ShrinkPanel";

type RunState = {
  text: string;
  warning: string | null;
  error: string | null;
  running: boolean;
  stats: { tokens: number | null; elapsed_s: number; tokens_per_s: number | null } | null;
};

const INITIAL: RunState = {
  text: "",
  warning: null,
  error: null,
  running: false,
  stats: null,
};

type Props = {
  model: Model;
  artifact: CompressedArtifact | null;
};

export function Playground({ model, artifact }: Props) {
  const [prompt, setPrompt] = useState(
    "Explain why 1-bit quantization works in one short paragraph.",
  );
  const [nPredict, setNPredict] = useState(256);
  const [left, setLeft] = useState<RunState>(INITIAL);
  const [right, setRight] = useState<RunState>(INITIAL);
  const leftAbort = useRef<AbortController | null>(null);
  const rightAbort = useRef<AbortController | null>(null);

  async function run(
    side: "left" | "right",
    modelPath: string,
  ) {
    const setter = side === "left" ? setLeft : setRight;
    const abortRef = side === "left" ? leftAbort : rightAbort;

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setter({ ...INITIAL, running: true });

    try {
      for await (const evt of inferenceStream(
        {
          model_name: model.name,
          model_path: modelPath,
          architecture: model.architecture,
          prompt,
          n_predict: nPredict,
        },
        ctrl.signal,
      )) {
        if (evt.type === "token") {
          setter((s) => ({ ...s, text: s.text + evt.text }));
        } else if (evt.type === "warning") {
          setter((s) => ({ ...s, warning: evt.message }));
        } else if (evt.type === "error") {
          setter((s) => ({ ...s, error: evt.message, running: false }));
        } else if (evt.type === "done") {
          setter((s) => ({
            ...s,
            running: false,
            stats: {
              tokens: evt.tokens,
              elapsed_s: evt.elapsed_s,
              tokens_per_s: evt.tokens_per_s,
            },
          }));
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setter((s) => ({ ...s, error: (e as Error).message, running: false }));
      }
    } finally {
      setter((s) => ({ ...s, running: false }));
    }
  }

  function runBoth() {
    run("left", model.path);
    if (artifact) run("right", artifact.outputPath);
  }

  return (
    <section className="space-y-4">
      <div className="glass-strong p-6">
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[--color-accent-300]">
          <span className="pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-[--color-accent-500]" />
          Step 02 · Playground
        </div>
        <h2 className="mt-2 text-2xl font-semibold tracking-tight">Side-by-side</h2>
        <p className="mt-1 text-sm text-[--color-ink-300]">
          Same prompt, two models. Compare quality, throughput, and RAM.
        </p>

        <div className="mt-5 space-y-3">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="w-full resize-none rounded-2xl border border-white/8 bg-black/30 p-4 font-mono text-sm leading-relaxed text-[--color-ink-50] placeholder:text-[--color-ink-400] focus:border-[--color-accent-500] focus:outline-none"
            placeholder="Type a prompt..."
          />
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-3 text-xs text-[--color-ink-300]">
              <span className="uppercase tracking-[0.18em]">Max tokens</span>
              <input
                type="range"
                min={32}
                max={1024}
                step={32}
                value={nPredict}
                onChange={(e) => setNPredict(parseInt(e.target.value, 10))}
                className="h-1 w-44 cursor-pointer appearance-none rounded-full bg-white/[0.08] accent-[--color-accent-500]"
              />
              <span className="font-mono text-[--color-ink-100]">{nPredict}</span>
            </label>
            <div className="flex-1" />
            <button
              onClick={runBoth}
              className="btn-primary"
              disabled={left.running || right.running}
            >
              Run both
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <PlaygroundColumn
          title="Original"
          subtitle={model.quant ?? "unknown"}
          state={left}
          onRun={() => run("left", model.path)}
          accent="cyan"
        />
        <PlaygroundColumn
          title="1-bit"
          subtitle="IQ1_S"
          state={right}
          disabled={!artifact}
          disabledHint="Compress this model first."
          onRun={() => artifact && run("right", artifact.outputPath)}
          accent="accent"
        />
      </div>
    </section>
  );
}

function PlaygroundColumn({
  title,
  subtitle,
  state,
  onRun,
  disabled,
  disabledHint,
  accent,
}: {
  title: string;
  subtitle: string;
  state: RunState;
  onRun: () => void;
  disabled?: boolean;
  disabledHint?: string;
  accent: "accent" | "cyan";
}) {
  const tokensPerS = state.stats?.tokens_per_s ?? null;
  const tone =
    accent === "accent"
      ? "linear-gradient(135deg, rgba(139,92,246,0.18), transparent 70%)"
      : "linear-gradient(135deg, rgba(34,211,238,0.16), transparent 70%)";

  return (
    <div className="glass relative overflow-hidden">
      <div aria-hidden className="absolute inset-x-0 top-0 h-px" style={{ background: tone }} />

      <div className="flex items-center justify-between border-b border-white/5 p-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
          <span className={`pill ${accent === "accent" ? "pill-accent" : "pill-cyan"}`}>
            {subtitle}
          </span>
          {state.running && (
            <span className="pill">
              <span className="pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-[--color-accent-400]" />
              streaming
            </span>
          )}
        </div>
        <button
          onClick={onRun}
          disabled={disabled || state.running}
          className="btn-ghost px-3 py-1.5 text-xs"
        >
          {state.running ? "running..." : "run"}
        </button>
      </div>

      <div className="min-h-[14rem] p-4">
        {disabled && !state.text ? (
          <div className="flex h-full items-center justify-center py-8 text-sm text-[--color-ink-400]">
            {disabledHint}
          </div>
        ) : (
          <>
            {state.warning && (
              <div className="mb-3 rounded-lg border border-amber-400/30 bg-amber-400/5 p-2.5 text-xs text-amber-200">
                {state.warning}
              </div>
            )}
            {state.error && (
              <div className="mb-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-2.5 text-xs text-rose-200">
                {state.error}
              </div>
            )}
            <pre className="whitespace-pre-wrap break-words font-mono text-[13px] leading-relaxed text-[--color-ink-100]">
              {state.text || (
                <span className="text-[--color-ink-400]">
                  output will stream here…
                </span>
              )}
            </pre>
          </>
        )}
      </div>

      <div className="grid grid-cols-3 border-t border-white/5">
        <Stat label="tokens/s" value={tokensPerS != null ? tokensPerS.toFixed(1) : "—"} />
        <Stat label="tokens" value={state.stats?.tokens?.toString() ?? "—"} />
        <Stat
          label="elapsed"
          value={state.stats ? `${state.stats.elapsed_s.toFixed(1)}s` : "—"}
        />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-r border-white/5 p-3 last:border-r-0">
      <div className="text-[10px] uppercase tracking-[0.18em] text-[--color-ink-400]">
        {label}
      </div>
      <div
        className="mt-1 font-mono text-base text-[--color-ink-50]"
        style={{ fontVariantNumeric: "tabular-nums" }}
      >
        {value}
      </div>
    </div>
  );
}
