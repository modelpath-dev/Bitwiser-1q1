import type { Model } from "../lib/api";
import { humanizeBytes, inferFamily } from "../lib/format";

const ALREADY_TINY = new Set(["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "TQ1_0", "TQ2_0", "Q1_0"]);

type Props = {
  models: Model[];
  selected: Model | null;
  onSelect: (m: Model) => void;
};

export function ModelPicker({ models, selected, onSelect }: Props) {
  if (models.length === 0) {
    return (
      <div className="glass p-8 text-center text-sm text-[--color-ink-300]">
        No models found. Pull one with{" "}
        <code className="font-mono text-[--color-ink-100]">ollama pull gemma2</code> or
        drop a <code className="font-mono text-[--color-ink-100]">.gguf</code> into{" "}
        <code className="font-mono text-[--color-ink-100]">~/Models</code>.
      </div>
    );
  }

  return (
    <ul className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
      {models.map((m) => {
        const isSelected = selected?.name === m.name;
        const tiny = ALREADY_TINY.has((m.quant ?? "").toUpperCase());
        const family = inferFamily(m);
        return (
          <li key={m.name}>
            <button
              type="button"
              onClick={() => onSelect(m)}
              className={`group relative w-full overflow-hidden rounded-2xl border p-4 text-left transition-all duration-200 ${
                isSelected
                  ? "border-[--color-accent-500] bg-white/[0.06]"
                  : "border-white/8 bg-white/[0.02] hover:bg-white/[0.04]"
              }`}
            >
              {isSelected && (
                <div
                  aria-hidden
                  className="absolute inset-0 -z-10 opacity-50"
                  style={{
                    background:
                      "radial-gradient(60% 100% at 0% 0%, rgba(139,92,246,0.18), transparent 70%)",
                  }}
                />
              )}

              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate font-medium text-[--color-ink-50]">
                    {m.name}
                  </div>
                  <div className="mt-0.5 text-xs uppercase tracking-wider text-[--color-ink-400]">
                    {family} · {m.source}
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm text-[--color-ink-100]">
                    {humanizeBytes(m.size_bytes)}
                  </div>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-1.5">
                <span className={`pill ${tiny ? "pill-amber" : "pill-cyan"}`}>
                  {m.quant ?? "unknown"}
                </span>
                {tiny ? (
                  <span className="pill pill-amber">already tiny</span>
                ) : null}
                {m.multimodal ? (
                  <span
                    className="pill pill-amber"
                    title="Vision/audio tensors — llama.cpp's quantizer can't process this GGUF"
                  >
                    multimodal · unsupported
                  </span>
                ) : null}
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
