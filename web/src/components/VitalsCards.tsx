import { humanizeBytes, ratioPct } from "../lib/format";
import type { CompressedArtifact } from "./ShrinkPanel";

type Props = { artifact: CompressedArtifact | null };

export function VitalsCards({ artifact }: Props) {
  const saved = artifact ? Math.max(artifact.inputBytes - artifact.outputBytes, 0) : 0;
  const reduction = artifact ? ratioPct(artifact.inputBytes, artifact.outputBytes) : "—";
  const speedup = artifact
    ? `${(artifact.inputBytes / Math.max(artifact.outputBytes, 1)).toFixed(1)}×`
    : "—";

  const cards = [
    {
      label: "RAM saved",
      value: artifact ? humanizeBytes(saved) : "—",
      sub: artifact ? `${reduction} smaller on disk` : "compress a model to populate",
      tone: "accent" as const,
    },
    {
      label: "Bandwidth speedup",
      value: speedup,
      sub: "decode is memory-bandwidth bound",
      tone: "cyan" as const,
    },
    {
      label: "New file",
      value: artifact ? humanizeBytes(artifact.outputBytes) : "—",
      sub: artifact ? artifact.quant : "IQ1_S",
      tone: "amber" as const,
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      {cards.map((c) => (
        <div
          key={c.label}
          className="glass relative overflow-hidden p-5"
        >
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 -z-10 opacity-50"
            style={{
              background:
                c.tone === "accent"
                  ? "radial-gradient(80% 100% at 0% 0%, rgba(139,92,246,0.18), transparent 70%)"
                  : c.tone === "cyan"
                    ? "radial-gradient(80% 100% at 0% 0%, rgba(34,211,238,0.14), transparent 70%)"
                    : "radial-gradient(80% 100% at 0% 0%, rgba(251,191,36,0.12), transparent 70%)",
            }}
          />
          <div className="text-[11px] uppercase tracking-[0.2em] text-[--color-ink-400]">
            {c.label}
          </div>
          <div
            className="mt-3 font-mono text-3xl font-medium tracking-tight"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {c.value}
          </div>
          <div className="mt-2 text-xs text-[--color-ink-300]">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}
