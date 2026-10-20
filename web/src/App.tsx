import { useEffect, useState } from "react";
import { fetchModels, type Model, type SystemInfo } from "./lib/api";
import { ModelPicker } from "./components/ModelPicker";
import { ShrinkPanel, type CompressedArtifact } from "./components/ShrinkPanel";
import { VitalsCards } from "./components/VitalsCards";
import { DownloadCard } from "./components/DownloadCard";
import { Playground } from "./components/Playground";

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [selected, setSelected] = useState<Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [artifact, setArtifact] = useState<CompressedArtifact | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fetchModels()
      .then((res) => {
        if (!alive) return;
        setModels(res.models);
        setSystem(res.system);
        if (res.models.length > 0) setSelected(res.models[0]);
      })
      .catch((e: Error) => alive && setLoadError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  // Reset compressed-artifact when the selection changes.
  useEffect(() => {
    setArtifact(null);
  }, [selected?.name]);

  return (
    <div className="mx-auto min-h-full max-w-6xl px-6 py-12 lg:px-10 lg:py-16">
      <Header system={system} />

      <main className="mt-12 space-y-10">
        <Section
          eyebrow="Step 00"
          title="Pick a model"
          description="Auto-discovered from Ollama and ~/Models."
        >
          {loading ? (
            <SkeletonGrid />
          ) : loadError ? (
            <div className="glass p-6 text-sm text-rose-200">
              Could not reach the API: <code className="font-mono">{loadError}</code>.
              Is <code className="font-mono">server.py</code> running on port 8000?
            </div>
          ) : (
            <ModelPicker models={models} selected={selected} onSelect={setSelected} />
          )}
        </Section>

        {selected && (
          <>
            {/* Keyed by model so switching selection unmounts (and aborts) the
                previous model's compress/inference streams instead of letting
                them report into the newly selected model. */}
            <ShrinkPanel key={`shrink:${selected.name}`} model={selected} onCompressed={setArtifact} />
            <VitalsCards artifact={artifact} />
            {artifact && <DownloadCard model={selected} artifact={artifact} />}
            <Playground key={`play:${selected.name}`} model={selected} artifact={artifact} />
          </>
        )}
      </main>

      <footer className="mt-16 flex items-center justify-between border-t border-white/5 pt-6 text-xs text-[--color-ink-400]">
        <span>BitWiser · llama.cpp{system ? ` · ${system.backend}` : ""}</span>
        {system && (
          <span className="font-mono">
            {system.physical_cores} cores · {system.optimal_threads} threads · -ngl {system.gpu_layers}
          </span>
        )}
      </footer>
    </div>
  );
}

function Header({ system }: { system: SystemInfo | null }) {
  return (
    <header className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-[--color-accent-300]">
          <Logo />
          BitWiser
        </div>
        <h1 className="mt-4 text-balance text-4xl font-semibold tracking-tight lg:text-5xl">
          Any local LLM,{" "}
          <span
            className="bg-clip-text text-transparent"
            style={{
              backgroundImage:
                "linear-gradient(135deg, #c4b5fd 0%, #22d3ee 60%, #fbbf24 100%)",
            }}
          >
            one bit at a time.
          </span>
        </h1>
        <p className="mt-3 max-w-xl text-sm leading-relaxed text-[--color-ink-300]">
          Pick an Ollama or GGUF model, click Shrink, and run the original and the
          1-bit version side-by-side. GPU-accelerated where available, no cloud.
        </p>
      </div>
      {system && (
        <div className="flex shrink-0 gap-2">
          <span className="pill pill-accent">{system.backend} · -ngl {system.gpu_layers}</span>
          <span className="pill">{system.optimal_threads} threads</span>
          <span className="pill">IQ1_S</span>
        </div>
      )}
    </header>
  );
}

function Logo() {
  return (
    <span
      aria-hidden
      className="inline-flex h-5 w-5 items-center justify-center rounded-md"
      style={{
        background:
          "linear-gradient(135deg, var(--color-accent-500), var(--color-cyan-glow))",
        boxShadow: "0 0 18px rgba(139,92,246,0.5)",
      }}
    >
      <span className="font-mono text-[10px] font-bold text-black">1</span>
    </span>
  );
}

function Section({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-[0.2em] text-[--color-accent-300]">
            {eyebrow}
          </div>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h2>
        </div>
        <p className="hidden max-w-md text-sm text-[--color-ink-300] md:block">
          {description}
        </p>
      </div>
      {children}
    </section>
  );
}

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="glass relative overflow-hidden p-4">
          <div className="h-4 w-2/3 rounded bg-white/[0.06]" />
          <div className="mt-2 h-3 w-1/3 rounded bg-white/[0.04]" />
          <div className="mt-5 flex gap-2">
            <div className="h-5 w-16 rounded-full bg-white/[0.04]" />
            <div className="h-5 w-12 rounded-full bg-white/[0.04]" />
          </div>
          <div className="shimmer absolute inset-0" />
        </div>
      ))}
    </div>
  );
}
