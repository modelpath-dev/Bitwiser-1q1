#!/usr/bin/env bash
# BitWiser — environment bootstrap for Apple Silicon.
# Builds llama.cpp with Metal acceleration and installs Python deps.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENDOR="$ROOT/vendor"
LLAMA="$VENDOR/llama.cpp"
# Pinned: llama.cpp renames/removes CLI flags on master (e.g. --no-mmap became
# --load-mode), which breaks the commands BitWiser builds. Bump deliberately
# after re-testing compress + inference.
LLAMA_COMMIT="2744e48d1e88d7416224e92bf58d4ffedda4a205"

mkdir -p "$VENDOR"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. Install with: brew install cmake"
  exit 1
fi

if [ ! -d "$LLAMA" ]; then
  echo "==> Cloning llama.cpp"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
echo "==> Checking out llama.cpp @ ${LLAMA_COMMIT:0:7}"
git -C "$LLAMA" fetch --quiet origin "$LLAMA_COMMIT"
git -C "$LLAMA" checkout --quiet "$LLAMA_COMMIT"

echo "==> Configuring with Metal"
cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_METAL=ON -DLLAMA_CURL=OFF

echo "==> Building (Release)"
cmake --build "$LLAMA/build" --config Release -j

echo "==> Installing Python deps into .venv"
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"

echo
echo "Done. Binaries:"
ls "$LLAMA/build/bin" | grep -E '^(llama-quantize|llama-imatrix|llama-completion)$' || true
