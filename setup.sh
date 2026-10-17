#!/usr/bin/env bash
# BitWiser — environment bootstrap for macOS (Metal) and Linux (CUDA or CPU).
# Builds llama.cpp for the detected GPU backend and installs Python deps.
#
#   BACKEND=cpu ./setup.sh   force a CPU-only build (Linux)
#   JOBS=2 ./setup.sh        limit parallel compile jobs (CUDA builds are RAM-hungry)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENDOR="$ROOT/vendor"
LLAMA="$VENDOR/llama.cpp"
# Pinned: llama.cpp renames/removes CLI flags on master (e.g. --no-mmap became
# --load-mode), which breaks the commands BitWiser builds. Bump deliberately
# after re-testing compress + inference.
LLAMA_COMMIT="2744e48d1e88d7416224e92bf58d4ffedda4a205"

case "$(uname -s)" in
  Darwin)
    BACKEND="${BACKEND:-metal}"
    JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
    CMAKE_HINT="brew install cmake"
    ;;
  Linux)
    if [ -z "${BACKEND:-}" ]; then
      if command -v nvcc >/dev/null 2>&1; then BACKEND=cuda; else BACKEND=cpu; fi
    fi
    JOBS="${JOBS:-$(nproc)}"
    CMAKE_HINT="sudo apt install cmake build-essential"
    ;;
  *)
    echo "Unsupported OS: $(uname -s)"
    exit 1
    ;;
esac

# Backend flags are always passed explicitly so a re-run with a different
# BACKEND overrides what CMake cached from the previous configure.
case "$BACKEND" in
  metal) BACKEND_FLAGS=(-DGGML_METAL=ON) ;;
  cuda)
    BACKEND_FLAGS=(-DGGML_CUDA=ON)
    # Compile only for this machine's GPU; the default arch list takes far longer.
    if CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)" && [ -n "$CAP" ]; then
      BACKEND_FLAGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CAP//./}")
    fi
    # nvcc < 11.6 can't parse GCC 11+'s <functional> ("parameter packs not
    # expanded"), and Ubuntu 22.04's apt CUDA is 11.5. Build with g++-10 instead.
    read -r NVCC_MAJOR NVCC_MINOR < <(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
    GCC_MAJOR="$(c++ -dumpversion | cut -d. -f1)"
    if [ "$NVCC_MAJOR" -eq 11 ] && [ "$NVCC_MINOR" -lt 6 ] && [ "$GCC_MAJOR" -ge 11 ]; then
      if command -v g++-10 >/dev/null 2>&1; then
        BACKEND_FLAGS+=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-10)")
      else
        echo "CUDA $NVCC_MAJOR.$NVCC_MINOR can't compile with g++ $GCC_MAJOR. Fix with one of:"
        echo "  sudo apt install g++-10     (setup.sh then uses it automatically)"
        echo "  BACKEND=cpu ./setup.sh      (CPU-only build)"
        exit 1
      fi
    fi
    ;;
  cpu) BACKEND_FLAGS=(-DGGML_CUDA=OFF) ;;
  *)
    echo "Unknown BACKEND=$BACKEND (expected metal, cuda or cpu)"
    exit 1
    ;;
esac

mkdir -p "$VENDOR"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. Install with: $CMAKE_HINT"
  exit 1
fi

if [ ! -d "$LLAMA" ]; then
  echo "==> Cloning llama.cpp"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
echo "==> Checking out llama.cpp @ ${LLAMA_COMMIT:0:7}"
git -C "$LLAMA" fetch --quiet origin "$LLAMA_COMMIT"
git -C "$LLAMA" checkout --quiet "$LLAMA_COMMIT"

echo "==> Configuring for $BACKEND"
cmake -S "$LLAMA" -B "$LLAMA/build" "${BACKEND_FLAGS[@]}" -DLLAMA_CURL=OFF

echo "==> Building (Release, $JOBS jobs)"
cmake --build "$LLAMA/build" --config Release -j "$JOBS" \
  --target llama-quantize llama-imatrix llama-completion

echo "==> Installing Python deps into .venv"
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"

echo
echo "Done ($BACKEND). Binaries:"
ls "$LLAMA/build/bin" | grep -E '^(llama-quantize|llama-imatrix|llama-completion)$' || true
