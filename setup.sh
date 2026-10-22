#!/usr/bin/env bash
# One-step setup for medextract (macOS and Linux).
#
#   ./setup.sh                    auto-detect hardware, create .venv, install, run a quick check
#   ./setup.sh cpu                force a backend: mlx | cuda | cpu
#   ./setup.sh --recreate         delete and rebuild .venv
#   ./setup.sh --download-model   also download the model weights now (~5 GB MLX, ~17 GB otherwise)
#   ./setup.sh --check            only show which Python and backend would be used; change nothing
#   ./setup.sh --no-test          skip the unit-test check at the end
#
# Set PYTHON=/path/to/python3.x to choose the interpreter yourself.

set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
MIN_MINOR=10          # Python 3.10+
BACKEND="auto"
RECREATE=0
DOWNLOAD=0
CHECK_ONLY=0
RUN_TESTS=1

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
fail() { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    auto|mlx|cuda|cpu) BACKEND="$arg" ;;
    --recreate) RECREATE=1 ;;
    --download-model) DOWNLOAD=1 ;;
    --check) CHECK_ONLY=1 ;;
    --no-test) RUN_TESTS=0 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown option '$arg' (see ./setup.sh --help)" ;;
  esac
done

# --- 1. Find a Python 3.10+ interpreter --------------------------------------------------

python_ok() {  # usable interpreter: version >= 3.MIN_MINOR and has the venv module
  "$1" -c "import sys, venv; sys.exit(0 if sys.version_info >= (3, $MIN_MINOR) else 1)" >/dev/null 2>&1
}

find_python() {
  local candidates=()
  [ -n "${PYTHON:-}" ] && candidates+=("$PYTHON")
  # An active conda environment wins: the user chose its Python on purpose.
  [ -n "${CONDA_PREFIX:-}" ] && candidates+=("$CONDA_PREFIX/bin/python3")
  # Then tested versions, newest first; the default python3; common install locations.
  candidates+=(python3.13 python3.12 python3.11 python3.10 python3)
  candidates+=(
    /opt/homebrew/bin/python3 /usr/local/bin/python3
    "$HOME/miniforge3/bin/python3" "$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3"
    /opt/homebrew/Caskroom/miniforge/base/bin/python3
    /Library/Frameworks/Python.framework/Versions/Current/bin/python3
  )
  local c path
  for c in "${candidates[@]}"; do
    path="$(command -v "$c" 2>/dev/null || true)"
    [ -z "$path" ] && [ -x "$c" ] && path="$c"
    if [ -n "$path" ] && python_ok "$path"; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

bold "1/4  Python"
if ! PY="$(find_python)"; then
  info "No Python 3.$MIN_MINOR+ found. (macOS ships python3 3.9, which is too old.)"
  info "Install one, then run ./setup.sh again:"
  info "  macOS:  brew install python@3.12        (or download from https://www.python.org/downloads/)"
  info "  Linux:  sudo apt install python3.12 python3.12-venv   (or your distro's equivalent)"
  fail "Python 3.$MIN_MINOR or newer is required."
fi
info "using $PY ($("$PY" --version 2>&1))"

# --- 2. Pick the backend -----------------------------------------------------------------

detect_backend() {
  local os arch
  os="$(uname -s)"; arch="$(uname -m)"
  if [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
    echo mlx
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo cuda
  else
    echo cpu
  fi
}

bold "2/4  Hardware"
if [ "$BACKEND" = "auto" ]; then
  BACKEND="$(detect_backend)"
  info "detected backend: $BACKEND"
else
  info "backend chosen on the command line: $BACKEND"
fi
if [ "$BACKEND" = "mlx" ] && { [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; }; then
  fail "the mlx backend needs a Mac with Apple Silicon (M1 or newer). Use: ./setup.sh cpu"
fi
case "$BACKEND" in
  mlx)  info "Apple GPU (Metal) with mlx-community/Qwen3-VL-8B-Instruct-4bit (~5 GB download, ~7 GB RAM)" ;;
  cuda) info "NVIDIA GPU, Qwen/Qwen3-VL-8B-Instruct in 4-bit (~17 GB download)" ;;
  cpu)  info "CPU only: works everywhere but is slow; the 8B model needs ~17 GB RAM" ;;
esac

if [ "$CHECK_ONLY" = 1 ]; then
  bold "Check only: nothing was changed."
  exit 0
fi

# --- 3. Create the virtual environment and install --------------------------------------

bold "3/4  Install into $VENV/"
if [ -d "$VENV" ] && [ "$RECREATE" = 0 ]; then
  if ! [ -x "$VENV/bin/python" ] || ! python_ok "$VENV/bin/python"; then
    info "existing $VENV uses an unsupported or broken Python; rebuilding it"
    RECREATE=1
  fi
fi
if [ "$RECREATE" = 1 ] || [ ! -d "$VENV" ]; then
  "$PY" -m venv --clear "$VENV"
  info "created $VENV with $("$VENV/bin/python" --version 2>&1)"
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip
info "pip $("$VENV/bin/python" -m pip --version | awk '{print $2}')"
info "installing medextract[$BACKEND,dev] (first time takes a few minutes)..."
"$VENV/bin/python" -m pip install --quiet -e ".[$BACKEND,dev]"

# --- 4. Check -------------------------------------------------------------------------------

bold "4/4  Check"
MEDEXTRACT_DEVICE="$BACKEND" "$VENV/bin/medextract" info | sed -n '2,3p' | sed 's/^ */  /'
if [ "$RUN_TESTS" = 1 ]; then
  "$VENV/bin/python" -m pytest -q -p no:cacheprovider 2>&1 | tail -1 | sed 's/^/  unit tests: /'
fi
if [ "$DOWNLOAD" = 1 ]; then
  info "downloading model weights..."
  MEDEXTRACT_DEVICE="$BACKEND" "$VENV/bin/python" - <<'EOF'
import os
from huggingface_hub import snapshot_download
from medextract.backends.factory import resolve_model_id
model_id = resolve_model_id(os.environ["MEDEXTRACT_DEVICE"], os.environ.get("MEDEXTRACT_MODEL_ID"))
print("  saved to", snapshot_download(model_id))
EOF
fi

echo
bold "Done. Try it:"
cat <<EOF
  $VENV/bin/medextract run document-tests/01_receipt_en.pdf          # one file -> JSON on screen
  $VENV/bin/medextract run your_file.pdf your_scan.jpg -o output/    # your own files -> output/*.json
  $VENV/bin/medextract run document-tests/ -o results/ --meta        # whole folder, with diagnostics

  Or activate the environment once and drop the prefix:
  source $VENV/bin/activate && medextract run document-tests/01_receipt_en.pdf
EOF
if [ "$BACKEND" != "$(detect_backend)" ]; then
  echo
  echo "  Note: auto-detection would pick '$(detect_backend)'. To use '$BACKEND', add --device $BACKEND"
  echo "  to each command or run: export MEDEXTRACT_DEVICE=$BACKEND"
fi
