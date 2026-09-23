#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Qwen3.8 + llama.cpp bootstrap for Debian/Ubuntu/RunPod-like images.
# Usage:
#   chmod +x qwen_setup.sh
#   ./qwen_setup.sh               # interactive TUI
#   ./qwen_setup.sh all           # deps -> download -> build -> start
#   ./qwen_setup.sh start
#   ./qwen_setup.sh status
#   ./qwen_setup.sh attach
#   ./qwen_setup.sh stop
#
# Optional:
#   export HF_TOKEN='hf_...'
#   export LLAMA_API_KEY='...'
#   export WORKSPACE=/workspace

WORKSPACE="${WORKSPACE:-/workspace}"
MODEL_REPO="${MODEL_REPO:-huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF}"
MODEL_FILE="${MODEL_FILE:-Huihui-Qwen3.8-27B-abliterated-Q8_0_L.gguf}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE/models/qwen}"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

LLAMA_DIR="${LLAMA_DIR:-$WORKSPACE/llama.cpp}"
HF_VENV="${HF_VENV:-$WORKSPACE/.venvs/hf}"
TMUX_SESSION="${TMUX_SESSION:-qwen}"
SERVER_RUNNER="$LLAMA_DIR/run-qwen.sh"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX="${CTX:-524288}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
THREADS="${THREADS:-8}"
THREADS_BATCH="${THREADS_BATCH:-8}"
HTTP_THREADS="${HTTP_THREADS:-4}"

# Requested Qwen3.8 tuning.
ROPE_SCALE="${ROPE_SCALE:-2}"
YARN_ORIG_CTX="${YARN_ORIG_CTX:-262144}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q4_0}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-256}"

# Q8_0_L is ~38.8 GB. Leave room for repo/build/temp files.
MIN_FREE_GB="${MIN_FREE_GB:-45}"

C_RESET='\033[0m'
C_BOLD='\033[1m'
C_BLUE='\033[34m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'
C_RED='\033[31m'

log()  { printf "${C_BLUE}==>${C_RESET} %s\n" "$*"; }
ok()   { printf "${C_GREEN}[OK]${C_RESET} %s\n" "$*"; }
warn() { printf "${C_YELLOW}[WARN]${C_RESET} %s\n" "$*" >&2; }
die()  { printf "${C_RED}[ERR]${C_RESET} %s\n" "$*" >&2; exit 1; }

trap 'printf "\n%b[ERR]%b line %s: %s\n" "$C_RED" "$C_RESET" "$LINENO" "$BASH_COMMAND" >&2' ERR

if [[ $EUID -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  die "Serve root o sudo per installare i pacchetti APT."
fi

have() { command -v "$1" >/dev/null 2>&1; }

header() {
  clear 2>/dev/null || true
  printf "${C_BOLD}Qwen3.8 / llama.cpp CUDA bootstrap${C_RESET}\n"
  printf "Workspace : %s\nModel     : %s/%s\nSession   : %s\n\n" \
    "$WORKSPACE" "$MODEL_REPO" "$MODEL_FILE" "$TMUX_SESSION"
}

disk_preflight() {
  mkdir -p "$WORKSPACE" "$MODEL_DIR"
  local free_kb free_gb
  free_kb="$(df -Pk "$WORKSPACE" | awk 'NR==2 {print $4}')"
  free_gb=$((free_kb / 1024 / 1024))

  if [[ -f "$MODEL_PATH" ]]; then
    ok "Modello già presente: $MODEL_PATH ($(du -h "$MODEL_PATH" | awk '{print $1}'))"
    return 0
  fi

  log "Spazio libero su $WORKSPACE: ${free_gb} GB"
  (( free_gb >= MIN_FREE_GB )) || die \
    "Spazio insufficiente: servono almeno ${MIN_FREE_GB} GB liberi prima del download."
}

apt_setup() {
  have apt-get || die "Questo script richiede Debian/Ubuntu con apt-get."

  export DEBIAN_FRONTEND=noninteractive
  log "APT update"
  "${SUDO[@]}" apt-get update

  log "APT upgrade"
  "${SUDO[@]}" apt-get upgrade -y

  log "Installazione dipendenze"
  "${SUDO[@]}" apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libcurl4-openssl-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    tmux

  ok "Dipendenze base installate"
}

cuda_preflight() {
  have nvidia-smi || die "nvidia-smi non trovato: il container/host non espone una GPU NVIDIA."
  nvidia-smi >/dev/null || die "nvidia-smi esiste ma non riesce a comunicare con il driver NVIDIA."

  if ! have nvcc; then
    warn "nvcc non trovato. Per compilare GGML_CUDA serve il CUDA Toolkit."
    if have apt-cache && apt-cache show nvidia-cuda-toolkit >/dev/null 2>&1; then
      log "Provo a installare nvidia-cuda-toolkit da APT"
      "${SUDO[@]}" apt-get install -y nvidia-cuda-toolkit
    fi
  fi

  have nvcc || die \
    "CUDA Toolkit/nvcc ancora assente. Usa un'immagine CUDA *devel* (consigliato) oppure installa un Toolkit compatibile."

  ok "GPU NVIDIA + CUDA Toolkit rilevati"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  nvcc --version | tail -n 4 || true
}

setup_hf() {
  have python3 || die "python3 non trovato."
  mkdir -p "$(dirname "$HF_VENV")"

  if [[ ! -x "$HF_VENV/bin/python" ]]; then
    log "Creo virtualenv Hugging Face: $HF_VENV"
    python3 -m venv "$HF_VENV"
  fi

  log "Aggiorno pip + huggingface_hub"
  "$HF_VENV/bin/python" -m pip install -U pip huggingface_hub

  [[ -x "$HF_VENV/bin/hf" ]] || die "Installazione hf CLI fallita."
  "$HF_VENV/bin/hf" version 2>/dev/null || "$HF_VENV/bin/hf" --help >/dev/null
  ok "hf CLI pronto: $HF_VENV/bin/hf"
}

hf_auth_if_needed() {
  local hf="$HF_VENV/bin/hf"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN presente: login non interattivo"
    "$hf" auth login --token "$HF_TOKEN" >/dev/null
    return 0
  fi

  # Public model: auth is not required. If already logged in, just report it.
  if "$hf" auth whoami >/dev/null 2>&1; then
    ok "Hugging Face: sessione già autenticata"
  else
    warn "HF_TOKEN non impostato. Il repo è pubblico: provo il download anonimo."
  fi
}

download_model() {
  disk_preflight
  setup_hf
  hf_auth_if_needed

  log "Download/resume modello"
  "$HF_VENV/bin/hf" download \
    "$MODEL_REPO" \
    "$MODEL_FILE" \
    --local-dir "$MODEL_DIR"

  [[ -s "$MODEL_PATH" ]] || die "Download completato ma file modello non trovato: $MODEL_PATH"
  ok "Modello pronto: $(du -h "$MODEL_PATH" | awk '{print $1}')"
}

clone_or_update_llama() {
  if [[ -d "$LLAMA_DIR/.git" ]]; then
    log "Aggiorno llama.cpp"
    git -C "$LLAMA_DIR" pull --ff-only --depth 1
  else
    [[ ! -e "$LLAMA_DIR" ]] || die "$LLAMA_DIR esiste ma non è un repository Git."
    log "Clone llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi
  ok "llama.cpp: $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
}

build_llama() {
  cuda_preflight
  clone_or_update_llama

  log "Configuro build CUDA Release"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

  log "Compilo llama.cpp con $(nproc) job"
  cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"

  [[ -x "$LLAMA_DIR/build/bin/llama-cli" ]] || die "llama-cli non trovato dopo la build."
  [[ -x "$LLAMA_DIR/build/bin/llama-server" ]] || die "llama-server non trovato dopo la build."

  ok "Build completata"
}

sanity_check() {
  [[ -x "$LLAMA_DIR/build/bin/llama-cli" ]] || die "Esegui prima la build."
  [[ -x "$LLAMA_DIR/build/bin/llama-server" ]] || die "llama-server assente."
  [[ -s "$MODEL_PATH" ]] || die "Modello assente: $MODEL_PATH"

  printf "\n${C_BOLD}llama-cli --version${C_RESET}\n"
  "$LLAMA_DIR/build/bin/llama-cli" --version

  printf "\n${C_BOLD}llama-server --version${C_RESET}\n"
  "$LLAMA_DIR/build/bin/llama-server" --version || true

  printf "\n${C_BOLD}llama-server --list-devices${C_RESET}\n"
  "$LLAMA_DIR/build/bin/llama-server" --list-devices || true

  printf "\n${C_BOLD}nvidia-smi${C_RESET}\n"
  nvidia-smi

  ok "Sanity check completato"
}

write_runner() {
  mkdir -p "$LLAMA_DIR"

  cat >"$SERVER_RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

: "\${LLAMA_API_KEY:?LLAMA_API_KEY non impostata}"

cd "$LLAMA_DIR"

exec ./build/bin/llama-server \\
  -m "$MODEL_PATH" \\
  --alias qwen \\
  --host "$HOST" \\
  --port "$PORT" \\
  -ngl "$N_GPU_LAYERS" \\
  -c "$CTX" \\
  --rope-scaling yarn \\
  --rope-scale "$ROPE_SCALE" \\
  --yarn-orig-ctx "$YARN_ORIG_CTX" \\
  --override-kv qwen35.context_length=int:"$CTX" \\
  -fa on \\
  -ctk "$KV_CACHE_TYPE" \\
  -ctv "$KV_CACHE_TYPE" \\
  -b "$BATCH" \\
  -ub "$UBATCH" \\
  -t "$THREADS" \\
  -tb "$THREADS_BATCH" \\
  -np 1 \\
  --cont-batching \\
  --cache-prompt \\
  --cache-reuse 256 \\
  --spec-type ngram-mod \\
  --spec-ngram-mod-n-match 24 \\
  --spec-ngram-mod-n-min 48 \\
  --spec-ngram-mod-n-max 64 \\
  --backend-sampling \\
  --warmup \\
  --jinja \\
  --metrics \\
  --slots \\
  --threads-http "$HTTP_THREADS" \\
  --timeout 3600 \\
  --log-colors on \\
  --log-timestamps \\
  --log-prefix \\
  --verbose
EOF

  chmod 700 "$SERVER_RUNNER"
}

ensure_api_key() {
  if [[ -n "${LLAMA_API_KEY:-}" ]]; then
    return 0
  fi

  [[ -t 0 ]] || die "Imposta LLAMA_API_KEY nell'ambiente prima di usare 'start'."

  printf "LLAMA_API_KEY (input nascosto): "
  IFS= read -r -s LLAMA_API_KEY
  printf "\n"
  [[ -n "$LLAMA_API_KEY" ]] || die "API key vuota."
  export LLAMA_API_KEY
}

start_server() {
  have tmux || die "tmux non trovato."
  sanity_check
  ensure_api_key
  write_runner

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    warn "Sessione tmux '$TMUX_SESSION' già presente: la riavvio."
    tmux kill-session -t "$TMUX_SESSION"
  fi

  log "Creo sessione tmux: $TMUX_SESSION"
  tmux new-session -d -s "$TMUX_SESSION" -c "$LLAMA_DIR"

  # Keep the secret out of shell history and llama-server argv.
  tmux set-environment -t "$TMUX_SESSION" LLAMA_API_KEY "$LLAMA_API_KEY"
  tmux send-keys -t "$TMUX_SESSION" "exec ./run-qwen.sh" C-m

  sleep 1

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    ok "llama-server avviato in tmux '$TMUX_SESSION'"
    printf "Attach : tmux attach -t %q\n" "$TMUX_SESSION"
    printf "API    : http://127.0.0.1:%s/v1\n" "$PORT"
    printf "Health : curl -s http://127.0.0.1:%s/health\n" "$PORT"
  else
    die "La sessione tmux è terminata subito. Avvia '$SERVER_RUNNER' a mano per vedere l'errore."
  fi
}

status_server() {
  printf "${C_BOLD}TMUX${C_RESET}\n"
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    ok "Sessione '$TMUX_SESSION' attiva"
    tmux list-panes -t "$TMUX_SESSION" -F 'pane=#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}'
    printf "\nUltime righe:\n"
    tmux capture-pane -pt "$TMUX_SESSION" -S -30 || true
  else
    warn "Sessione '$TMUX_SESSION' non attiva"
  fi

  printf "\n${C_BOLD}GPU${C_RESET}\n"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true

  printf "\n${C_BOLD}HTTP health${C_RESET}\n"
  curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null || warn "Server non ancora healthy/non raggiungibile."
  printf "\n"
}

attach_server() {
  tmux has-session -t "$TMUX_SESSION" 2>/dev/null || die "Sessione '$TMUX_SESSION' non attiva."
  exec tmux attach -t "$TMUX_SESSION"
}

stop_server() {
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux kill-session -t "$TMUX_SESSION"
    ok "Sessione '$TMUX_SESSION' fermata."
  else
    warn "Sessione '$TMUX_SESSION' non esiste."
  fi
}

full_install() {
  header
  apt_setup
  disk_preflight
  cuda_preflight
  download_model
  build_llama
  start_server
  printf "\n"
  status_server
}

python_menu() {
  python3 - <<'PY'
import sys

items = [
    ("all",      "FULL install: APT + HF download + build CUDA + tmux"),
    ("deps",     "APT update/upgrade + dipendenze"),
    ("download", "Download/resume del GGUF da Hugging Face"),
    ("build",    "Clone/update + build CUDA llama.cpp"),
    ("check",    "Sanity check llama.cpp + NVIDIA"),
    ("start",    "Avvia/restart llama-server in tmux"),
    ("status",   "Stato tmux/GPU/health"),
    ("attach",   "Attach alla sessione tmux"),
    ("stop",     "Ferma llama-server"),
    ("quit",     "Esci"),
]

print("\033[1mQwen3.8 bootstrap\033[0m", file=sys.stderr)
for i, (_, label) in enumerate(items, 1):
    print(f"  \033[36m{i:>2}\033[0m  {label}", file=sys.stderr)

try:
    print("\nSelezione: ", end="", flush=True, file=sys.stderr)
    raw = input().strip()
except (EOFError, KeyboardInterrupt):
    print(file=sys.stderr)
    sys.exit(130)

if raw.isdigit() and 1 <= int(raw) <= len(items):
    print(items[int(raw)-1][0])
else:
    print(raw or "quit")
PY
}

bash_menu() {
  local choice
  cat >&2 <<'EOF'
1) FULL install: APT + HF download + build CUDA + tmux
2) APT update/upgrade + dipendenze
3) Download/resume GGUF
4) Build CUDA llama.cpp
5) Sanity check
6) Start tmux
7) Status
8) Attach
9) Stop
0) Quit
EOF
  read -r -p "Selezione: " choice </dev/tty
  case "$choice" in
    1) echo all ;;
    2) echo deps ;;
    3) echo download ;;
    4) echo build ;;
    5) echo check ;;
    6) echo start ;;
    7) echo status ;;
    8) echo attach ;;
    9) echo stop ;;
    *) echo quit ;;
  esac
}

usage() {
  cat <<EOF
Uso:
  $0 [all|deps|download|build|check|start|status|attach|stop]

Esempio non interattivo:
  export LLAMA_API_KEY='la-tua-key'
  $0 all

Variabili principali:
  WORKSPACE=$WORKSPACE
  CTX=$CTX
  N_GPU_LAYERS=$N_GPU_LAYERS
  KV_CACHE_TYPE=$KV_CACHE_TYPE
  PORT=$PORT
EOF
}

main() {
  local action="${1:-}"

  if [[ -z "$action" ]]; then
    header
    if have python3 && [[ -t 0 && -t 1 ]]; then
      action="$(python_menu | tail -n 1)"
    else
      action="$(bash_menu | tail -n 1)"
    fi
  fi

  case "$action" in
    all)      full_install ;;
    deps)     apt_setup ;;
    download) download_model ;;
    build)    build_llama ;;
    check)    sanity_check ;;
    start)    start_server ;;
    status)   status_server ;;
    attach)   attach_server ;;
    stop)     stop_server ;;
    quit|q|exit) exit 0 ;;
    help|-h|--help) usage ;;
    *) usage; die "Azione sconosciuta: $action" ;;
  esac
}

main "$@"
